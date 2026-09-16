#!/usr/bin/env python3
"""Inférence SAM 3.1 (Object Multiplex) sur un dossier d'images.

    .venv/bin/python run_sam31.py --frames "/chemin/*.jpg" --prompt car
    .venv/bin/python run_sam31.py --frames dossier/ --prompt car --propagate

Deux limites mesurées sur RTX 4000 Ada (12 Go) :

- Une seule classe par prompt. SAM 3.1 ne découpe pas "car. person" en deux
  concepts (vérifié : 0 objet). Pour du multi-classes, boucler sur les classes
  avec un reset_session entre chaque, comme le fait le patch X-AnyLabeling.
- ~6 frames par propagation avec 10 objets suivis (8,1 Go de crête) ; 10 frames
  saturent. L'état de tracking ne peut pas être délesté vers le CPU (l'init_state
  du multiplex n'accepte pas offload_state_to_cpu), et multiplex_count est figé
  à 16 par le checkpoint. Découper les longues séquences en tronçons.
"""
import argparse, glob, os, time, uuid
import torch
from PIL import Image
from sam3.model_builder import build_sam3_multiplex_video_predictor

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "X-AnyLabeling-Server", "SAM3.1", "sam3.1_multiplex.pt")

def open_session(predictor, frames, offload_video=True):
    """Crée une session.

    On n'utilise pas predictor.handle_request(type="start_session") : la
    start_session héritée passe offload_state_to_cpu, que l'init_state du
    modèle multiplex n'accepte pas (bug amont facebookresearch/sam3).
    """
    state = predictor.model.init_state(resource_path=frames,
                                       offload_video_to_cpu=offload_video,
                                       async_loading_frames=False)
    sid = str(uuid.uuid4())
    predictor._all_inference_states[sid] = dict(
        state=state, session_id=sid, start_time=time.time(), last_use_time=time.time())
    return sid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="dossier ou motif glob")
    ap.add_argument("--prompt", required=True, help="une classe, ex: car")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--limit", type=int, default=0, help="nb max de frames")
    ap.add_argument("--propagate", action="store_true", help="propager sur toute la séquence")
    ap.add_argument("--repeat", type=int, default=1, help="répéter la séquence (mesure de débit)")
    ap.add_argument("--no-offload", action="store_true",
                    help="garder les frames en VRAM (plus rapide, OOM au-delà de ~20 frames sur 12 Go)")
    a = ap.parse_args()

    pattern = os.path.join(a.frames, "*") if os.path.isdir(a.frames) else a.frames
    files = sorted(f for f in glob.glob(pattern)
                   if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"))
    if a.limit: files = files[:a.limit]
    if not files: raise SystemExit(f"aucune image pour {pattern}")
    frames = [Image.open(f) for f in files] * a.repeat
    print(f"{len(frames)} frames | GPU {torch.cuda.get_device_name(0)}")

    # FlashAttention 3 exige sm_90 (Hopper) ; sur Ada il faut use_fa3=False.
    use_fa3 = torch.cuda.get_device_capability(0)[0] >= 9
    t0 = time.time()
    # multiplex_count est figé à 16 : les poids encodent 16 slots (iou_token,
    # no_obj_embed_spatial...). Toute autre valeur casse le load_state_dict.
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=a.checkpoint, use_fa3=use_fa3)
    print(f"modèle prêt en {time.time()-t0:.1f}s (use_fa3={use_fa3})")

    sid = open_session(predictor, frames, offload_video=not a.no_offload)
    t0 = time.time()
    out = predictor.handle_request(request=dict(
        type="add_prompt", session_id=sid, frame_index=0, text=a.prompt))["outputs"]
    n = len(list(out.get("out_obj_ids", [])))
    print(f"prompt '{a.prompt}' sur la frame 0 : {n} objets en {time.time()-t0:.2f}s")

    if a.propagate:
        t0 = time.time(); nf = 0
        for _ in predictor.handle_stream_request(
                request=dict(type="propagate_in_video", session_id=sid)):
            nf += 1
        dt = time.time() - t0
        print(f"propagation : {nf} frames en {dt:.1f}s -> {nf/dt:.1f} fps")
    print(f"VRAM crête : {torch.cuda.max_memory_allocated()/2**30:.2f} Go")

if __name__ == "__main__":
    main()
