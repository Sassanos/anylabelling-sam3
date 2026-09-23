#!/usr/bin/env python3
"""Annote les frames d'une vidéo Anafi via le serveur X-AnyLabeling-Server (SAM 3).

Chemin image, frame par frame, à la cadence native du flux : la vidéo est décodée
en streaming (PyAV), AUCUNE frame n'est écrite sur disque (quotas d'inodes du
cluster), chaque frame est encodée JPEG en mémoire et envoyée au serveur
(POST /v1/predict, image en base64), les annotations sont écrites en JSONL.

Sélection des frames : l'index samples_default.csv produit par build_index.py
indique quelles frames du DefaultVideo sont du vrai RGB (mode=RGB, valid=1).
Pendant les rafales thermiques, le DefaultVideo est un rendu d'affichage gelé
qu'il ne faut pas annoter ; --all-frames court-circuite ce filtre (déconseillé).

Découpage et reprise : un tronçon est une plage [--start, --end) de sample_index.
La sortie <out-dir>/<flight>_chunk_<start>_<end>.jsonl est écrite
incrémentalement (append + fsync périodique) et le marqueur .done n'est posé
qu'à la fin. Un relancement reprend après la dernière ligne valide ; les frames
marquées "error" sont reprises avec --retry-errors.

Une ligne JSONL = une frame :
  {"flight": "0000011", "sample_index": 123, "dts_s": 4.1, "dts_ticks": 123123,
   "stem": "0000011_rgb_4100", "utc_us": ..., "width": 3840, "height": 2160,
   "shapes": [...], "t_inf_s": 3.2}
   ou {"flight": ..., "sample_index": ..., "error": "message", ...}

Les shapes sont au format X-AnyLabeling (rectangles, comme preannotate_sam3.py)
: label fin mappé via sam3_prompt_map, score, attributes source/prompt/coarse.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path

# sam3_prompt_map et coarse_of de taxonomy.yaml (annotation/taxonomy.yaml du
# pipeline AnafiUKR). En dur ici pour n'exiger aucun venv côté client ;
# --taxonomy surcharge le mapping.
DEFAULT_PROMPT_MAP = {
    "person": "person", "pedestrian": "person", "soldier": "person",
    "car": "car", "van": "van", "truck": "truck", "bus": "bus",
    "motorcycle": "motorcycle", "bicycle": "bicycle", "tractor": "agri_vehicle",
    "tank": "mil_tank", "armored vehicle": "mil_apc_ifv",
    "military truck": "mil_truck", "military vehicle": "mil_other",
    "vehicle": "vehicle_unknown",
}
DEFAULT_COARSE = {
    "person": "person",
    "car": "vehicle", "van": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "motorcycle": "vehicle", "bicycle": "vehicle", "agri_vehicle": "vehicle",
    "mil_tank": "vehicle", "mil_apc_ifv": "vehicle", "mil_truck": "vehicle",
    "mil_other": "vehicle", "vehicle_unknown": "vehicle",
}
DEFAULT_PROMPT = "person. car. van. truck. bus. motorcycle. tank. armored vehicle. military truck"


def load_taxonomy(path: Path | None):
    if path is None:
        return dict(DEFAULT_PROMPT_MAP), dict(DEFAULT_COARSE)
    import yaml  # optionnel
    tax = yaml.safe_load(path.read_text(encoding="utf-8"))
    pmap = {k.lower(): v for k, v in tax["sam3_prompt_map"].items()}
    coarse = {k: v["coarse"] for k, v in tax["fine_labels"].items()}
    return pmap, coarse


def load_index(index_csv: Path):
    """dts_ticks -> ligne d'index (sample_index, dts_s, valid, mode, utc_us...)."""
    with open(index_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_tick = {}
    for r in rows:
        by_tick[int(r["dts_ticks"])] = r
    return rows, by_tick


def read_done(jsonl: Path, retry_errors: bool):
    """(done={sample_index}, offset_octets_fin_de_derniere_ligne_valide)."""
    done: dict[int, str] = {}
    good_end = 0
    if not jsonl.exists():
        return done, good_end
    with open(jsonl, "rb") as f:
        for line in f:
            end = f.tell()
            try:
                rec = json.loads(line)
            except Exception:
                break  # ligne tronquée (job tué en pleine écriture)
            good_end = end
            idx = rec.get("sample_index")
            if idx is None:
                continue
            if retry_errors and rec.get("error"):
                continue  # à refaire
            done[idx] = rec.get("error") or ""
    return done, good_end


def shape_of(s, pmap, coarse, w, h, uncertain_below):
    """Shape X-AnyLabeling depuis la réponse du serveur (comme preannotate_sam3)."""
    pts = s.get("points") or []
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
    x2, y2 = min(float(w), max(xs)), min(float(h), max(ys))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    raw = str(s.get("label", "")).strip().lower()
    fine = pmap.get(raw, raw or "vehicle_unknown")
    score = s.get("score")
    return {
        "label": fine,
        "score": None if score is None else float(score),
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {
            "source": "sam3",
            "prompt": raw,
            "coarse": coarse.get(fine, ""),
            "uncertain": bool(score is not None and score < uncertain_below),
        },
        "kie_linking": [],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="MP4 source (piste DefaultVideo)")
    ap.add_argument("--index", help="samples_default.csv (build_index.py)")
    ap.add_argument("--flight", help="identifiant du vol (ex. 0000011)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", type=int, default=0, help="premier sample_index du tronçon (inclus)")
    ap.add_argument("--end", type=int, default=None, help="dernier sample_index (exclu) ; défaut: fin")
    ap.add_argument("--tasks-file", help="tasks.json (genere-taches.py) : la tâche --task-id "
                                          "fournit video/index/flight/start/end")
    ap.add_argument("--task-id", type=int, help="index dans tasks.json (ou SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--print-only", action="store_true",
                    help="imprimer l'état du tronçon (todo|done) et sortir, sans serveur")
    ap.add_argument("--all-frames", action="store_true",
                    help="ignorer le filtre mode=RGB/valid (déconseillé : rendu gelé en mode IR)")
    ap.add_argument("--retry-errors", action="store_true", help="refaire les frames marquées error")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="segment_anything_3_tiled")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--uncertain-below", type=float, default=0.5)
    ap.add_argument("--tile-size", type=int, default=1008)
    ap.add_argument("--tile-overlap", type=float, default=0.2)
    ap.add_argument("--tile-match", type=float, default=0.5)
    ap.add_argument("--no-full-image", action="store_true",
                    help="tuilé : pas de passe plein cadre en plus des tuiles")
    ap.add_argument("--stream-index", type=int, default=0,
                    help="piste vidéo à décoder (0 = DefaultVideo)")
    ap.add_argument("--jpeg-q", type=int, default=95)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--taxonomy", default=None, help="taxonomy.yaml (surcharge le mapping en dur)")
    ap.add_argument("--max-errors", type=int, default=20,
                    help="abandon si autant de frames échouent (0 = jamais)")
    args = ap.parse_args()

    import requests
    import av
    import cv2
    import numpy as np  # noqa: F401  (requis par cv2 en interne, import explicite)

    if args.tasks_file:
        tasks = json.loads(Path(args.tasks_file).read_text(encoding="utf-8"))
        if args.task_id is None or not (0 <= args.task_id < len(tasks)):
            print(f"[!] --task-id requis et dans [0, {len(tasks)})", file=sys.stderr)
            return 2
        t = tasks[args.task_id]
        args.video, args.index = t["video"], t["index"]
        args.flight, args.start, args.end = t["flight"], t["start"], t["end"]

    if args.print_only:
        # État du tronçon sans serveur ni décodage (utilisé par le sbatch pour
        # ne pas réserver un GPU pour un tronçon déjà fini).
        if not args.flight or args.end is None:
            print("[!] --print-only exige --tasks-file/--task-id", file=sys.stderr)
            return 2
        flight_dir = Path(args.out_dir) / args.flight
        jsonl = flight_dir / f"{args.flight}_chunk_{args.start:07d}_{args.end:07d}.jsonl"
        st = "done" if jsonl.with_suffix(".done").exists() else "todo"
        print(f"{st} {args.flight} {args.start} {args.end}")
        return 0

    pmap, coarse = load_taxonomy(Path(args.taxonomy) if args.taxonomy else None)

    rows, by_tick = load_index(Path(args.index))
    rows = [r for r in rows if r["stream"] == "default"]
    if not rows:
        print("[!] index sans ligne stream=default", file=sys.stderr)
        return 2
    timescale = int(rows[0]["timescale"])

    todo = []
    for r in rows:
        i = int(r["sample_index"])
        if i < args.start:
            continue
        if args.end is not None and i >= args.end:
            continue
        if not args.all_frames and (r.get("mode") != "RGB" or r.get("valid") != "1"):
            continue
        todo.append(int(r["dts_ticks"]))
    if not todo:
        print(f"[i] {args.flight} [{args.start},{args.end or 'fin'}) : aucune frame à annoter")
        return 0

    out_dir = Path(args.out_dir) / args.flight
    out_dir.mkdir(parents=True, exist_ok=True)
    end = args.end if args.end is not None else int(rows[-1]["sample_index"]) + 1
    jsonl = out_dir / f"{args.flight}_chunk_{args.start:07d}_{end:07d}.jsonl"
    done_marker = jsonl.with_suffix(".done")

    if done_marker.exists():
        print(f"[i] {jsonl.name} déjà terminé (.done présent)")
        return 0
    done, good_end = read_done(jsonl, args.retry_errors)
    pending = [t for t in todo if int(by_tick[t]["sample_index"]) not in done]
    if not pending:
        done_marker.write_text("ok\n", encoding="utf-8")
        print(f"[i] {jsonl.name} : rien à faire, marqueur .done posé")
        return 0

    print(f"[i] {args.flight} tronçon [{args.start}, {end}) : {len(pending)} frames à faire "
          f"({len(done)} déjà faites) -> {jsonl.name}", flush=True)

    sess = requests.Session()
    sess.get(f"{args.server}/health", timeout=10).raise_for_status()

    params = {
        "text_prompt": args.prompt,
        "conf_threshold": args.conf,
        "show_boxes": True,
        "show_masks": False,
        "epsilon_factor": 0.001,
    }
    if args.model == "segment_anything_3_tiled":
        params.update({
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "tile_match_threshold": args.tile_match,
            "include_full_image": not args.no_full_image,
        })

    container = av.open(args.video)
    streams = container.streams.video
    if args.stream_index >= len(streams):
        print(f"[!] piste vidéo {args.stream_index} absente ({len(streams)} pistes)",
              file=sys.stderr)
        return 2
    stream = streams[args.stream_index]
    if stream.time_base.denominator != timescale or stream.time_base.numerator != 1:
        print(f"[!] time_base inattendu {stream.time_base} (timescale index {timescale})",
              file=sys.stderr)
        return 2

    def to_tick(pts):
        return int(pts)  # time_base = 1/timescale => pts == dts_ticks

    # Seek au plus proche keyframe AVANT la première frame à faire, puis on
    # avance en décodant jusqu'à dépasser la dernière.
    first = pending[0]
    container.seek(first, stream=stream, backward=True, any_frame=False)
    pending_set = set(pending)
    last = pending[-1]
    saw_last = False

    out = open(jsonl, "ab")
    if good_end:
        out.truncate(good_end)
    out.flush()
    os.fsync(out.fileno())

    t0 = time.time()
    n_done = n_shapes = n_err = 0
    last_print = 0.0
    backoffs = (5, 15, 45)
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                tick = to_tick(frame.pts)
                if tick in pending_set:
                    row = by_tick[tick]
                    img = frame.to_ndarray(format="bgr24")
                    h, w = img.shape[:2]
                    ok, buf = cv2.imencode(".jpg", img,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_q])
                    if not ok:
                        rec_err = "erreur encodage jpeg"
                    else:
                        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                        t_inf = time.time()
                        rec_err = None
                        body = None
                        for attempt in range(args.retries):
                            try:
                                r = sess.post(
                                    f"{args.server}/v1/predict",
                                    json={"model": args.model,
                                          "image": f"data:image/jpeg;base64,{b64}",
                                          "params": params},
                                    timeout=args.timeout)
                                r.raise_for_status()
                                body = r.json()
                                if body.get("error"):
                                    rec_err = str(body)[:300]
                                    body = None
                                break
                            except Exception as e:
                                rec_err = str(e)[:300]
                                if attempt < args.retries - 1:
                                    time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
                        t_inf = time.time() - t_inf
                    if rec_err is not None:
                        n_err += 1
                        print(f"  !! sample {row['sample_index']} : {rec_err}", flush=True)
                        rec = {"flight": args.flight,
                               "sample_index": int(row["sample_index"]),
                               "dts_ticks": tick,
                               "error": rec_err}
                    else:
                        shapes = []
                        for s in (body.get("data") or {}).get("shapes") or []:
                            sh = shape_of(s, pmap, coarse, w, h, args.uncertain_below)
                            if sh is not None:
                                shapes.append(sh)
                        rec = {"flight": args.flight,
                               "sample_index": int(row["sample_index"]),
                               "dts_ticks": tick,
                               "dts_s": float(row.get("dts_s") or 0),
                               "utc_us": row.get("utc_us") or None,
                               "stem": f"{args.flight}_rgb_{tick * 1000 // timescale}",
                               "width": w, "height": h,
                               "shapes": shapes,
                               "t_inf_s": round(t_inf, 3)}
                        n_shapes += len(shapes)
                    n_done += 1
                    out.write(json.dumps(rec, ensure_ascii=False).encode("utf-8") + b"\n")
                    if n_done % 25 == 0:
                        out.flush()
                        os.fsync(out.fileno())
                    now = time.time()
                    if now - last_print > 30:
                        el = now - t0
                        print(f"  {n_done}/{len(pending)} | {el:.0f}s | "
                              f"{n_done / el:.2f} frames/s | {n_shapes} shapes | "
                              f"{n_err} err", flush=True)
                        last_print = now
                    if n_err and args.max_errors and n_err >= args.max_errors:
                        print("[!] trop d'erreurs, abandon du tronçon", file=sys.stderr)
                        raise SystemExit(3)
                if tick == last:
                    saw_last = True
                    break
            if saw_last:
                break
    finally:
        out.flush()
        os.fsync(out.fileno())
        out.close()
        container.close()

    if not saw_last:
        print(f"[!] fin de flux atteinte avant la dernière frame du tronçon "
              f"(dernière demandée : dts {last})", file=sys.stderr)
        return 3

    tmp = done_marker.with_suffix(".done.tmp")
    tmp.write_text(
        json.dumps({"flight": args.flight, "start": args.start, "end": end,
                    "frames": len(pending), "shapes": n_shapes, "errors": n_err,
                    "model": args.model, "prompt": args.prompt, "conf": args.conf,
                    "tile_size": args.tile_size if args.model.endswith("_tiled") else None})
        + "\n", encoding="utf-8")
    os.replace(tmp, done_marker)

    el = time.time() - t0
    print(f"=== TRONCON FINI === {n_done} frames, {n_shapes} shapes, {n_err} erreurs, "
          f"{el:.0f}s ({n_done / el:.2f} frames/s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
