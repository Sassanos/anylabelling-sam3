#!/usr/bin/env python3
"""Mesure le chemin IMAGE de SAM 3 frame par frame : debit et detections.

Envoie chaque frame en requete /v1/predict independante, comme le ferait le mode
batch du client, et releve debit, latences, crete VRAM/RSS et nombre de formes.
"""
import argparse
import base64
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import requests

BASE = "http://127.0.0.1:8000"


def server_pid():
    out = subprocess.run(["pgrep", "-f", "bin/x-anylabeling-server"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


class Sampler(threading.Thread):
    def __init__(self, pid, period=0.5):
        super().__init__(daemon=True)
        self.pid, self.period = pid, period
        self.stop = threading.Event()
        self.peak_vram_mib = 0
        self.peak_rss_gib = 0.0

    def run(self):
        while not self.stop.is_set():
            try:
                v = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()[0]
                self.peak_vram_mib = max(self.peak_vram_mib, int(v))
            except Exception:
                pass
            try:
                with open(f"/proc/{self.pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            self.peak_rss_gib = max(
                                self.peak_rss_gib,
                                int(line.split()[1]) / 1024 / 1024)
                            break
            except Exception:
                pass
            self.stop.wait(self.period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--model", default="segment_anything_3")
    ap.add_argument("--prompt", default="Person")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--show-masks", action="store_true",
                    help="produire des polygones au lieu de boites")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-json", default=None, metavar="DIR",
                    help="ecrire les annotations au format X-AnyLabeling dans "
                         "ce dossier (jamais celui des frames sources)")
    args = ap.parse_args()

    files = sorted(
        os.path.join(args.frames_dir, f)
        for f in os.listdir(args.frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if args.limit:
        files = files[: args.limit]

    pid = server_pid()
    print(f"[i] {len(files)} frames | modele {args.model} | prompt "
          f"'{args.prompt}' | conf {args.conf} | show_masks={args.show_masks} "
          f"| serveur pid={pid}", flush=True)

    if args.write_json:
        if os.path.abspath(args.write_json) == os.path.abspath(args.frames_dir):
            print("[!] --write-json ne doit pas viser le dossier des frames "
                  "sources : il ecraserait les annotations existantes.",
                  file=sys.stderr)
            return 2
        os.makedirs(args.write_json, exist_ok=True)

    sampler = Sampler(pid)
    sampler.start()

    sess = requests.Session()
    params = {
        "text_prompt": args.prompt,
        "conf_threshold": args.conf,
        "show_boxes": not args.show_masks,
        "show_masks": args.show_masks,
        "epsilon_factor": 0.001,
    }

    lat = []
    total_shapes = 0
    frames_avec = 0
    par_frame = {}
    erreurs = 0
    t0 = time.time()
    last_print = 0.0

    try:
        for i, p in enumerate(files):
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            t = time.time()
            try:
                r = sess.post(
                    f"{BASE}/v1/predict",
                    json={"model": args.model,
                          "image": f"data:image/jpeg;base64,{b64}",
                          "params": params},
                    timeout=300,
                )
                r.raise_for_status()
                body = r.json()
            except Exception as e:
                erreurs += 1
                if erreurs <= 3:
                    print(f"[!] frame {i} : {e}", flush=True)
                continue
            lat.append(time.time() - t)
            shapes = ((body.get("data") or {}).get("shapes")) or []
            if body.get("error"):
                erreurs += 1
                if erreurs <= 3:
                    print("[!]", json.dumps(body)[:300], flush=True)
                continue
            if args.write_json:
                import cv2  # seulement pour lire les dimensions
                im = cv2.imread(p)
                h, w = (im.shape[:2] if im is not None else (0, 0))
                stem = os.path.splitext(os.path.basename(p))[0]
                with open(os.path.join(args.write_json, stem + ".json"),
                          "w", encoding="utf-8") as jf:
                    json.dump({
                        "version": "3.2.4",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": os.path.basename(p),
                        "imageData": None,
                        "imageHeight": h,
                        "imageWidth": w,
                        "description": "",
                    }, jf, ensure_ascii=False, indent=2)
            n = len(shapes)
            total_shapes += n
            if n:
                frames_avec += 1
                par_frame[i] = n
            now = time.time()
            if now - last_print > 15:
                el = now - t0
                print(f"    {i + 1}/{len(files)} | {el:.0f}s | "
                      f"{(i + 1) / el:.2f} fps | {total_shapes} formes | "
                      f"VRAM crete {sampler.peak_vram_mib} Mio", flush=True)
                last_print = now
    finally:
        sampler.stop.set()
        sampler.join(timeout=3)

    el = time.time() - t0
    res = {
        "modele": args.model,
        "prompt": args.prompt,
        "conf_threshold": args.conf,
        "show_masks": args.show_masks,
        "n_frames": len(files),
        "erreurs": erreurs,
        "duree_s": round(el, 1),
        "fps": round(len(lat) / el, 2) if el else None,
        "latence_ms_mediane": round(statistics.median(lat) * 1000, 1) if lat else None,
        "latence_ms_p90": round(
            statistics.quantiles(lat, n=10)[8] * 1000, 1) if len(lat) > 10 else None,
        "latence_ms_max": round(max(lat) * 1000, 1) if lat else None,
        "total_shapes": total_shapes,
        "frames_avec_detection": frames_avec,
        "instances_par_frame_detectee": round(
            total_shapes / frames_avec, 2) if frames_avec else 0,
        "premiere_frame_detectee": min(par_frame) if par_frame else None,
        "derniere_frame_detectee": max(par_frame) if par_frame else None,
        "max_objets_sur_une_frame": max(par_frame.values()) if par_frame else 0,
        "crete_vram_mib": sampler.peak_vram_mib,
        "crete_vram_gio": round(sampler.peak_vram_mib / 1024, 2),
        "crete_rss_gio": round(sampler.peak_rss_gib, 2),
    }
    print("\n=== RESULTAT ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"resume": res, "par_frame": par_frame}, f,
                      indent=2, ensure_ascii=False)
        print(f"[i] ecrit dans {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
