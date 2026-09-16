#!/usr/bin/env python3
"""Mesure le plafond de frames d'un modele video SAM du serveur X-AnyLabeling.

Rejoue ce que fait le client : init de session avec toutes les frames, prompt
texte sur une frame, puis propagation en flux. Releve la crete VRAM (pilote) et
la crete RSS du serveur pendant toute la propagation, et rapporte la frame ou la
propagation s'arrete.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time

import requests

BASE = "http://127.0.0.1:8000"


def server_pid():
    out = subprocess.run(
        ["pgrep", "-f", "bin/x-anylabeling-server"],
        capture_output=True, text=True,
    ).stdout.split()
    return int(out[0]) if out else None


class Sampler(threading.Thread):
    """Echantillonne VRAM pilote et RSS serveur jusqu'a l'arret."""

    def __init__(self, pid, period=0.5):
        super().__init__(daemon=True)
        self.pid = pid
        self.period = period
        self.stop = threading.Event()
        self.peak_vram_mib = 0
        self.peak_rss_gib = 0.0
        self.samples = 0

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
            if self.pid:
                try:
                    with open(f"/proc/{self.pid}/status") as f:
                        for line in f:
                            if line.startswith("VmRSS:"):
                                kb = int(line.split()[1])
                                self.peak_rss_gib = max(
                                    self.peak_rss_gib, kb / 1024 / 1024)
                                break
                except Exception:
                    pass
            self.samples += 1
            self.stop.wait(self.period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--model", default="segment_anything_3_video")
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--prompt-frame", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="ne charger que les N premieres frames (0 = toutes)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--end-frame", type=int, default=None,
                    help="borner la propagation a cette frame incluse")
    ap.add_argument("--out", default=None, help="fichier JSON de resultat")
    args = ap.parse_args()

    files = sorted(
        os.path.join(args.frames_dir, f)
        for f in os.listdir(args.frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if args.limit:
        files = files[: args.limit]
    print(f"[i] {len(files)} frames depuis {args.frames_dir}", flush=True)

    pid = server_pid()
    print(f"[i] serveur pid={pid}", flush=True)

    t0 = time.time()
    frames_b64 = []
    for p in files:
        with open(p, "rb") as f:
            frames_b64.append(base64.b64encode(f.read()).decode("ascii"))
    payload_mib = sum(len(x) for x in frames_b64) / 2 ** 20
    print(f"[i] base64 pret : {payload_mib:.0f} Mio "
          f"({time.time() - t0:.1f}s)", flush=True)

    sampler = Sampler(pid)
    sampler.start()

    res = {
        "model": args.model,
        "prompt": args.prompt,
        "frames_dir": args.frames_dir,
        "n_frames": len(files),
    }

    try:
        t = time.time()
        r = requests.post(
            f"{BASE}/v1/video/init",
            json={"model": args.model, "frames": frames_b64,
                  "start_frame_index": 0},
            timeout=900,
        )
        r.raise_for_status()
        body = r.json()
        if not body.get("success", True) or "error" in body:
            print("[!] init KO :", json.dumps(body)[:500])
            return 1
        session_id = body["data"]["session_id"]
        res["init_s"] = round(time.time() - t, 2)
        print(f"[i] session {session_id} ({res['init_s']}s)", flush=True)
        del frames_b64

        t = time.time()
        r = requests.post(
            f"{BASE}/v1/video/prompt",
            json={"session_id": session_id, "model": args.model,
                  "text_prompt": args.prompt,
                  "frame_index": args.prompt_frame,
                  "params": {"conf_threshold": args.conf}},
            timeout=300,
        )
        r.raise_for_status()
        pb = r.json()
        shapes = (pb.get("data", {}) or {}).get("shapes", [])
        res["prompt_s"] = round(time.time() - t, 2)
        res["objets_frame_prompt"] = len(shapes)
        print(f"[i] prompt '{args.prompt}' -> {len(shapes)} objet(s) "
              f"sur la frame {args.prompt_frame} ({res['prompt_s']}s)",
              flush=True)

        t = time.time()
        max_frame = -1
        n_progress = 0
        completed = None
        last_print = 0.0
        with requests.post(
            f"{BASE}/v1/video/propagate/stream",
            json={"session_id": session_id, "model": args.model,
                  "end_frame": args.end_frame},
            stream=True, timeout=(30, 7200),
        ) as sr:
            sr.raise_for_status()
            for raw in sr.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                ev = json.loads(raw[6:])
                k = ev.get("type")
                if k == "progress":
                    n_progress += 1
                    max_frame = max(max_frame, ev.get("current_frame", -1))
                    now = time.time()
                    if now - last_print > 10:
                        el = now - t
                        print(f"    frame {max_frame} "
                              f"({n_progress} traitees, {el:.0f}s, "
                              f"{n_progress / max(el, 1e-9):.2f} fps, "
                              f"VRAM crete {sampler.peak_vram_mib} Mio)",
                              flush=True)
                        last_print = now
                elif k == "completed":
                    rs = ev.get("results") or {}
                    n_shapes = sum(len(v.get("masks") or []) for v in rs.values())
                    n_frames_avec = sum(
                        1 for v in rs.values() if (v.get("masks") or []))
                    completed = {
                        "stopped_early": ev.get("stopped_early"),
                        "frames_processed": ev.get("frames_processed"),
                        "n_frames_resultats": len(rs),
                        "total_shapes": n_shapes,
                        "frames_avec_detection": n_frames_avec,
                    }
                elif k == "error":
                    res["erreur_flux"] = ev.get("message")
                    print("[!] erreur flux :", ev.get("message"), flush=True)

        el = time.time() - t
        res.update(
            propagation_s=round(el, 1),
            frames_propagees=n_progress,
            derniere_frame=max_frame,
            fps=round(n_progress / el, 2) if el else None,
            completed=completed,
            crete_vram_mib=sampler.peak_vram_mib,
            crete_vram_gio=round(sampler.peak_vram_mib / 1024, 2),
            crete_rss_gio=round(sampler.peak_rss_gib, 2),
        )
    finally:
        sampler.stop.set()
        sampler.join(timeout=3)
        try:
            requests.post(
                f"{BASE}/v1/video/cleanup/{session_id}",
                params={"model": args.model}, timeout=60,
            )
            print("[i] session nettoyee", flush=True)
        except Exception:
            pass

    print("\n=== RESULTAT ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"[i] ecrit dans {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
