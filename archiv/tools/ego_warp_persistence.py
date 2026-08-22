#!/usr/bin/env python3
# =============================================================================
# ego_warp_persistence.py  —  Task 16d-Vorstufe: ego-gewarpte Persistenz-Baseline
# =============================================================================
# Die naive Persistenz (16c) kopiert F3 unveraendert -> enthaelt die triviale
# "Szene bewegt sich mit dem Ego"-Komponente. Diese Baseline warpt F3 mit der
# ECHTEN Ego-Bewegung (aus nuScenes ego2global) in das Ego-Frame von F_{3+k}
# und vergleicht dann -> haertere, ehrlichere Referenz.
#   mIoU_pers_ego(k) = IoU( warp(decode(F3), T_{3->3+k}) , decode(F_{3+k}) )
# Vergleich mit der naiven Persistenz + dem Modell (rollout_metrics) im Bericht.
#
# GEOMETRIE: BEV-Maske [n_cls,200,200], output_scope [-50,50]/0.5m. Ego-Pose je
# Frame (ego2global_translation + _rotation-Quaternion) aus der pkl, ueber token.
# Konvention (Achse/Yaw-Vorzeichen) ist EMPIRISCH validiert:
#   Gate 1: Selbst-Warp k=0 -> mIoU ~ 1.0
#   Gate 2: ego-gewarpt >= naiv (sonst AXIS_SWAP / YAW_SIGN drehen)
#
# USAGE (PLAIN, kein --export):
#   python -u ego_warp_persistence.py --config config_rollout.yaml \
#       --phase cell --k 4 --max_windows 300 --out predictions/task16c/ego_pers.json
# =============================================================================
import argparse, json, pickle, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import train_linux as T

# --- Konvention (per Smoke-Sweep bestaetigt: yaw=+1 swap=1 inv=0, Gate1=1.0,
#     Gate2 ego>=naiv fuer alle k; Achsentausch korrigiert x/y<->Grid der BEV-Maske) ---
AXIS_SWAP = True      # (x,y) -> (A1,A0)
YAW_SIGN  = 1.0
INVERT    = False     # W = inv(Ti)@Tj
EXTENT    = 50.0      # +/- Meter (output_scope)


def parse_args():
    p = argparse.ArgumentParser(description="ego-gewarpte Persistenz-Baseline (16d-Vorstufe).")
    p.add_argument("--config", required=True)
    p.add_argument("--phase",  default="cell", choices=["frame", "cell"])
    p.add_argument("--k",      type=int, default=4)
    p.add_argument("--max_windows", type=int, default=None)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--out",    default="predictions/task16c/ego_pers.json")
    p.add_argument("--smoke",  action="store_true", help="wenige Fenster + Validierungs-Gates drucken.")
    return p.parse_args()


def quat_to_yaw(q):
    """q = [w,x,y,z] -> yaw (Rotation um z)."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def load_poses(pkl_path):
    """token -> (tx, ty, yaw) aus ego2global."""
    infos = pickle.load(open(pkl_path, "rb"))["infos"]
    poses = {}
    for i in infos:
        t = i["ego2global_translation"]; r = i["ego2global_rotation"]
        poses[i["token"]] = (float(t[0]), float(t[1]), float(quat_to_yaw(r)))
    return poses


def rigid2d(dx, dy, dyaw):
    c, s = np.cos(dyaw), np.sin(dyaw)
    return np.array([[c, -s, dx], [s, c, dy], [0, 0, 1.0]])


def warp_mask(mask_t, pose_i, pose_j, device, yaw_sign=YAW_SIGN, axis_swap=AXIS_SWAP, invert=False):
    """Warpt F_i-Maske ins Ego-Frame von F_j. mask_t: [n_cls,200,200] float(0/1).
    Konvention parametrierbar (yaw_sign/axis_swap/invert) fuer den Smoke-Sweep."""
    xi, yi, yi_yaw = pose_i
    xj, yj, yj_yaw = pose_j
    Ti = rigid2d(xi, yi, yaw_sign * yi_yaw)
    Tj = rigid2d(xj, yj, yaw_sign * yj_yaw)
    W = (np.linalg.inv(Tj) @ Ti) if invert else (np.linalg.inv(Ti) @ Tj)
    n_cls, H, Wd = mask_t.shape
    lin = torch.linspace(-EXTENT + EXTENT / H, EXTENT - EXTENT / H, H, device=device)
    a0, a1 = torch.meshgrid(lin, lin, indexing="ij")
    mx = a1 if axis_swap else a0
    my = a0 if axis_swap else a1
    ones = torch.ones_like(mx)
    pj = torch.stack([mx.reshape(-1), my.reshape(-1), ones.reshape(-1)], dim=0)
    pi = torch.tensor(W, dtype=torch.float32, device=device) @ pj
    sx, sy = pi[0], pi[1]
    src_a0 = sy if axis_swap else sx
    src_a1 = sx if axis_swap else sy
    gx = src_a1 / EXTENT
    gy = src_a0 / EXTENT
    grid = torch.stack([gx, gy], dim=-1).reshape(1, H, Wd, 2)
    out = F.grid_sample(mask_t.unsqueeze(0), grid, mode="nearest",
                        padding_mode="zeros", align_corners=False)
    return out[0] > 0.5


@torch.no_grad()
def main():
    args = parse_args()
    cfg = T.load_config(args.config, args.phase)
    T.set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not T.SEG_DECODER_AVAILABLE:
        raise SystemExit("[ego] seg_decoder nicht importierbar.")

    n_input = cfg["model"].get("n_frames", 3)
    win_len = n_input + args.k
    seg = T.build_seg_decoder(cfg.get("decode_validation", {}).get("seg_checkpoint"), device, verbose=True)
    _, val_loader, _ = T.build_dataloaders(cfg)
    dataset = val_loader.dataset
    poses = load_poses(cfg["data"]["val_sources"][0]["pkl_path"])

    windows = []
    for scene in dataset._all_scenes:
        for i in range(len(scene) - win_len + 1):
            w = scene[i:i + win_len]
            if all(f["token"] in poses for f in w):
                windows.append(w)
    if args.max_windows and args.max_windows < len(windows):
        rng = np.random.default_rng(args.seed)
        windows = [windows[j] for j in sorted(rng.choice(len(windows), args.max_windows, replace=False).tolist())]

    n = len(windows); t0 = time.time()

    # --- Vorab: decode alle noetigen Masken je Fenster EINMAL (Warp ist billig) ---
    cache = []   # (m3_bool_np, {k: mj_bool_np}, {k: (pose3, posej)}, m3_float_t)
    for wi, w in enumerate(windows):
        lat = [torch.from_numpy(dataset._load_latent(f)).float().to(device) for f in w]
        f3 = w[n_input - 1]
        m3, _ = seg.latent_to_mask(lat[n_input - 1], device=device)
        m3f = (m3 if torch.is_tensor(m3) else torch.as_tensor(np.asarray(m3))).float().to(device)
        mjs, ps = {}, {}
        for k in range(1, args.k + 1):
            fj = w[n_input - 1 + k]
            mj, _ = seg.latent_to_mask(lat[n_input - 1 + k], device=device)
            mjs[k] = (mj.cpu().numpy() if torch.is_tensor(mj) else np.asarray(mj)).astype(bool)
            ps[k]  = (poses[f3["token"]], poses[fj["token"]])
        cache.append(((m3f > 0.5).cpu().numpy(), mjs, ps, m3f))
        if (wi + 1) % 25 == 0 or wi + 1 == n:
            print(f"\r  decode {wi+1}/{n}  {(wi+1)/(time.time()-t0):.1f} w/s", end="", flush=True)
    print()

    def mean_ego(ys, ax, inv):
        out = {k: [] for k in range(1, args.k + 1)}
        for (m3b, mjs, ps, m3f) in cache:
            for k in range(1, args.k + 1):
                wm = warp_mask(m3f, ps[k][0], ps[k][1], device, ys, ax, inv).cpu().numpy()
                out[k].append(T.compute_iou(wm, mjs[k], T.MAP_CLASSES)["mIoU"])
        return {k: float(np.mean(out[k])) for k in out}

    naive = {k: float(np.mean([T.compute_iou(m3b, mjs[k], T.MAP_CLASSES)["mIoU"]
                               for (m3b, mjs, _, _) in cache])) for k in range(1, args.k + 1)}
    # Gate 1: Selbst-Warp (Identitaet)
    self0 = np.mean([T.compute_iou(
        warp_mask(m3f, ps[1][0], ps[1][0], device, YAW_SIGN, AXIS_SWAP, INVERT).cpu().numpy(),
        m3b, T.MAP_CLASSES)["mIoU"] for (m3b, _, ps, m3f) in cache[:20]])
    print(f"[gate1] Selbst-Warp mIoU = {self0:.4f}  (soll ~1.0)")
    print(f"[naiv]  " + "  ".join(f"k{k}={naive[k]:.4f}" for k in naive))

    if args.smoke:
        print("[sweep] Konventionen (ego-gewarpt bei k=1 / k=4, soll > naiv):")
        best = None
        for ys in (1.0, -1.0):
            for ax in (False, True):
                for inv in (False, True):
                    e = mean_ego(ys, ax, inv)
                    ok = all(e[k] >= naive[k] - 1e-6 for k in e)
                    tag = "  <-- OK" if ok else ""
                    print(f"  yaw={ys:+.0f} swap={int(ax)} inv={int(inv)}: "
                          f"k1={e[1]:.4f} k4={e[args.k]:.4f}{tag}")
                    score = e[1] - naive[1]
                    if best is None or score > best[0]:
                        best = (score, ys, ax, inv, e)
        print(f"[best]  yaw={best[1]:+.0f} swap={int(best[2])} inv={int(best[3])}  "
              f"(k1 ego={best[4][1]:.4f} vs naiv={naive[1]:.4f})")
        return

    egow = mean_ego(YAW_SIGN, AXIS_SWAP, INVERT)
    curve = {str(k): {"pers_naiv": round(naive[k], 4), "pers_ego": round(egow[k], 4),
                      "n": n} for k in range(1, args.k + 1)}
    for k in range(1, args.k + 1):
        print(f"  k={k}: naiv={naive[k]:.4f}  ego-gewarpt={egow[k]:.4f}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_windows": n, "K": args.k, "curve": curve,
                               "gate1_selfwarp": round(float(self0), 4),
                               "AXIS_SWAP": AXIS_SWAP, "YAW_SIGN": YAW_SIGN, "INVERT": INVERT},
                              indent=2))
    print(f"[ego] -> {out}")


if __name__ == "__main__":
    main()
