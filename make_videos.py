#!/usr/bin/env python3
"""Render per-stem 2-panel mp4s from sora2-gentraj run output.

Left panel: sora video (RGB) with rainbow-colored 2D track trails overlaid.
Right panel: matplotlib 3D scatter+trail of the lifted 3D tracks, same colors.

Both panels share the same per-track rainbow colormap so a noisy 2D track on
the left maps to a noisy 3D trail on the right.

Source per stem (from `_GenTraj/sora2/runs/<dataset>/<stem>/`):
  rgb.mp4              ViPE-resampled RGB at 480p
  tracks_2d.npz        tracks (T, N, 2) px + visibility (T, N) bool + dim (H, W)
  tracks_3d.npz        points_3d (N, T, 3) + visibility (N, T, 1)

Usage:
  python make_videos.py --stems davis:bear worldtrack:01f258...
  python make_videos.py --all                # render every .done stem
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers projection

REPO = Path(__file__).resolve().parent
RUNS_ROOT = Path("/weka/prior-default/jianingz/home/project/_GenTraj/sora2/runs")
OUT_DIR = REPO / "videos"
TMP_DIR = REPO / "tmp_frames"


def colors_for(n: int) -> np.ndarray:
    cmap = plt.get_cmap("hsv")
    return np.stack([np.array(cmap(i / max(n, 1)))[:3] for i in range(n)], axis=0)


def load_arrays(dataset: str, stem: str):
    base = RUNS_ROOT / dataset / stem
    if not (base / ".done").exists():
        raise RuntimeError(f"{dataset}/{stem}: not done")
    with np.load(base / "tracks_2d.npz") as d:
        tracks_2d = d["tracks"]            # (T, N, 2) (x, y)
        vis_2d = d["visibility"]           # (T, N)
        dim_2d = tuple(int(x) for x in d["dim"])  # (H, W)
    with np.load(base / "tracks_3d.npz") as d:
        pts3d = d["points_3d"]             # (N, T, 3)
        vis3d_raw = d["visibility"]        # (N, T, 1) bool
    vis3d = vis3d_raw[..., 0].astype(bool)  # (N, T)
    rgb = base / "rgb.mp4"
    return tracks_2d, vis_2d, dim_2d, pts3d, vis3d, rgb


def read_all_frames(mp4: Path):
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def shared_3d_lim(P: np.ndarray, V: np.ndarray, pad_frac=0.1):
    pts = []
    n, T, _ = P.shape
    for p in range(n):
        m = V[p]
        if m.sum() < 1:
            continue
        pts.append(P[p, m])
    if not pts:
        return ((-1, 1), (-1, 1), (-1, 1))
    pts = np.concatenate(pts, axis=0)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return ((-1, 1), (-1, 1), (-1, 1))
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    span = max((mx - mn).max(), 1e-3)
    pad = span * pad_frac
    ctr = (mn + mx) / 2
    half = span / 2 + pad
    return tuple((float(c - half), float(c + half)) for c in ctr)


def render_frame(rgb_frame, tracks_2d, vis_2d, pts3d, vis3d, t,
                 colors, lim_3d, dataset, stem, total_T, fig_dpi=110):
    """Render one frame: panel A = RGB + 2D track trail up to time t,
    panel B = 3D track trail up to time t."""
    H_img, W_img = rgb_frame.shape[:2]
    aspect = W_img / H_img
    fig_w = 14.0
    fig_h = fig_w / 2 / aspect + 0.6  # left panel height drives canvas
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=fig_dpi)

    # ── Left: RGB + 2D overlay ──────────────────────────────────────────────
    ax_l = fig.add_subplot(1, 2, 1)
    ax_l.imshow(rgb_frame)
    ax_l.set_xlim(0, W_img); ax_l.set_ylim(H_img, 0)
    ax_l.set_xticks([]); ax_l.set_yticks([])
    n_pts = tracks_2d.shape[1]
    for p in range(n_pts):
        m = vis_2d[: t + 1, p]
        if m.sum() < 2:
            continue
        xy = tracks_2d[: t + 1, p][m]
        ax_l.plot(xy[:, 0], xy[:, 1], color=colors[p], linewidth=1.2, alpha=0.9)
        ax_l.plot(xy[-1:, 0], xy[-1:, 1], "o", color=colors[p],
                  markersize=3.0, markeredgewidth=0)
    ax_l.set_title(f"sora 2 video + 2D tracks   t={t+1}/{total_T}",
                   fontsize=11)

    # ── Right: 3D ───────────────────────────────────────────────────────────
    ax_r = fig.add_subplot(1, 2, 2, projection="3d")
    n3 = pts3d.shape[0]
    for p in range(n3):
        m = vis3d[p, : t + 1]
        if m.sum() < 2:
            continue
        xs = pts3d[p, : t + 1, 0][m]
        ys = pts3d[p, : t + 1, 1][m]
        zs = pts3d[p, : t + 1, 2][m]
        ax_r.plot(xs, ys, zs, color=colors[p], linewidth=1.0, alpha=0.85)
        ax_r.scatter(xs[-1:], ys[-1:], zs[-1:], color=colors[p],
                     s=12, depthshade=False)
    xl, yl, zl = lim_3d
    ax_r.set_xlim(*xl); ax_r.set_ylim(*yl); ax_r.set_zlim(*zl)
    azim = -60 + 30 * (t / max(total_T - 1, 1))   # gentle rotation
    ax_r.view_init(elev=18, azim=azim)
    ax_r.set_xlabel("x", fontsize=8, labelpad=-6)
    ax_r.set_ylabel("y", fontsize=8, labelpad=-6)
    ax_r.set_zlabel("z", fontsize=8, labelpad=-6)
    for axis in (ax_r.xaxis, ax_r.yaxis, ax_r.zaxis):
        axis.set_tick_params(labelsize=6, pad=-2)
    ax_r.set_title("ViPE-lifted 3D tracks (camera frame)", fontsize=11)
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(f"{dataset} · {stem}", fontsize=12, y=0.98)
    fig.tight_layout()
    return fig


def render_clip(dataset: str, stem: str, fps: int, fig_dpi: int):
    print(f"  loading {dataset}/{stem}")
    tr2d, v2d, dim_2d, p3d, v3d, rgb_path = load_arrays(dataset, stem)
    frames = read_all_frames(rgb_path)
    T_video = len(frames)
    T_2d = tr2d.shape[0]
    T_3d = p3d.shape[1]
    T = min(T_video, T_2d, T_3d)
    if T < 2:
        raise RuntimeError(
            f"{dataset}/{stem}: too few frames (video={T_video} 2d={T_2d} 3d={T_3d})")
    n_pts = tr2d.shape[1]
    colors = colors_for(n_pts)
    lim_3d = shared_3d_lim(p3d[:, :T], v3d[:, :T])

    safe_stem = stem.replace("/", "_")
    out_mp4 = OUT_DIR / f"{dataset}__{safe_stem}.mp4"
    frames_dir = TMP_DIR / f"{dataset}__{safe_stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Clear any prior pngs
    for f in frames_dir.glob("*.png"):
        f.unlink()

    print(f"  rendering {T} frames @ dpi={fig_dpi}")
    for t in range(T):
        fig = render_frame(frames[t], tr2d, v2d, p3d, v3d, t, colors,
                           lim_3d, dataset, stem, T, fig_dpi=fig_dpi)
        fig.savefig(frames_dir / f"f{t:04d}.png", dpi=fig_dpi,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # Encode mp4 with libx264 yuv420p +faststart for inline GitHub Pages playback
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "f%04d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-preset", "medium", "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    # Cleanup pngs
    for f in frames_dir.glob("*.png"):
        f.unlink()
    frames_dir.rmdir()
    print(f"  -> {out_mp4} ({out_mp4.stat().st_size//1024} KB)")
    return out_mp4, {"dataset": dataset, "stem": stem,
                     "T": T, "n_pts": n_pts}


def discover_done_stems():
    out = []
    for ds_dir in sorted(RUNS_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for stem_dir in sorted(ds_dir.iterdir()):
            if (stem_dir / ".done").exists():
                out.append((ds_dir.name, stem_dir.name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=None,
                    help="Filter to 'dataset:stem' (default: all .done stems)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if args.stems:
        pairs = [tuple(s.split(":", 1)) for s in args.stems]
    else:
        pairs = discover_done_stems()
    if not pairs:
        print("no stems to render", file=sys.stderr)
        sys.exit(1)

    print(f"rendering {len(pairs)} stems")
    manifest = []
    for ds, stem in pairs:
        try:
            _, info = render_clip(ds, stem, fps=args.fps, fig_dpi=args.dpi)
            manifest.append(info)
        except Exception as e:
            print(f"  ERR {ds}/{stem}: {e}")

    (REPO / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {REPO / 'manifest.json'}")


if __name__ == "__main__":
    main()
