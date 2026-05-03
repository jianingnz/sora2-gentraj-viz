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
DAS_BENCH = Path(
    "/weka/prior-default/jianingz/home/project/DiffusionAsShader/"
    "outputs/bench_noobj/v3_pred2"
)
TASKS_JSON = Path(
    "/weka/prior-default/jianingz/home/project/_GenTraj/sora2/tasks.json"
)
OUT_DIR = REPO / "videos"
TMP_DIR = REPO / "tmp_frames"


def _meta_for(dataset: str, stem: str):
    p = DAS_BENCH / "das" / dataset / stem / "pred_track_meta.json"
    return json.loads(p.read_text())


def _gt_at_target(dataset: str, stem: str):
    return DAS_BENCH / "gt_clipped" / dataset / stem / "gt_at_target.mp4"


def _gt_npz_for(dataset: str, stem: str):
    """Look up the source GT NPZ from sora2 tasks.json."""
    if not hasattr(_gt_npz_for, "_cache"):
        _gt_npz_for._cache = {
            (t["dataset"], t["stem"]): t["gt_npz"]
            for t in json.loads(TASKS_JSON.read_text())
        }
    return Path(_gt_npz_for._cache[(dataset, stem)])


# ─────────────────────────────────────────────────────────────────────────────
# GT 2D track extraction over the prediction window [src_start, src_end].
# Returns (N_query, 2) frame-0 query points + (T_target, N_query, 2) GT 2D
# positions + (T_target, N_query) visibility, ordered the same way as the
# sora pipeline's query points so colors match across panels.
# ─────────────────────────────────────────────────────────────────────────────
def gt_window_davis(gt_npz: Path, src_start: int, src_end: int):
    d = np.load(gt_npz, allow_pickle=True)
    tracks_dict = d["tracks"].item()
    vis_dict = d["visibility"].item()
    H, W = int(d["dim"][0]), int(d["dim"][1])
    a, b = src_start, src_end + 1
    T = b - a
    f0_xy_list = []
    win_xy_list = []
    win_vis_list = []
    for obj in tracks_dict:
        tr = tracks_dict[obj]              # (T_full, N_obj, 2)
        v = vis_dict[obj]                  # (T_full, N_obj)
        if tr.shape[0] == 0:
            continue
        keep_obj_idx = np.where(v[0])[0]   # frame-0 visible indices
        if len(keep_obj_idx) == 0:
            continue
        f0_xy = tr[0, keep_obj_idx]
        if b > tr.shape[0]:
            raise RuntimeError(
                f"davis: src_end+1={b} exceeds GT T_full={tr.shape[0]} for {gt_npz}")
        win_xy = tr[a:b, keep_obj_idx]      # (T, N_obj_visible_at_0, 2)
        win_v = v[a:b, keep_obj_idx]        # (T, N_obj_visible_at_0)
        f0_xy_list.append(f0_xy)
        win_xy_list.append(win_xy)
        win_vis_list.append(win_v)
    f0_xy = np.concatenate(f0_xy_list, axis=0)
    win_xy = np.concatenate(win_xy_list, axis=1)
    win_vis = np.concatenate(win_vis_list, axis=1)
    return f0_xy, win_xy, win_vis, (H, W)


def gt_window_hot3d(gt_npz: Path, src_start: int, src_end: int):
    d = np.load(gt_npz)
    tr = d["tracks"]                       # (T_full, 2000, 2)
    v = d["visibility"]                    # (T_full, 2000)
    H, W = int(d["dim"][0]), int(d["dim"][1])
    a, b = src_start, src_end + 1
    if b > tr.shape[0]:
        raise RuntimeError(
            f"hot3d: src_end+1={b} exceeds GT T_full={tr.shape[0]} for {gt_npz}")
    keep = np.where(v[0])[0]
    f0_xy = tr[0, keep]
    win_xy = tr[a:b, keep]
    win_vis = v[a:b, keep]
    return f0_xy, win_xy, win_vis, (H, W)


def gt_window_worldtrack(gt_npz: Path, src_start: int, src_end: int):
    d = np.load(gt_npz, allow_pickle=True)
    tracks_xyz = d["tracks_XYZ"]           # (T_full, N, 3) cam-space metres
    vis = d["visibility"]                  # (T_full, N)
    fx, fy, cx, cy = d["fx_fy_cx_cy"]
    jpg = d["images_jpeg_bytes"][0]
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    a, b = src_start, src_end + 1
    if b > tracks_xyz.shape[0]:
        raise RuntimeError(
            f"worldtrack: src_end+1={b} exceeds T_full={tracks_xyz.shape[0]} for {gt_npz}")

    f0 = tracks_xyz[0]
    valid0 = vis[0] & (f0[:, 2] > 1e-6)
    f0_in = f0[valid0]
    px0 = f0_in[:, 0] / f0_in[:, 2] * fx + cx
    py0 = f0_in[:, 1] / f0_in[:, 2] * fy + cy
    in_b = (px0 >= 0) & (px0 < W) & (py0 >= 0) & (py0 < H)
    keep = np.where(valid0)[0][in_b]

    f0_xy = np.stack([px0[in_b], py0[in_b]], axis=1)

    T = b - a
    N = len(keep)
    win_xy = np.zeros((T, N, 2), dtype=np.float32)
    win_vis = np.zeros((T, N), dtype=bool)
    for ti, t in enumerate(range(a, b)):
        xyz_t = tracks_xyz[t, keep]
        vt = vis[t, keep] & (xyz_t[:, 2] > 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            px = xyz_t[:, 0] / xyz_t[:, 2] * fx + cx
            py = xyz_t[:, 1] / xyz_t[:, 2] * fy + cy
        win_xy[ti, :, 0] = px
        win_xy[ti, :, 1] = py
        win_vis[ti] = vt & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return f0_xy, win_xy, win_vis, (H, W)


GT_WINDOW = {
    "davis": gt_window_davis,
    "hot3d": gt_window_hot3d,
    "worldtrack": gt_window_worldtrack,
}


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


def _draw_2d_overlay(ax, frame, tracks, vis, t, colors, title):
    H_img, W_img = frame.shape[:2]
    ax.imshow(frame)
    ax.set_xlim(0, W_img); ax.set_ylim(H_img, 0)
    ax.set_xticks([]); ax.set_yticks([])
    n_pts = tracks.shape[1]
    for p in range(n_pts):
        m = vis[: t + 1, p]
        if m.sum() < 2:
            continue
        xy = tracks[: t + 1, p][m]
        ax.plot(xy[:, 0], xy[:, 1], color=colors[p], linewidth=1.2, alpha=0.9)
        ax.plot(xy[-1:, 0], xy[-1:, 1], "o", color=colors[p],
                markersize=3.0, markeredgewidth=0)
    ax.set_title(title, fontsize=11)


def render_frame(rgb_frame, tracks_2d, vis_2d, pts3d, vis3d, t,
                 colors_q, colors_sora, lim_3d, dataset, stem, total_T,
                 gt_frame=None, gt_tracks=None, gt_vis=None,
                 fig_dpi=110):
    """3-panel layout: GT video | sora video | 3D.

    `colors_q` indexes the GT query points; `colors_sora` indexes the sora-side
    AllTracker tracks. They are independent — sora-side N may differ from
    GT-side N (e.g. davis has multi-object scaling and clipping)."""
    H_img, W_img = rgb_frame.shape[:2]
    aspect = W_img / H_img
    has_gt = gt_frame is not None and gt_tracks is not None
    n_panels = 3 if has_gt else 2
    fig_w = 6.6 * n_panels
    fig_h = fig_w / n_panels / aspect + 0.7
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=fig_dpi)

    panel = 1
    if has_gt:
        ax_gt = fig.add_subplot(1, n_panels, panel)
        _draw_2d_overlay(ax_gt, gt_frame, gt_tracks, gt_vis, t, colors_q,
                         f"GT video + 2D GT tracks   t={t+1}/{total_T}")
        panel += 1

    ax_sora = fig.add_subplot(1, n_panels, panel)
    _draw_2d_overlay(ax_sora, rgb_frame, tracks_2d, vis_2d, t, colors_sora,
                     f"sora 2 + AllTracker 2D   t={t+1}/{total_T}")
    panel += 1

    # ── 3D ──────────────────────────────────────────────────────────────────
    ax_r = fig.add_subplot(1, n_panels, panel, projection="3d")
    n3 = pts3d.shape[0]
    for p in range(n3):
        m = vis3d[p, : t + 1]
        if m.sum() < 2:
            continue
        xs = pts3d[p, : t + 1, 0][m]
        ys = pts3d[p, : t + 1, 1][m]
        zs = pts3d[p, : t + 1, 2][m]
        ax_r.plot(xs, ys, zs, color=colors_sora[p], linewidth=1.0, alpha=0.85)
        ax_r.scatter(xs[-1:], ys[-1:], zs[-1:], color=colors_sora[p],
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
    n_pts_sora = tr2d.shape[1]
    colors_sora = colors_for(n_pts_sora)
    lim_3d = shared_3d_lim(p3d[:, :T], v3d[:, :T])

    # GT panel: source video clip + GT 2D tracks of the same query subset
    gt_frames = gt_tracks = gt_vis = None
    colors_q = None
    try:
        meta = _meta_for(dataset, stem)
        src_a, src_b = int(meta["src_start"]), int(meta["src_end"])
        f0_xy, win_xy, win_vis, gt_dim_HW = GT_WINDOW[dataset](
            _gt_npz_for(dataset, stem), src_a, src_b)
        gt_path = _gt_at_target(dataset, stem)
        if gt_path.exists():
            gt_frames = read_all_frames(gt_path)
            T_gt = min(len(gt_frames), win_xy.shape[0], T)
            gt_frames = gt_frames[:T_gt]
            # Slice all per-frame arrays to the shared length
            win_xy = win_xy[:T_gt]
            win_vis = win_vis[:T_gt]
            # Scale GT tracks if the clip mp4 was resized away from the GT NPZ dim
            H_clip, W_clip = gt_frames[0].shape[:2]
            H_npz, W_npz = gt_dim_HW
            if (H_clip, W_clip) != (H_npz, W_npz):
                sx = W_clip / W_npz; sy = H_clip / H_npz
                win_xy = win_xy.copy()
                win_xy[..., 0] *= sx; win_xy[..., 1] *= sy
            gt_tracks = win_xy
            gt_vis = win_vis
            colors_q = colors_for(gt_tracks.shape[1])
            T = T_gt
    except Exception as e:
        print(f"  [warn] no GT panel for {dataset}/{stem}: {e}")

    safe_stem = stem.replace("/", "_")
    out_mp4 = OUT_DIR / f"{dataset}__{safe_stem}.mp4"
    frames_dir = TMP_DIR / f"{dataset}__{safe_stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()

    print(f"  rendering {T} frames @ dpi={fig_dpi} (gt_panel={gt_tracks is not None})")
    for t in range(T):
        fig = render_frame(
            frames[t], tr2d, v2d, p3d, v3d, t,
            colors_q if colors_q is not None else colors_sora,
            colors_sora, lim_3d, dataset, stem, T,
            gt_frame=gt_frames[t] if gt_frames is not None else None,
            gt_tracks=gt_tracks, gt_vis=gt_vis, fig_dpi=fig_dpi)
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
                     "T": T, "n_pts": n_pts_sora,
                     "n_pts_gt": int(gt_tracks.shape[1]) if gt_tracks is not None else None}


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
