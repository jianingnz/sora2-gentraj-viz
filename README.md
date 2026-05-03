# sora2-gentraj-viz

Per-clip 2-panel previews of the **Sora 2 GenTraj** pipeline: frame-0 GT
visible query points are tracked through each Sora-2 generated video by
AllTracker, then lifted to 3D using ViPE depth + camera pose + intrinsics
estimated on the same Sora video.

- **Left panel** — Sora video frame with the 2D track trail up to time `t`,
  rainbow-colored per query point.
- **Right panel** — matplotlib 3D plot of the lifted camera-frame tracks up to
  time `t`. Per-track color matches the left panel; the view rotates gently as
  the clip plays so you can read depth.

Open [`index.html`](index.html) (or the GitHub Pages URL) for the gallery.

## Layout

| path | purpose |
|---|---|
| `index.html` | grid view of all rendered clips |
| `videos/*.mp4` | per-clip mp4s (libx264 yuv420p +faststart, ~8 fps) |
| `manifest.json` | one entry per video: dataset, stem, T frames, N points |
| `make_videos.py` | renders mp4 from `_GenTraj/sora2/runs/<dataset>/<stem>/` |
| `build_index.py` | regenerates `index.html` from the `videos/` folder |

## Reproduce

The source data lives at
`/weka/prior-default/jianingz/home/project/_GenTraj/sora2/runs/<dataset>/<stem>/`.
Each completed stem has `tracks_2d.npz`, `tracks_3d.npz`, `rgb.mp4`, and a
`.done` marker.

```bash
cd /weka/prior-default/jianingz/home/visual/sora2-gentraj-viz
# all stems with .done:
python make_videos.py --all
# specific stems only:
python make_videos.py --stems davis:bear worldtrack:01f258-3_obj_source_left_6_clip01_obj1_2_3_4_5_t0-127
python build_index.py
```

## Pipeline (recap)

For each Sora-generated stem:

1. Extract frame-0 visible query points from the source-dataset GT NPZ.
2. Re-encode `result_at_target.mp4` to 480p.
3. AllTracker tracks those points through the 480p Sora video → `tracks_2d.npz`.
4. ViPE infers depth, camera pose, intrinsics on the Sora video.
5. `vipe_to_colmap_general.py` lifts the 2D tracks to 3D → `tracks_3d.npz`.

No filter+smooth (Stage 5) was run for sora2 — the right panel is the raw 3D
lift.
