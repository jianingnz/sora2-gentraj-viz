#!/usr/bin/env python3
"""Emit index.html that lists every rendered mp4 in videos/ alongside its
manifest entry. Run after make_videos.py."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
VIDS = REPO / "videos"


def main():
    manifest = json.loads((REPO / "manifest.json").read_text())
    by_key = {f"{m['dataset']}__{m['stem']}": m for m in manifest}

    cards = []
    for mp4 in sorted(VIDS.glob("*.mp4")):
        key = mp4.stem
        m = by_key.get(key, {})
        ds = m.get("dataset", "?"); stem = m.get("stem", key)
        T = m.get("T", "?"); n_pts = m.get("n_pts", "?")
        meta = (
            f"<b>{ds}</b> · <code>{stem}</code> · "
            f"T={T} · N={n_pts} query points"
        )
        cards.append(
            f"<div class='card' id='{key}'>"
            f"<div class='hdr'>{meta}</div>"
            f"<video src='videos/{mp4.name}' controls loop muted "
            f"  preload='metadata' playsinline></video>"
            f"</div>"
        )

    n = len(cards)
    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sora 2 GenTraj — 2D + ViPE-lifted 3D tracks</title>
<style>
  :root {{ --bg:#f4f5f7; --card:#ffffff; --ink:#202124; --mute:#5f6368; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--ink);
                 font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, Helvetica, Arial, sans-serif; }}
  header {{ padding: 18px 22px 6px; }}
  header h1 {{ margin: 0 0 4px; font-size: 18px; }}
  header p  {{ margin: 0 0 12px; color: var(--mute); font-size: 13px; line-height: 1.45; }}
  main {{ max-width: 1280px; margin: 0 auto; padding: 0 18px 32px; }}
  .card {{ background: var(--card); margin: 14px 0; padding: 10px 14px;
            border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
  .hdr {{ font-size: 12.5px; color: var(--ink); margin-bottom: 6px;
           font-feature-settings: "tnum"; }}
  code {{ font-size: 11.5px; color: #3f51b5; }}
  video {{ width: 100%; height: auto; display: block; background: black;
            border-radius: 4px; }}
</style></head>
<body>
<header>
  <h1>sora 2 GenTraj — 2D + ViPE-lifted 3D tracks</h1>
  <p>
    Frame-0 GT visible points are tracked through the Sora 2 generated video by
    AllTracker, then lifted to 3D using ViPE depth/pose/intrinsics estimated on
    the same Sora video. Left panel: Sora frame + 2D track trail. Right panel:
    rotating 3D camera-frame trails. Per-track rainbow color is shared.
  </p>
</header>
<main>
{chr(10).join(cards)}
</main>
</body></html>
"""
    (REPO / "index.html").write_text(html)
    print(f"wrote {REPO / 'index.html'} ({n} cards)")


if __name__ == "__main__":
    main()
