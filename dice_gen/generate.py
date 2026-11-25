#!/usr/bin/env python3
"""Generate pixel-art d6 roll animations.

Examples:
  python generate.py --motion drop_in --result 6 --palette bone --seed 1
  python generate.py --motion pair --result 3,4 --palette bone,royal
  python generate.py --batch            # the curated gallery set
  python generate.py --list             # show motions and palettes
"""
import argparse
import json
import os

import numpy as np

from dicegen.animate import render_clip, export_clip
from dicegen.motions import MOTIONS
from dicegen.palettes import PALETTES

BATCH = [
    # name,                   motion,       results,  palettes,                    seed
    ("01_drop_in_bone_1",     "drop_in",    [1],      ["bone"],                    1),
    ("02_drop_in_royal_6",    "drop_in",    [6],      ["royal"],                   5),
    ("03_side_left_onyx_3",   "side_left",  [3],      ["onyx"],                    2),
    ("04_side_right_gold_5",  "side_right", [5],      ["gold"],                    3),
    ("05_bank_shot_royal_4",  "bank_shot",  [4],      ["royal"],                   4),
    ("06_high_toss_jade_2",   "high_toss",  [2],      ["jade"],                    6),
    ("07_slam_ivory_red_6",   "slam",       [6],      ["ivory_red"],               7),
    ("08_skid_bone_1",        "skid",       [1],      ["bone"],                    8),
    ("09_spinner_amethyst_5", "spinner",    [5],      ["amethyst"],                9),
    ("10_pair_bone_royal",    "pair",       [3, 4],   ["bone", "royal"],           10),
    ("11_scatter_triple_six", "scatter3",   [6, 6, 6], ["gold", "onyx", "jade"],   11),
    ("12_drop_in_frost_2",    "drop_in",    [2],      ["frost"],                   12),
]


def run(name, motion, results, palettes, seed, outroot, video=True):
    frames, meta = render_clip(motion, result=results, palettes=palettes, seed=seed)
    outdir = os.path.join(outroot, name)
    meta = export_clip(frames, meta, outdir, video_scale=6 if video else 0)
    print(f"{name}: {meta['frames']} frames, results {meta['results']}")
    return dict(name=name, motion=motion, results=meta["results"], palettes=palettes,
                frames=meta["frames"], fps=meta["fps"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", default="drop_in", choices=sorted(MOTIONS))
    ap.add_argument("--result", default=None, help="value per die, e.g. 6 or 3,4 (random if omitted)")
    ap.add_argument("--palette", default="bone", help="palette per die, e.g. bone or bone,royal")
    ap.add_argument("--seed", type=int, default=0, help="changes the throw; same seed = same motion")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-video", action="store_true", help="skip the 1080p mp4")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("motions: ", ", ".join(MOTIONS))
        print("palettes:", ", ".join(PALETTES))
        return
    os.makedirs(a.out, exist_ok=True)
    if a.batch:
        index = [run(n, m, r, p, s, a.out, not a.no_video) for n, m, r, p, s in BATCH]
        with open(os.path.join(a.out, "index.json"), "w") as fh:
            json.dump(index, fh, indent=1)
        return
    results = [int(v) for v in a.result.split(",")] if a.result else None
    pals = a.palette.split(",")
    name = a.name or f"{a.motion}_{'-'.join(pals)}_{a.result or 'rand'}_s{a.seed}"
    # Repeat the palette list to cover every die in the throw.
    ndice = len(MOTIONS[a.motion](np.random.default_rng(0)))
    pals = (pals * ndice)[:ndice]
    run(name, a.motion, results, pals, a.seed, a.out, not a.no_video)


if __name__ == "__main__":
    main()
