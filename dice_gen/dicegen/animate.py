"""Run a throw, choose results, render frames and export them."""
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

from .shapes import get_shape
from .physics import Body, World, simulate, quat_to_mat
from .motions import MOTIONS
from .palettes import PALETTES
from . import render as rd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FELT = os.path.join(HERE, "assets", "felt_320x180.png")
SHADOW_TINT = np.array([70, 10, 25], float)


def run_throw(motion, seed, die="d6", size=15.0, fps=24, tries=40):
    """Simulate, retrying seeds until the throw is well behaved."""
    shape = get_shape(die)
    for k in range(tries):
        rng = np.random.default_rng(seed * 1000 + k)
        init = MOTIONS[motion](rng)
        bodies = [Body(shape, size, x.astype(float), v.astype(float), q.astype(float), w.astype(float))
                  for x, v, q, w in init]
        frames = simulate(bodies, World(**getattr(MOTIONS[motion], "world", {})), fps=fps)
        dur = len(frames) / fps
        zmax = max(st[0][2] for fr in frames for sub in fr for st in sub)
        final = frames[-1][-1]
        inside = all(24 < x[0] < 296 and 22 < x[1] < 158 for x, _ in final)
        if 1.0 < dur < 5.5 and zmax < 290 and inside:
            return shape, frames, rng
    raise RuntimeError(f"no stable throw found for {motion} seed {seed}")


def choose_labels(shape, frames, targets, rng):
    """Pick a symmetry rotation per die so its final up face shows target."""
    Qs = []
    for b, tgt in enumerate(targets):
        R = quat_to_mat(frames[-1][-1][b][1])
        if tgt is None:
            Qs.append(np.eye(3))
            continue
        cands = [Q for Q in shape.rot_group if shape.labels[shape.up_face(R @ Q)] == tgt]
        Qs.append(cands[rng.integers(len(cands))])
    return Qs


def render_clip(motion="drop_in", result=None, palette="bone", seed=0, die="d6",
                size=15.0, fps=24, smear=True, palettes=None):
    shape, frames, rng = run_throw(motion, seed, die, size, fps)
    n = len(frames[0][-1])
    if result is None or isinstance(result, int):
        targets = [result] * n
    else:
        targets = list(result) + [None] * (n - len(result))
    targets = [int(rng.integers(1, 7)) if t is None and die == "d6" else t for t in targets]
    Qs = choose_labels(shape, frames, targets, rng)
    if palettes is None:
        palettes = [palette] * n
    pals = [PALETTES[p] for p in palettes]

    out = []
    prev = frames[0][0]
    for i, fr in enumerate(frames):
        cur = fr[-1]
        layer = np.zeros((rd.H, rd.W, 4), np.uint8)
        polys = []
        order = sorted(range(n), key=lambda b: cur[b][0][2])
        for b in order:
            x, q = cur[b]
            if smear and len(fr) > 1:
                subs = [s[b] for s in fr]
                sm = rd.smear_layer(shape, size, subs, pals[b], Qs[b], prev[b], rng_seed=seed * 97 + i * 7 + b)
                if sm is not None:
                    m = sm[..., 3] > 0
                    layer[m] = sm[m]
            rgba, xy, _ = rd.render_die(shape, size, x, q, pals[b], Qs[b])
            rd.paste(layer, rgba, xy)
            polys.append(rd.shadow_polygon(shape, size, x, q))
        out.append(dict(layer=layer, polys=polys,
                        pos=[[float(v) for v in rd.project(cur[b][0])[0]] for b in range(n)],
                        height=[float(cur[b][0][2] - size) for b in range(n)]))
        prev = cur
    # Drop leading frames where nothing is on screen yet (keep one).
    first = next((i for i, f in enumerate(out) if f["layer"][..., 3].any()), 0)
    out = out[max(0, first - 1):]
    meta = dict(motion=motion, die=die, palette=palettes, seed=seed, fps=fps,
                frames=len(out), results=targets)
    return out, meta


def composite(frame, scale, bg=None):
    sa = rd.shadow_alpha(frame["polys"], scale)
    if bg is None:
        bg = np.asarray(Image.open(FELT).convert("RGB"))
    big = np.asarray(Image.fromarray(bg).resize((rd.W * scale, rd.H * scale), Image.NEAREST), float)
    big = big * (1 - sa[..., None]) + SHADOW_TINT * sa[..., None]
    die = np.asarray(Image.fromarray(frame["layer"]).resize((rd.W * scale, rd.H * scale), Image.NEAREST))
    m = die[..., 3:] > 0
    return np.where(m, die[..., :3], big).astype(np.uint8)


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def export_clip(frames, meta, outdir, gif_scale=2, video_scale=6, sprites=True, hold_end=0.6):
    os.makedirs(outdir, exist_ok=True)
    fps = meta["fps"]
    hold = int(hold_end * fps)
    seq = frames + [frames[-1]] * hold
    tmp = tempfile.mkdtemp()
    try:
        bg = np.asarray(Image.open(FELT).convert("RGB"))
        for i, f in enumerate(seq):
            Image.fromarray(composite(f, gif_scale, bg)).save(f"{tmp}/g{i:04d}.png")
        _ffmpeg(["-framerate", str(fps), "-i", f"{tmp}/g%04d.png",
                 "-vf", "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=none",
                 "-loop", "0", f"{outdir}/preview.gif"])
        if video_scale:
            for i, f in enumerate(seq):
                Image.fromarray(composite(f, video_scale, bg)).save(f"{tmp}/v{i:04d}.png")
            _ffmpeg(["-framerate", str(fps), "-i", f"{tmp}/v%04d.png", "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", f"{outdir}/preview_1080p.mp4"])
    finally:
        shutil.rmtree(tmp)
    if sprites:
        sd = os.path.join(outdir, "sprites")
        os.makedirs(sd, exist_ok=True)
        cols = 8
        rows = (len(frames) + cols - 1) // cols
        sheet = Image.new("RGBA", (cols * rd.W, rows * rd.H))
        shsheet = Image.new("L", (cols * rd.W, rows * rd.H))
        for i, f in enumerate(frames):
            die = Image.fromarray(f["layer"], "RGBA")
            sh = Image.fromarray((rd.shadow_alpha(f["polys"], 1) * 255).astype(np.uint8), "L")
            die.save(f"{sd}/die_{i:03d}.png")
            sh.save(f"{sd}/shadow_{i:03d}.png")
            sheet.paste(die, ((i % cols) * rd.W, (i // cols) * rd.H))
            shsheet.paste(sh, ((i % cols) * rd.W, (i // cols) * rd.H))
        sheet.save(f"{outdir}/die_sheet.png")
        shsheet.save(f"{outdir}/shadow_sheet.png")
    meta = dict(meta, canvas=[rd.W, rd.H], sheet_columns=8,
                positions=[f["pos"] for f in frames], heights=[f["height"] for f in frames])
    with open(f"{outdir}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta
