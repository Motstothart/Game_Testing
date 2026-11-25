"""Pixel-art renderer for dice.

Dice are ray-cast at the native 320x180 resolution through a gentle
perspective camera above the table centre, toon shaded with a small ramp,
outlined in maroon and finished with pips or pixel numerals. Shadows are
soft projected silhouettes rendered at output scale.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .physics import quat_to_mat

W, H = 320, 180
CAM = np.array([W / 2, H / 2, 360.0])
LIGHT = np.array([0.40, -0.50, 1.0]) / np.linalg.norm([0.40, -0.50, 1.0])
# Direction shadows fall per unit of height (toward lower-left like the ref).
SHADOW_DIR = np.array([-0.55, 0.38])

PIP_LAYOUT = {
    1: [(0, 0)],
    2: [(-1, -1), (1, 1)],
    3: [(-1, -1), (0, 0), (1, 1)],
    4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
    5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
    6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)],
}

FONT = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "011", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "_": ["000", "000", "000", "000", "111"],
}


def _glyph_bitmap(value):
    s = str(value)
    rows = [""] * 5
    for i, ch in enumerate(s):
        for r in range(5):
            rows[r] += ("0" if i else "") + FONT[ch][r]
    bm = np.array([[c == "1" for c in row] for row in rows])
    if value in (6, 9):  # underline to tell them apart
        bm = np.vstack([bm, np.zeros((1, bm.shape[1]), bool), np.ones((1, bm.shape[1]), bool)])
    return bm


def project(p):
    """World point(s) -> screen coords (native pixels)."""
    p = np.atleast_2d(p)
    s = CAM[2] / (CAM[2] - p[:, 2])
    return CAM[:2] + (p[:, :2] - CAM[:2]) * s[:, None]


def _raycast(shape, size, x, R, x0, y0, x1, y1):
    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    P = np.stack([gx, gy, np.zeros_like(gx)], -1).reshape(-1, 3)
    o = (CAM - x) @ R / size                       # R^T (C - x)
    r = (P - CAM) @ R / size
    Nn, Dd = shape.all_n, shape.all_d
    num = Dd[None, :] - (o @ Nn.T)[None, :]        # (1,K)
    den = r @ Nn.T                                 # (M,K)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num / den
    enter = np.where(den < 0, t, -np.inf)
    exit_ = np.where(den > 0, t, np.inf)
    miss_par = (np.abs(den) < 1e-12) & (num < 0)
    te = enter.max(1)
    tx = exit_.min(1)
    hit = (te <= tx) & (te > 0) & ~miss_par.any(1)
    fid = np.where(hit, enter.argmax(1), -1)
    pl = o[None, :] + te[:, None] * r
    hshape = (y1 - y0, x1 - x0)
    return fid.reshape(hshape), pl.reshape(hshape + (3,))


def _tone(nz_dot):
    # 0 = brightest. Thresholds tuned so a resting top face is tone 0.
    return np.select([nz_dot > 0.74, nz_dot > 0.52, nz_dot > 0.26, nz_dot > -0.05], [0, 1, 2, 3], 4)


def render_die(shape, size, x, q, pal, Qlab=None, flat=False, decal_axes=None):
    """Render one die. Returns (rgba patch, (x0, y0), face-id patch).

    Qlab: optional symmetry rotation used to choose which number lands up.
    flat: skip decals (used for smear trails).
    """
    R = quat_to_mat(q)
    if Qlab is not None:
        R = R @ Qlab
    V = x + (shape.verts * size) @ R.T
    sp = project(V)
    x0 = int(np.floor(sp[:, 0].min())) - 2
    y0 = int(np.floor(sp[:, 1].min())) - 2
    x1 = int(np.ceil(sp[:, 0].max())) + 3
    y1 = int(np.ceil(sp[:, 1].max())) + 3
    fid, pl = _raycast(shape, size, x, R, x0, y0, x1, y1)
    hshape = fid.shape
    F = len(shape.normals)
    alln = shape.all_n
    nworld = alln[np.clip(fid, 0, None)] @ R.T
    shade = nworld @ LIGHT
    tone = _tone(shade)
    bevel = fid >= F
    # Bevels read one step darker so edges get a soft rounded rim.
    tone = np.where(bevel, np.minimum(tone + 1, 4), tone)
    ramp = np.array(pal["ramp"], float)
    rgb = ramp[tone]

    ink = np.zeros(hshape, bool)
    if not flat:
        for f in range(F):
            m = fid == f
            if not m.any():
                continue
            fu, fv = shape.face_u[f], shape.face_v[f]
            if decal_axes is not None and f in decal_axes:
                fu, fv = decal_axes[f]
            a = pl[m] @ fu
            b = pl[m] @ fv
            val = shape.labels[f]
            hit = np.zeros(a.shape, bool)
            if shape.decal == "pips":
                rad = pal.get("pip_r", 0.23) * (1.18 if val == 1 else 1.0)
                for cx, cy in PIP_LAYOUT[val]:
                    hit |= (a - cx * 0.5) ** 2 + (b - cy * 0.5) ** 2 < rad ** 2
            else:
                bm = _glyph_bitmap(val)
                gh, gw = bm.shape
                g = shape.glyph_scale
                ci = np.floor(-a / g + gw / 2).astype(int)
                ri = np.floor(-b / g + gh / 2 - (1 if gh > 5 else 0)).astype(int)
                ok = (ci >= 0) & (ci < gw) & (ri >= 0) & (ri < gh)
                hit[ok] = bm[ri[ok], ci[ok]]
            ink[m] = hit
    ink_col = np.array(pal["ink"], float)
    # Ink on darker faces is lifted slightly so it never vanishes.
    ink_rgb = ink_col * (1 - 0.06 * tone[..., None]) + ramp[0] * 0.06 * tone[..., None]
    rgb = np.where(ink[..., None], ink_rgb, rgb)

    die = fid >= 0
    # Lines where two main faces of different tone meet (through a bevel).
    line = np.zeros(hshape, bool)
    t_pad = np.pad(np.where(die, tone, -9), 1, constant_values=-9)
    c = t_pad[1:-1, 1:-1]
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = t_pad[1 + dy:1 + dy + hshape[0], 1 + dx:1 + dx + hshape[1]]
        line |= die & (nb >= 0) & (nb - c >= 2) & ~ink
    rgb = np.where(line[..., None], np.array(pal["edge"], float), rgb)

    # Outer outline (1px, 4-neighbour) in maroon.
    d_pad = np.pad(die, 1)
    ring = np.zeros(hshape, bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        ring |= d_pad[1 + dy:1 + dy + hshape[0], 1 + dx:1 + dx + hshape[1]]
    ring &= ~die
    alpha = np.where(die | ring, 255, 0)
    rgb = np.where(ring[..., None], np.array(pal["outline"], float), rgb)
    rgba = np.dstack([rgb, alpha]).clip(0, 255).astype(np.uint8)
    return rgba, (x0, y0), np.where(ring, -2, fid)


def shadow_polygon(shape, size, x, q):
    R = quat_to_mat(q)
    V = x + (shape.verts * size) @ R.T
    pts = V[:, :2] + V[:, 2:3] * SHADOW_DIR[None, :] * 0.55 + SHADOW_DIR * 2.0
    # Monotone chain convex hull.
    P = sorted(map(tuple, np.round(pts, 4)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    height = float(max(0.0, x[2] - size))
    return hull, height


def shadow_alpha(polys, scale, base_opacity=0.62):
    """Soft shadow alpha (float 0..1) at the given output scale."""
    acc = np.zeros((H * scale, W * scale), float)
    for hull, height in polys:
        img = Image.new("L", (W * scale, H * scale), 0)
        dr = ImageDraw.Draw(img)
        grow = 1.0 + height * 0.004
        cx = np.mean([p[0] for p in hull])
        cy = np.mean([p[1] for p in hull])
        pts = [((cx + (px - cx) * grow) * scale, (cy + (py - cy) * grow) * scale) for px, py in hull]
        dr.polygon(pts, fill=255)
        blur = float((1.6 + height * 0.06) * scale)
        img = img.filter(ImageFilter.GaussianBlur(blur))
        op = base_opacity / (1.0 + height * 0.012)
        acc = np.maximum(acc, np.asarray(img, float) / 255.0 * op)
    return acc


def smear_layer(shape, size, substates, pal, Qlab, prev_state, rng_seed=0):
    """Jagged motion smear built from sub-frame poses (oldest first).

    Returns rgba (H, W, 4) native-size layer containing only trail pixels, or
    None if the die is not moving fast enough to warrant a smear.
    """
    x_now = substates[-1][0]
    x_prev = prev_state[0]
    s_now = project(x_now)[0]
    s_prev = project(x_prev)[0]
    mv = s_now - s_prev
    speed = np.linalg.norm(mv)
    if speed < 9:
        return None
    d = mv / speed
    perp = np.array([-d[1], d[0]])
    poses = [prev_state] + list(substates[:-1])
    K = len(poses)
    canvas = np.zeros((H, W, 4), np.uint8)
    rng = np.random.default_rng(rng_seed)
    lens = rng.uniform(0.25, 1.0, 512)
    for k, (xk, qk) in enumerate(poses):
        age = 1.0 - k / K                     # 1 = oldest
        rgba, (ox, oy), fid = render_die(shape, size, xk, qk, pal, Qlab, flat=True)
        hh, ww = fid.shape
        ys, xs = np.nonzero(fid >= 0)
        gx, gy = xs + ox, ys + oy
        # 2px-wide streak columns, each with its own random length.
        bucket = (np.floor((gx * perp[0] + gy * perp[1]) / 2).astype(int)) % 512
        keep = age <= lens[bucket]
        ok = keep & (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        col = rgba[ys[ok], xs[ok]].copy()
        # Trails flatten to the light body tones; the tips go one step darker.
        tip = age > lens[bucket[ok]] * 0.7
        col[:, :3] = np.where(tip[:, None], np.array(pal["ramp"][1], np.uint8),
                              np.array(pal["ramp"][0], np.uint8))
        col[:, 3] = 255
        canvas[gy[ok], gx[ok]] = col
    # Outline the trail.
    a = canvas[..., 3] > 0
    ring = np.zeros_like(a)
    pad = np.pad(a, 1)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        ring |= pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    ring &= ~a
    canvas[ring] = (*pal["outline"], 255)
    return canvas


def paste(canvas, rgba, xy):
    """Alpha-over paste of a native patch onto an (H,W,4) uint8 canvas."""
    x0, y0 = xy
    h, w = rgba.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    sub = rgba[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    m = sub[..., 3] > 0
    region = canvas[cy0:cy1, cx0:cx1]
    region[m] = sub[m]
