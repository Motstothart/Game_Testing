"""Throw presets. Each returns a list of (x, v, q, w) initial conditions.

Units: native pixels and seconds. Canvas is 320x180, camera above centre,
z up toward the camera. A d6 is ~30px across.
"""
import numpy as np
from .physics import random_quat


def _spin(rng, mag, axis_bias=None):
    w = rng.normal(size=3)
    if axis_bias is not None:
        w = w * 0.35 + np.asarray(axis_bias, float)
    return w / np.linalg.norm(w) * mag


def drop_in(rng):
    """The reference: tossed in over the top edge, arcs toward camera."""
    x = np.array([rng.uniform(120, 200), -30, 60])
    v = np.array([rng.uniform(-50, 50), rng.uniform(210, 260), rng.uniform(640, 760)])
    return [(x, v, random_quat(rng), _spin(rng, rng.uniform(12, 18), (1, 0, 0)))]


def side_left(rng):
    """Hurled in from the left edge, skips and tumbles across."""
    x = np.array([-30, rng.uniform(70, 120), 45])
    v = np.array([rng.uniform(360, 440), rng.uniform(-40, 40), rng.uniform(380, 480)])
    return [(x, v, random_quat(rng), _spin(rng, rng.uniform(14, 20), (0, 1, 0)))]


def side_right(rng):
    """Hurled in from the right edge."""
    x = np.array([350, rng.uniform(70, 120), 45])
    v = np.array([-rng.uniform(360, 440), rng.uniform(-40, 40), rng.uniform(380, 480)])
    return [(x, v, random_quat(rng), _spin(rng, rng.uniform(14, 20), (0, -1, 0)))]


def bank_shot(rng):
    """Thrown hard from the bottom-right corner, banks off the far walls."""
    x = np.array([345, 200, 40])
    v = np.array([-rng.uniform(380, 460), -rng.uniform(300, 360), rng.uniform(80, 160)])
    return [(x, v, random_quat(rng), _spin(rng, rng.uniform(16, 22)))]


def high_toss(rng):
    """Flipped straight up from the table: swells toward camera and drops."""
    x = np.array([rng.uniform(140, 180), rng.uniform(95, 115), 16])
    v = np.array([rng.uniform(-25, 25), rng.uniform(-30, 10), rng.uniform(820, 900)])
    return [(x, v, random_quat(rng), _spin(rng, rng.uniform(8, 12)))]


def slam(rng):
    """Starts right under the lens and slams down with spin."""
    x = np.array([rng.uniform(130, 190), rng.uniform(75, 105), 230])
    v = np.array([rng.uniform(-40, 40), rng.uniform(-30, 30), -rng.uniform(150, 260)])
    return [(x, v, random_quat(rng), _spin(rng, rng.uniform(10, 16)))]


def skid(rng):
    """Low, fast slide in from the left that catches an edge and flips."""
    x = np.array([-30, rng.uniform(80, 110), 20])
    v = np.array([rng.uniform(620, 700), rng.uniform(-20, 20), rng.uniform(60, 120)])
    return [(x, v, random_quat(rng), _spin(rng, rng.uniform(4, 8), (0, 1, 0.3)))]


def spinner(rng):
    """Dropped onto a corner with heavy yaw: spins like a top, then falls."""
    x = np.array([rng.uniform(140, 180), rng.uniform(80, 100), 60])
    v = np.array([rng.uniform(-20, 20), rng.uniform(-20, 20), -20])
    # Corner-down orientation: rotate so the (1,1,1) diagonal points up.
    ax = np.cross([1, 1, 1], [0, 0, 1.0])
    ax /= np.linalg.norm(ax)
    ang = np.arccos(1 / np.sqrt(3))
    q = np.array([np.cos(ang / 2), *(ax * np.sin(ang / 2))])
    w = np.array([rng.normal() * 1.5, rng.normal() * 1.5, rng.choice([-1, 1]) * rng.uniform(34, 44)])
    return [(x, v, q, w)]


def pair(rng):
    """Two dice tossed in together from the top edge."""
    out = []
    for lane in (-1, 1):
        x = np.array([160 + lane * rng.uniform(28, 45), -30 - rng.uniform(0, 15), 60])
        v = np.array([lane * rng.uniform(10, 50), rng.uniform(210, 260), rng.uniform(560, 680)])
        out.append((x, v, random_quat(rng), _spin(rng, rng.uniform(12, 18), (1, 0, 0))))
    return out


def scatter3(rng):
    """Three dice flung from the bottom edge, fanning out."""
    out = []
    for lane in (-1, 0, 1):
        x = np.array([160 + lane * 34, 210 + rng.uniform(0, 12), 50])
        v = np.array([lane * rng.uniform(60, 110), -rng.uniform(250, 300), rng.uniform(520, 640)])
        out.append((x, v, random_quat(rng), _spin(rng, rng.uniform(12, 18), (1, 0, 0))))
    return out


MOTIONS = {
    "drop_in": drop_in,
    "side_left": side_left,
    "side_right": side_right,
    "bank_shot": bank_shot,
    "high_toss": high_toss,
    "slam": slam,
    "skid": skid,
    "spinner": spinner,
    "pair": pair,
    "scatter3": scatter3,
}

# Per-motion world overrides (e.g. a slicker felt for the skid).
skid.world = dict(friction=0.16, lin_damp=0.15)
spinner.world = dict(floor_spin_damp=1.2)
