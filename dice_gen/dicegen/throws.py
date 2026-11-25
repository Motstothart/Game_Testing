"""Five-dice throws: aggressive presets, simulation with metrics, validation.

A throw is a "handful" of dice flung in from one edge (or slammed down from
above the lens). Every die must keep tumbling while it moves along the felt;
throws where a die skates flat are rejected and re-rolled with a new seed.
"""
import numpy as np

from .physics import Body, World, step, flat_rest_quat, rest_height, slerp, random_quat
from .shapes import get_shape

W, H = 320, 180
CENTER = np.array([160.0, 92.0])

AGGRESSIVE = dict(gravity=4200.0, restitution=0.45, wall_restitution=0.5, friction=0.6,
                  lin_damp=0.3, ang_damp=0.35, floor_spin_damp=2.5)

# Edge name -> (start point on/just past the edge, unit throw direction)
EDGES = {
    "top": ((160, -34), (0, 1)),
    "bottom": ((160, 214), (0, -1)),
    "left": ((-34, 92), (1, 0)),
    "right": ((354, 92), (-1, 0)),
    "top_left": ((-24, -24), (0.83, 0.55)),
    "top_right": ((344, -24), (-0.83, 0.55)),
    "bottom_left": ((-24, 204), (0.83, -0.55)),
    "bottom_right": ((344, 204), (-0.83, -0.55)),
}


def handful(rng, edge, n=5):
    """n dice flung in a loose cluster from one edge, rolling forward hard."""
    (sx, sy), d = EDGES[edge]
    d = np.array(d, float)
    d /= np.linalg.norm(d)
    side = np.array([-d[1], d[0]])
    aim = CENTER + rng.uniform(-30, 30) * side + d * rng.uniform(-10, 25)
    out = []
    lanes = rng.permutation(np.linspace(-1, 1, n))
    for lane in lanes:
        start = np.array([sx, sy], float) + side * lane * rng.uniform(26, 40) - d * rng.uniform(0, 26)
        tgt = aim + side * lane * rng.uniform(45, 80) + d * rng.uniform(-35, 35)
        dist = np.linalg.norm(tgt - start)
        speed = dist * rng.uniform(2.3, 3.1)              # px/s, overshoots so they roll
        dirn = (tgt - start) / dist
        vz = rng.choice([rng.uniform(260, 420), rng.uniform(520, 760)], p=[0.6, 0.4])
        x = np.array([start[0], start[1], rng.uniform(40, 75)])
        v = np.array([dirn[0] * speed, dirn[1] * speed, vz])
        # Forward tumble about the axis across the throw, plus some wobble.
        roll_axis = np.array([-dirn[1], dirn[0], 0.0]) * -1
        w = roll_axis * rng.uniform(20, 32) + rng.normal(size=3) * 7
        out.append((x, v, random_quat(rng), w))
    return out


def slam(rng, n=5):
    """Dropped from right under the lens with lateral scatter."""
    out = []
    angs = rng.permutation(np.linspace(0, 2 * np.pi, n, endpoint=False)) + rng.uniform(0, 1)
    for a in angs:
        r = rng.uniform(8, 26)
        x = np.array([CENTER[0] + np.cos(a) * r, CENTER[1] + np.sin(a) * r * 0.7, rng.uniform(190, 240)])
        v = np.array([np.cos(a) * rng.uniform(160, 260), np.sin(a) * rng.uniform(110, 190), -rng.uniform(250, 400)])
        w = rng.normal(size=3) * 14
        out.append((x, v, random_quat(rng), w))
    return out


PRESETS = {e: (lambda rng, e=e: handful(rng, e)) for e in EDGES}
PRESETS["slam"] = slam


def simulate_throw(bodies, world, fps=24, max_time=4.5, substeps_out=6, settle_frames=5):
    steps_per_frame = int(round(1 / (world.dt * fps)))
    sample_every = max(1, steps_per_frame // substeps_out)
    frames, metrics = [], []
    still = [0] * len(bodies)
    for f in range(int(max_time * fps)):
        for b in bodies:
            b.impact = 0.0
        subs = []
        touch = [False] * len(bodies)
        for s in range(steps_per_frame):
            step(bodies, world)
            for i, b in enumerate(bodies):
                touch[i] |= b.touching
            if (s + 1) % sample_every == 0 or s == steps_per_frame - 1:
                subs.append([(b.x.copy(), b.q.copy()) for b in bodies])
        frames.append(subs)
        metrics.append([dict(touch=touch[i], speed=float(np.linalg.norm(b.v[:2])),
                             tumble=float(np.linalg.norm(b.w[:2])), impact=float(b.impact),
                             z=float(b.x[2])) for i, b in enumerate(bodies)])
        for i, b in enumerate(bodies):
            low = b.x[2] < b.size * 1.9
            if low and np.linalg.norm(b.v) < 14 and np.linalg.norm(b.w) < 0.9:
                still[i] += 1
            else:
                still[i] = 0
            if still[i] > 2:
                b.asleep = True
        if all(b.asleep for b in bodies):
            break
    targets = []
    for b in bodies:
        qt = flat_rest_quat(b)
        xt = b.x.copy()
        xt[2] = rest_height(b, qt)
        targets.append((xt, qt))
    last = frames[-1][-1]
    for k in range(1, settle_frames + 1):
        t = k / settle_frames
        t = t * t * (3 - 2 * t)
        frames.append([[(x0 + (xt - x0) * t, slerp(q0, qt, t)) for (x0, q0), (xt, qt) in zip(last, targets)]])
        metrics.append([dict(touch=True, speed=0.0, tumble=0.0, impact=0.0, z=float(xt[2])) for xt, _ in targets])
    return frames, metrics


def check(frames, metrics, size, fps=24):
    """Return None if the throw is good, else the reason it was rejected."""
    dur = len(frames) / fps
    if not 1.2 < dur < 4.0:
        return f"duration {dur:.2f}s"
    n = len(frames[0][-1])
    zmax = max(st[0][2] for fr in frames for sub in fr for st in sub)
    if zmax > 285:
        return "too close to lens"
    final = [x for x, _ in frames[-1][-1]]
    for x in final:
        if not (26 < x[0] < 294 and 24 < x[1] < 158):
            return "rests off screen"
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(final[i][:2] - final[j][:2]) < size * 2.25:
                return "resting dice too close"
    for i in range(n):
        landed = False
        slide = tumble = 0.0
        for m in metrics:
            mi = m[i]
            landed |= mi["touch"]
            if not landed:
                continue
            tumble += mi["tumble"] / fps
            # Moving along the felt without turning over = a slide.
            if mi["touch"] and mi["speed"] > 30 and mi["tumble"] < 4.5:
                slide += 1
        if slide > 2:
            return f"die {i} slides"
        if tumble < np.pi * 1.2:
            return f"die {i} barely tumbles"
    return None


def run_throw(preset, seed, size=15.0, fps=24, tries=200, n=5):
    shape = get_shape("d6")
    reasons = {}
    for k in range(tries):
        rng = np.random.default_rng(seed * 10007 + k)
        init = PRESETS[preset](rng)[:n]
        bodies = [Body(shape, size, x.astype(float), v.astype(float), q.astype(float), w.astype(float))
                  for x, v, q, w in init]
        frames, metrics = simulate_throw(bodies, World(**AGGRESSIVE), fps=fps)
        why = check(frames, metrics, size, fps)
        if why is None:
            return shape, frames, metrics, k
        reasons[why.split(" ", 2)[-1] if why.startswith("die") else why] = reasons.get(why, 0) + 1
    raise RuntimeError(f"no good throw for {preset}/{seed}: {reasons}")
