"""Small impulse-based rigid body sim for dice on a walled table.

World units are native pixels (the 320x180 canvas). X right, Y down the
screen, Z up toward the camera. The table surface is z = 0.
"""
from dataclasses import dataclass, field
import numpy as np


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def slerp(q0, q1, t):
    d = q0 @ q1
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


@dataclass
class Body:
    shape: object
    size: float                     # half-size in pixels
    x: np.ndarray                   # position
    v: np.ndarray                   # linear velocity
    q: np.ndarray                   # orientation quaternion
    w: np.ndarray                   # angular velocity (world)
    mass: float = 1.0
    inside: np.ndarray = field(default_factory=lambda: np.zeros(4, bool))
    asleep: bool = False

    @property
    def R(self):
        return quat_to_mat(self.q)

    @property
    def inv_I(self):
        return 1.0 / (self.shape.inertia_k * self.mass * self.size ** 2)

    def world_verts(self):
        return self.x + (self.shape.verts * self.size) @ self.R.T

    @property
    def radius(self):
        return self.size * np.max(np.linalg.norm(self.shape.verts, axis=1))


@dataclass
class World:
    bounds: tuple = (14.0, 306.0, 12.0, 168.0)   # xmin, xmax, ymin, ymax
    gravity: float = 2600.0
    restitution: float = 0.5
    wall_restitution: float = 0.55
    friction: float = 0.38
    dt: float = 1 / 600
    lin_damp: float = 0.35
    ang_damp: float = 0.55
    floor_spin_damp: float = 3.5   # extra damping of yaw spin when touching


def _planes(world):
    xmin, xmax, ymin, ymax = world.bounds
    # (normal, offset, wall index) with solid on the side n.p < offset
    return [
        (np.array([0, 0, 1.0]), 0.0, -1),
        (np.array([1.0, 0, 0]), xmin, 0),
        (np.array([-1.0, 0, 0]), -xmax, 1),
        (np.array([0, 1.0, 0]), ymin, 2),
        (np.array([0, -1.0, 0]), -ymax, 3),
    ]


def _resolve_plane(b, n, off, e, mu, world):
    P = b.world_verts()
    depth = off - P @ n
    hit = np.where(depth > 0)[0]
    if len(hit) == 0:
        return False
    # Positional correction.
    b.x = b.x + n * depth[hit].max() * 0.9
    invm, invI = 1.0 / b.mass, b.inv_I
    order = hit[np.argsort(-depth[hit])]
    for _ in range(3):
        for i in order:
            r = P[i] - b.x
            vp = b.v + np.cross(b.w, r)
            vn = vp @ n
            if vn >= 0:
                continue
            ee = e if vn < -60 else 0.0
            rn = np.cross(r, n)
            jn = -(1 + ee) * vn / (invm + invI * (rn @ rn))
            vt = vp - vn * n
            vtl = np.linalg.norm(vt)
            J = jn * n
            if vtl > 1e-6:
                t = vt / vtl
                rt = np.cross(r, t)
                jt = vtl / (invm + invI * (rt @ rt))
                J = J - min(jt, mu * jn) * t
            b.v = b.v + J * invm
            b.w = b.w + invI * np.cross(r, J)
    return True


def _resolve_pair(a, b, e=0.5):
    """Cheap sphere-proxy collision between two dice."""
    d = b.x - a.x
    dist = np.linalg.norm(d)
    rad = (a.radius + b.radius) * 0.82
    if dist >= rad or dist < 1e-6:
        return
    n = d / dist
    pen = rad - dist
    a.x = a.x - n * pen / 2
    b.x = b.x + n * pen / 2
    rv = (b.v - a.v) @ n
    if rv < 0:
        j = -(1 + e) * rv / 2
        a.v = a.v - j * n
        b.v = b.v + j * n
        # Trade a little spin so the hit reads.
        kick = np.cross(n, np.array([0, 0, 1.0])) * j * 0.02
        a.w = a.w - kick
        b.w = b.w + kick
        a.asleep = b.asleep = False


def step(bodies, world):
    planes = _planes(world)
    for b in bodies:
        if b.asleep:
            continue
        b.v = b.v + np.array([0, 0, -world.gravity]) * world.dt
        b.x = b.x + b.v * world.dt
        wl = np.linalg.norm(b.w)
        if wl > 1e-9:
            ax = b.w / wl
            h = wl * world.dt / 2
            dq = np.array([np.cos(h), *(ax * np.sin(h))])
            b.q = quat_mul(dq, b.q)
            b.q /= np.linalg.norm(b.q)
        # Walls switch on once the die has entered that side of the table.
        xmin, xmax, ymin, ymax = world.bounds
        r = b.radius
        b.inside |= np.array([b.x[0] - r > xmin, b.x[0] + r < xmax, b.x[1] - r > ymin, b.x[1] + r < ymax])
        touching = False
        for n, off, wi in planes:
            if wi >= 0 and not b.inside[wi]:
                continue
            e = world.restitution if wi < 0 else world.wall_restitution
            hit = _resolve_plane(b, n, off, e, world.friction, world)
            if wi < 0 and hit:
                touching = True
        damp_l = world.lin_damp * (3.0 if touching else 0.15)
        damp_a = world.ang_damp * (3.0 if touching else 0.15)
        b.v = b.v * np.exp(-damp_l * world.dt)
        b.w = b.w * np.exp(-damp_a * world.dt)
        if touching:
            b.w[2] *= np.exp(-world.floor_spin_damp * world.dt)
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            _resolve_pair(bodies[i], bodies[j])


def flat_rest_quat(b):
    """Nearest orientation where the up face is exactly level, keeping yaw."""
    R = b.R
    f = b.shape.up_face(R)
    n_world = R @ b.shape.normals[f]
    target = np.array([0, 0, 1.0])
    axis = np.cross(n_world, target)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return b.q.copy()
    ang = np.arctan2(s, n_world @ target)
    axis /= s
    dq = np.array([np.cos(ang / 2), *(axis * np.sin(ang / 2))])
    return quat_mul(dq, b.q)


def rest_height(b, q):
    R = quat_to_mat(q)
    return -np.min(((b.shape.verts * b.size) @ R.T)[:, 2])


def simulate(bodies, world, fps=24, max_time=6.0, settle_frames=5, substeps_out=6):
    """Run until all bodies rest. Returns frames: list (per frame) of list of
    sub-states (per sub-sample) of list of (x, q) per body. The last sub-state
    of each frame is the frame's actual state; earlier ones are for smears."""
    steps_per_frame = int(round(1 / (world.dt * fps)))
    sample_every = max(1, steps_per_frame // substeps_out)
    frames = []
    still = [0] * len(bodies)
    nframes = int(max_time * fps)
    for f in range(nframes):
        subs = []
        for s in range(steps_per_frame):
            step(bodies, world)
            if (s + 1) % sample_every == 0 or s == steps_per_frame - 1:
                subs.append([(b.x.copy(), b.q.copy()) for b in bodies])
        frames.append(subs)
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
    # Ease every body into an exactly level pose so the final frame is crisp.
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
        st = []
        for (x0, q0), (xt, qt) in zip(last, targets):
            st.append((x0 + (xt - x0) * t, slerp(q0, qt, t)))
        frames.append([st])
    return frames
