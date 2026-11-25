"""Convex dice shapes defined as intersections of half-spaces.

Every die is described by its face planes (unit normal n, offset d, so the
solid is n.p <= d for all faces). Bevel planes are added on edges and
corners so silhouettes read as slightly rounded at pixel scale, like the
reference art. Physics uses the unbevelled vertices.
"""
from dataclasses import dataclass, field
import itertools
import numpy as np


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _vertices_from_planes(N, D, eps=1e-7):
    verts = []
    for i, j, k in itertools.combinations(range(len(N)), 3):
        A = np.stack([N[i], N[j], N[k]])
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        p = np.linalg.solve(A, np.array([D[i], D[j], D[k]]))
        if np.all(N @ p <= D + 1e-6):
            if not any(np.linalg.norm(p - q) < 1e-5 for q in verts):
                verts.append(p)
    return np.array(verts)


@dataclass
class DieShape:
    name: str
    normals: np.ndarray          # (F,3) main face normals
    offsets: np.ndarray          # (F,)  main face offsets (inradius, unit size)
    labels: list                 # value printed on each face
    verts: np.ndarray            # (V,3) physics vertices (unit size)
    face_u: np.ndarray           # (F,3) in-plane "right" axis for decals
    face_v: np.ndarray           # (F,3) in-plane "up" axis for decals
    bevel_n: np.ndarray = field(default=None)   # (B,3)
    bevel_d: np.ndarray = field(default=None)   # (B,)
    decal: str = "number"        # "pips" or "number"
    inertia_k: float = 0.6       # I = k * m * size^2
    glyph_scale: float = 0.16    # glyph cell size in unit-size coordinates
    rot_group: list = field(default=None)

    @property
    def all_n(self):
        return np.concatenate([self.normals, self.bevel_n])

    @property
    def all_d(self):
        return np.concatenate([self.offsets, self.bevel_d])

    def up_face(self, R):
        """Index of the face whose world normal points most toward +z."""
        return int(np.argmax((self.normals @ R.T)[:, 2]))


def _face_axes(normals, prefer=(0, 0, 1)):
    us, vs = [], []
    for n in normals:
        ref = np.array(prefer, float)
        if abs(n @ ref) > 0.9:
            ref = np.array([0, 1.0, 0])
        u = _unit(np.cross(ref, n))
        v = np.cross(n, u)
        us.append(u)
        vs.append(v)
    return np.array(us), np.array(vs)


def _add_bevels(shape, edge_cut, corner_cut):
    N, D, V = shape.normals, shape.offsets, shape.verts
    bn, bd = [], []
    # Edges: pairs of faces that share two vertices.
    on_face = [set(np.where(np.abs(V @ n - d) < 1e-5)[0]) for n, d in zip(N, D)]
    for i, j in itertools.combinations(range(len(N)), 2):
        shared = on_face[i] & on_face[j]
        if len(shared) >= 2:
            m = _unit(N[i] + N[j])
            p = V[list(shared)[0]]
            bn.append(m)
            bd.append(m @ p - edge_cut)
    for p in V:
        m = _unit(p)
        bn.append(m)
        bd.append(m @ p - corner_cut)
    shape.bevel_n = np.array(bn)
    shape.bevel_d = np.array(bd)
    return shape


def _rotation_group(normals):
    """All proper rotations mapping the face-normal set onto itself."""
    N = normals
    a = N[0]
    # A second normal not parallel to a.
    b_idx = next(i for i in range(1, len(N)) if abs(N[i] @ a) < 0.999)
    b = N[b_idx]
    ang = a @ b

    def frame(x, y):
        e1 = _unit(x)
        e2 = _unit(y - (y @ e1) * e1)
        return np.stack([e1, e2, np.cross(e1, e2)], 1)

    F0 = frame(a, b)
    rots = []
    for i in range(len(N)):
        for j in range(len(N)):
            if i == j or abs(N[i] @ N[j] - ang) > 1e-6:
                continue
            Q = frame(N[i], N[j]) @ F0.T
            if np.linalg.det(Q) < 0.5:
                continue
            M = N @ Q.T
            if np.all(np.min(np.linalg.norm(M[:, None] - N[None], axis=2), 1) < 1e-5):
                if not any(np.allclose(Q, R, atol=1e-6) for R in rots):
                    rots.append(Q)
    return rots


def make_d6(bevel=0.13):
    N = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    D = np.ones(6)
    labels = [3, 4, 2, 5, 1, 6]  # opposite faces sum to 7
    V = _vertices_from_planes(N, D)
    u, v = _face_axes(N)
    s = DieShape("d6", N, D, labels, V, u, v, decal="pips", inertia_k=2 / 3)
    _add_bevels(s, edge_cut=bevel * 0.75, corner_cut=bevel * 1.5)
    s.rot_group = _rotation_group(N)
    return s


def make_d8():
    N = _unit(np.array(list(itertools.product([1, -1], repeat=3)), float))
    D = np.full(8, 1 / np.sqrt(3))
    # Opposite faces sum to 9.
    labels = [0] * 8
    vals = iter([1, 2, 3, 4])
    for i in range(8):
        if labels[i]:
            continue
        j = int(np.argmin(N @ N[i]))
        labels[i] = next(vals)
        labels[j] = 9 - labels[i]
    V = _vertices_from_planes(N, D)
    V = V / np.max(np.linalg.norm(V, axis=1)) * 1.25
    D = D * 1.25
    u, v = _face_axes(N)
    s = DieShape("d8", N, D, labels, V, u, v, inertia_k=0.45, glyph_scale=0.16)
    _add_bevels(s, edge_cut=0.07, corner_cut=0.16)
    s.rot_group = _rotation_group(N)
    return s


def _icosa_verts():
    p = (1 + 5 ** 0.5) / 2
    vs = []
    for a, b in itertools.product([1, -1], repeat=2):
        vs += [(0, a, b * p), (a, b * p, 0), (b * p, 0, a)]
    return _unit(np.array(vs, float))


def _dodeca_verts():
    p = (1 + 5 ** 0.5) / 2
    vs = list(itertools.product([1, -1], repeat=3))
    for a, b in itertools.product([1, -1], repeat=2):
        vs += [(0, a / p, b * p), (a / p, b * p, 0), (b * p, 0, a / p)]
    return _unit(np.array(vs, float))


def _opposite_labels(N, total):
    labels = [0] * len(N)
    nxt = 1
    for i in range(len(N)):
        if labels[i]:
            continue
        j = int(np.argmin(N @ N[i]))
        labels[i] = nxt
        labels[j] = total - nxt
        nxt += 1
    return labels


def make_d12():
    N = _icosa_verts()          # dodecahedron face normals
    D = np.ones(12)
    labels = _opposite_labels(N, 13)
    V = _vertices_from_planes(N, D)
    sc = 1.15 / np.max(np.linalg.norm(V, axis=1))
    V, D = V * sc, D * sc
    u, v = _face_axes(N)
    s = DieShape("d12", N, D, labels, V, u, v, inertia_k=0.4, glyph_scale=0.13)
    _add_bevels(s, edge_cut=0.05, corner_cut=0.1)
    s.rot_group = _rotation_group(N)
    return s


def make_d20():
    N = _dodeca_verts()         # icosahedron face normals
    D = np.ones(20)
    labels = _opposite_labels(N, 21)
    V = _vertices_from_planes(N, D)
    sc = 1.2 / np.max(np.linalg.norm(V, axis=1))
    V, D = V * sc, D * sc
    u, v = _face_axes(N)
    s = DieShape("d20", N, D, labels, V, u, v, inertia_k=0.4, glyph_scale=0.115)
    _add_bevels(s, edge_cut=0.05, corner_cut=0.12)
    s.rot_group = _rotation_group(N)
    return s


SHAPES = {"d6": make_d6, "d8": make_d8, "d12": make_d12, "d20": make_d20}


def get_shape(name):
    return SHAPES[name]()
