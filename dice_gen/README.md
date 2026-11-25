# Pixel dice roll generator

Procedural pixel-art d6 roll animations in the style of the frames in `reference/`.
Each throw is a real rigid-body simulation rendered on a native 320x180 grid
(shown 6x at 1920x1080), with the result chosen up front: after the throw
settles, the faces are relabeled using one of the cube's 24 symmetry rotations,
so the motion stays physical and the requested number lands face up.

## Usage

```bash
pip install numpy pillow        # ffmpeg is used for GIF and MP4 previews
python generate.py --list
python generate.py --motion drop_in --result 6 --palette bone --seed 1
python generate.py --motion pair --result 3,4 --palette bone,royal --seed 10
python generate.py --batch --out gallery
```

Same seed, same throw. Change the seed for a new variation of a motion.

**Motions:** drop_in (the reference toss), side_left, side_right, bank_shot,
high_toss, slam, skid, spinner, pair, scatter3

**Palettes:** bone (reference), onyx, gold, jade, royal, ivory_red, amethyst, frost

## Output per clip

| File | What |
| --- | --- |
| `preview.gif` | 640x360 preview on the felt |
| `preview_1080p.mp4` | 1920x1080 preview on the felt |
| `sprites/die_###.png` | transparent die layer per frame, native 320x180 |
| `sprites/shadow_###.png` | soft shadow alpha per frame (draw as a dark tint under the die) |
| `die_sheet.png`, `shadow_sheet.png` | the same frames as 8-column sheets |
| `meta.json` | fps, results, palettes, and per-frame screen position and height of each die |

Sprites are at native resolution; scale them up with nearest-neighbour filtering.

## How it works

- `dicegen/physics.py` impulse-based rigid body sim on a walled table (gravity, restitution, friction, sleep, final ease to a level rest pose)
- `dicegen/render.py` per-pixel ray cast of a bevelled convex die through a gentle perspective camera, 5-tone toon ramp, maroon outline and edge lines, pips, jagged smear trails for fast frames, projected soft shadow
- `dicegen/motions.py` throw presets (initial position, velocity, spin, optional world overrides)
- `dicegen/animate.py` result selection, frame assembly, exporters
- `dicegen/shapes.py` also defines d8, d12 and d20 (numbered); those are experimental and not tuned yet
