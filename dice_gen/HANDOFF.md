# Herdsman dice: handoff

Everything built so far for the pixel-art dice rolls, and what is left, so the
work can continue inside the GameMaker Shepherd/Herdsman project.

## Where it came from

Ten reference frames of one d6 roll (`reference/ref_01..10.png`, 1920x1080,
drawn on a 6px grid = native 320x180). The look that was matched:

- cream faces, cooler gray side faces, maroon outline and maroon lines between faces, navy pips
- slightly rounded corners, slight perspective from a camera above the table centre
- soft shadow falling to the lower left
- fast frames turn into jagged cream "drip" smears (ref 6 and 7)
- a die high in the air swells toward the lens (ref 8)

## What works today (`generate.py`)

Single throws of 1 to 3 d6, physics simulated and rendered at native 320x180.
The landing result is chosen up front (faces are relabelled with one of the
cube's 24 symmetry rotations after the throw settles, so motion stays real).

```bash
pip install numpy pillow          # plus ffmpeg on PATH for gif/mp4 previews
python generate.py --list
python generate.py --motion drop_in --result 4 --palette onyx --seed 21
python generate.py --batch --out gallery
```

- 10 motions: drop_in, side_left, side_right, bank_shot, high_toss, slam, skid, spinner, pair, scatter3
- 8 palettes: bone, onyx, gold, jade, royal, ivory_red, amethyst, frost (`dicegen/palettes.py`)
- per clip: `preview.gif`, `preview_1080p.mp4`, transparent `sprites/die_###.png`,
  `sprites/shadow_###.png` (alpha), sheets, and `meta.json` with per-frame positions
- `gallery/` holds the 12 approved clips (the user said "these are really good")

## Code map

| File | Role |
| --- | --- |
| `dicegen/physics.py` | impulse rigid body sim on a walled table; floor/wall contacts, die-to-die contacts (vertex-in-polyhedron + core sphere), sleep, ease to a level rest pose |
| `dicegen/render.py` | per-pixel ray cast of a bevelled convex die, 5-tone ramp, outline/edge lines, pips/numerals, smear trails, projected soft shadow |
| `dicegen/shapes.py` | d6 (used) plus d8/d12/d20 (built, not tuned) with rotation groups |
| `dicegen/motions.py` | single-throw presets |
| `dicegen/animate.py` | result selection, frame assembly, exporters |
| `dicegen/throws.py` | NEW, unfinished: five-dice "handful" presets and the always-tumble validator |
| `assets/felt_320x180.png` | clean felt plate cut from the reference (to be replaced by grass) |

## New requirements (from the user, not done yet)

1. **Five dice rolling at once**, every throw.
2. **Always a roll, never a slide.** Some current clips skate across the felt. Every die must visibly tumble while it moves.
3. **More aggressive**, really tossed in, like the reference frames (fast entry with smears, big pop toward the lens, hard bounce).
4. **Grass background**: a small grassy field built from the grass textures already in the GameMaker project, replacing the felt. Shadows should tint dark green on grass, not maroon.
5. **A dice engine for the game** where the dice being thrown can be swapped. Decided with the user:
   - Engine style: **baked throw library + GameMaker player** (recommended option, chosen).
   - What can change per die: **colours/materials, custom face symbols, face values, and shapes** (shapes can come later; d6 first).

## Status of the five-dice work (`dicegen/throws.py`)

- `handful(rng, edge)` flings 5 dice from one of 8 edges/corners with forward tumble spin; `slam` drops them from under the lens.
- `AGGRESSIVE` world settings: gravity 4200, friction 0.6, restitution 0.45.
- `check()` rejects throws that run too long, go too near the lens, rest off screen or too close together,
  a die that slides (touching, moving > 30 px/s, tumbling < 4.5 rad/s for more than 2 frames), or barely tumbles (< 216 degrees after landing).
- **Blocking bug:** with 5 dice nearly every throw hits the 4.5 s cap without sleeping. One die on its face rocks forever at about 4 rad/s
  (sequential corner impulses pump energy). Tried applying multi-corner contacts at their centroid: it made it worse and was reverted.
  Ideas: split impulse / Baumgarte-free position correction, accumulate impulses per contact across iterations (proper sequential impulse with clamping),
  restitution 0 below a velocity threshold for every contact, or a sleep rule based on "face nearly level + low kinetic energy for 0.15 s".
- Sim speed is 1.5 to 3 s per five-dice throw in pure numpy; vectorising the pair checks or moving physics to numba would help, since the validator retries many seeds.

## Engine design (agreed, not built)

Bake motion offline, render skins at runtime:

1. **Bake** a library of five-dice throws (say 16 to 24). For each frame and die, ray-cast a small G-buffer sprite instead of final colours:
   - `R = code*16 + tone` where code 0-5 = physical face, 6 = bevel, 7 = edge line, 8 = outline, 9 = smear body, 10 = smear tip; tone 0-4 = lighting step
   - `G, B` = face-local u, v (0-255); `A` = 255 where the die is
   - pack every die-frame crop into one PNG atlas per throw, plus a soft shadow atlas (white RGB, alpha = shadow)
   - JSON per throw: fps, frame count, per frame per die `[atlas x, y, w, h, screen x, y, z order]`, each die's final up physical face, impact events per frame (for sound and screen shake)
   - one shared table of the 24 cube rotations: for each, `perm[6]` (physical face to skin face) and `rot[6]` (quarter-turns of the face image)
2. **GameMaker player**:
   - load the atlases from Included Files with `sprite_add`, texture filtering off
   - a die definition = `{ faces: [ {value, icon sprite} x6 ], ramp: 5 colours, outline, edge, ink }`
   - build each die's 6-cell skin atlas at runtime by drawing its face icons onto a surface, then `sprite_create_from_surface`
   - on roll: pick a random throw, assign the 5 dice to its slots, pick each die's landing face (random or forced), choose one of the 4 rotations whose `perm[up_face]` is that face, pass `perm`/`rot` as uniforms
   - fragment shader decodes the G-buffer: face code to skin cell via `perm`, rotate uv by `rot`, sample the skin; ink texels keep their colour darkened by tone, empty texels take `ramp[tone]`; outline/edge/smear codes take their palette colours
   - draw the shadow atlas first with a dark tint, then dice in baked z order, into a 320x180 surface scaled up with nearest filtering
   - report results (face values) when the last frame plays
3. **Check the shader outside GameMaker** with a small WebGL page using the identical GLSL ES fragment shader before porting.

## Suggested next steps, in order

1. Copy the grass textures from the project into `assets/grass/` and compose a 320x180 field plate (replace `FELT` in `animate.py`, retint `SHADOW_TINT`).
2. Fix five-dice settling in `physics.py`, then run `throws.run_throw(preset, seed)` across all presets until each passes `check()`.
3. Render five-dice previews on grass and review the motion against the reference frames.
4. Build the bake step and the GameMaker player per the design above.
