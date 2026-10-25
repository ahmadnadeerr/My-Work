#!/usr/bin/env python3
"""Animate the mosaic behind a portrait with Control-style shifting glitches.

The person is never re-rendered. Every output pixel is

    photo + (1 - alpha) * (animated_panel - static_panel)

so wherever the segmentation matte says "person" (alpha == 1) the original
photo pixels pass through untouched. Only the glass mosaic moves, and it
never stops:

  * the medallion splits into three rings that turn continuously in
    alternating directions, lurching forward on a one-second tick,
  * the 12-point rosettes and the small blue stars spin like counter-rotating
    gears, mirrored left/right, their ticks rippling out from the centre,
  * "resonance" ripples run out of and back into the medallion,
  * Hiss-style glitch bursts: RGB split, horizontal tears, macroblock
    corruption, red tint, scanlines and time stutter.

Every element turns a whole number of times per loop, so the clip loops
seamlessly.

The panel geometry below was measured on the original 1890x2576 photo.

Usage:
    python glitch_portrait.py photo.jpg out.mp4 [--mask mask.png] [--workers 4]
    python glitch_portrait.py photo.jpg out.mp4 --music song.mp3 --start 25.1 --end 54.8

With --music the moves follow the song instead (see music_sync.py) and the
song clip is muxed into the output.
"""
import argparse
import math
import multiprocessing as mp
import os
import subprocess

import cv2
import numpy as np

FPS = 30
LOOP = 16.0
BEATS = 16  # mechanical ticks per loop

# Glass corners in the photo (TL, TR, BR, BL) and the rectified panel size.
PANEL_QUAD = np.float32([[32, 517], [1574, 507], [1556, 2049], [58, 2033]])
S = 1536
C = 771.0  # medallion centre in rectified coords (x == y)

# Medallion rings: (inner radius, outer radius, turns per loop, beat offset).
# The red band (457-490) stays put.
RINGS = [(0, 170, 1, 0.0), (170, 365, -1, 0.33), (365, 455, 1, 0.66)]

# (x, y, radius) of the 12-point rosettes that sit clear of bars and edges.
ROSETTES = [
    (284, 77, 73), (1255, 77, 73), (656, 134, 85), (883, 137, 82),
    (443, 309, 73), (132, 309, 100), (1410, 309, 100), (1095, 312, 70),
    (444, 1234, 73), (1098, 1235, 74), (132, 1231, 100), (1410, 1231, 100),
    (654, 1421, 87), (890, 1410, 89), (290, 1466, 66), (1253, 1466, 66),
]

# Resonance ripples: (start, start radius, speed px/s, displacement px).
# Negative speed travels from the corners into the medallion.
WAVES = [(0.0, 0.0, 950.0, 6.0), (4.0, 1150.0, -950.0, 7.0),
         (8.0, 0.0, 950.0, 6.0), (12.0, 1150.0, -950.0, 7.0)]

# (start, duration, intensity, hiss)
GLITCHES = [
    (0.95, 0.12, 0.50, False),
    (2.50, 0.20, 0.65, False),
    (3.90, 0.08, 0.35, False),
    (5.05, 0.45, 1.00, True),
    (6.60, 0.08, 0.40, False),
    (7.95, 0.14, 0.60, False),
    (9.70, 0.20, 0.70, False),
    (11.20, 0.07, 0.30, False),
    (13.05, 0.40, 0.90, True),
    (14.60, 0.10, 0.45, False),
]

# Red flashes from the centre as ripples leave or arrive: (time, strength)
PULSES = [(0.0, 0.8), (4.0 + 1150.0 / 950.0, 1.0), (8.0, 0.8), (12.0 + 1150.0 / 950.0, 1.0)]


def smootherstep(x):
    return x * x * x * (x * (6 * x - 15) + 10)


def spin(t, turns, phase, mix=0.6):
    """Angle in degrees of a piece turning `turns` times per loop.

    A steady turn blended with a stepped one, so it never stops but lurches
    forward on every beat. `phase` shifts the ticks, in loops.
    """
    u = t / LOOP + phase
    k = math.floor(u * BEATS)
    stepped = (k + smootherstep(u * BEATS - k)) / BEATS
    return 360.0 * turns * ((1 - mix) * u + mix * stepped)


def ripple_delay(x, y):
    """Seconds after the medallion that a tick reaches (x, y)."""
    return (math.hypot(x - C, y - C) - 480) / 1200


def ring_spin(i):
    _, _, turns, beat = RINGS[i]
    return turns, beat / BEATS


def rosette_spin(x, y):
    return (1 if x < C else -1), -ripple_delay(x, y) / LOOP


def star_spin(x, y):
    # Opposite to the rosettes in the same half, ticking on the off-beat.
    return (-2 if x < C else 2), -ripple_delay(x, y) / LOOP + 0.5 / BEATS


def find_blue_stars(base):
    """Centres and radii of the small blue six-point stars in the tessellation."""
    img = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.5), cv2.COLOR_BGR2HSV).astype(int)
    blue = ((hsv[..., 0] >= 95) & (hsv[..., 0] <= 130) & (hsv[..., 1] > 45)).astype(np.uint8)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform(blue, cv2.DIST_L2, 5)
    peaks = (dist == cv2.dilate(dist, np.ones((31, 31)))) & (dist > 6)
    pad = 80
    g = cv2.copyMakeBorder(cv2.GaussianBlur(base, (0, 0), 1.2), pad, pad, pad, pad, cv2.BORDER_REFLECT)

    def asymmetry(cx, cy, r=34):
        x0, y0 = int(cx) + pad - r - 4, int(cy) + pad - r - 4
        sub = g[y0:y0 + 2 * r + 9, x0:x0 + 2 * r + 9]
        c = (cx + pad - x0, cy + pad - y0)
        M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), 60, 1)
        rot = cv2.warpAffine(sub, M, sub.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
        yy, xx = np.ogrid[:sub.shape[0], :sub.shape[1]]
        return np.abs(rot - sub)[(xx - c[0]) ** 2 + (yy - c[1]) ** 2 < r * r].mean()

    stars = []
    for y, x in zip(*np.nonzero(peaks)):
        if (math.hypot(x - C, y - C) < 500 or min(abs(x - C), abs(y - C)) < 60
                or min(x, y, S - 1 - x, S - 1 - y) < 40
                or any(math.hypot(x - rx, y - ry) < rr + 30 for rx, ry, rr in ROSETTES)
                or any(math.hypot(x - sx, y - sy) < 40 for sx, sy in stars)):
            continue
        # Refine the centre so the star turns without wobbling.
        best = (asymmetry(x, y), x, y)
        for step in (4, 2, 1):
            bx, by = best[1], best[2]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    e = asymmetry(bx + dx * step, by + dy * step)
                    if e < best[0]:
                        best = (e, bx + dx * step, by + dy * step)
        stars.append((best[1], best[2]))

    out = []
    for i, (x, y) in enumerate(stars):
        nn = min((math.hypot(x - a, y - b) for j, (a, b) in enumerate(stars) if j != i), default=99)
        out.append((int(x), int(y), int(min(34, nn / 2 - 2))))
    return out


class LoopTimeline:
    """The self-contained 16 s loop with its own one-second beat."""
    duration = LOOP
    waves, pulses, glitches = WAVES, PULSES, GLITCHES

    def lag(self, t, t0):
        """Signed seconds since t0, wrapped onto the loop."""
        return (t - t0 + LOOP / 2) % LOOP - LOOP / 2

    def wrap(self, t):
        return t % LOOP

    def pieces(self, stars):
        p = [(C, C, r0, r1, ring_spin(i)) for i, (r0, r1, _, _) in enumerate(RINGS)]
        p += [(x, y, 0, r, rosette_spin(x, y)) for x, y, r in ROSETTES]
        p += [(x, y, 0, r, star_spin(x, y)) for x, y, r in stars]
        return p

    def angle(self, t, spec):
        return spin(t, *spec)

    def beat_glow(self, t):
        return 0.0

    def portal(self, t):
        return None


# ---------------------------------------------------------------- assets --

G = {}


def segment_person(photo_path):
    from PIL import Image
    from rembg import new_session, remove
    session = new_session("birefnet-portrait")
    mask = remove(Image.open(photo_path).convert("RGB"), session=session, only_mask=True)
    return np.array(mask)


def load_assets(photo_path, mask_path):
    photo = cv2.imread(photo_path)
    mask = cv2.imread(mask_path, 0) if mask_path else segment_person(photo_path)
    alpha = mask.astype(np.float32) / 255.0
    alpha[alpha > 0.96] = 1.0  # solid person pixels stay bit-exact

    Hm = cv2.getPerspectiveTransform(PANEL_QUAD, np.float32([[0, 0], [S, 0], [S, S], [0, S]]))
    panel = cv2.warpPerspective(photo, Hm, (S, S), flags=cv2.INTER_CUBIC).astype(np.float32) / 255

    # Rebuild the pattern hidden behind the person by mirroring the left half
    # across the medallion axis (the mosaic is left/right symmetric).
    occ = cv2.warpPerspective(mask, Hm, (S, S)) > 8
    occ = cv2.dilate(occ.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(np.float32)
    occ = cv2.GaussianBlur(occ, (0, 0), 4)[..., None]
    mirrored = panel[:, np.clip(2 * int(C) - np.arange(S), 0, S - 1)]
    base = panel * (1 - occ) + mirrored * occ

    # Split fixed lighting from the glass so moving pieces keep the spotlights.
    lum = base @ np.float32([0.114, 0.587, 0.299])
    light = cv2.GaussianBlur(lum, (0, 0), 80)
    light = np.maximum(light / light.mean(), 0.05)[..., None]
    albedo = base / light

    # Photo-space lookup into the panel for the bounding box of the glass.
    x0, y0 = np.floor(PANEL_QUAD.min(0)).astype(int) - 2
    x1, y1 = np.ceil(PANEL_QUAD.max(0)).astype(int) + 3
    uu, vv = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    w = Hm[2, 0] * uu + Hm[2, 1] * vv + Hm[2, 2]
    mx = (Hm[0, 0] * uu + Hm[0, 1] * vv + Hm[0, 2]) / w
    my = (Hm[1, 0] * uu + Hm[1, 1] * vv + Hm[1, 2]) / w
    edge = 5.0  # keep the outermost glass pixels original
    q = np.clip(np.minimum.reduce([mx - edge, my - edge, S - 1 - edge - mx, S - 1 - edge - my]), 0, 1)

    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    rad = np.hypot(xx - C, yy - C)
    ring_band = ((rad > 459) & (rad < 488)).astype(np.float32)

    G.update(
        photo=photo, alpha=alpha, base=base, albedo=albedo, light=light,
        roi=(x0, y0, x1, y1), mx=mx.astype(np.float32), my=my.astype(np.float32),
        keep=((1 - alpha[y0:y1, x0:x1]) * q)[..., None],
        rad=rad, xx=xx, yy=yy,
        ring_band=ring_band, ring_glow=cv2.GaussianBlur(ring_band, (0, 0), 7),
        stars=find_blue_stars(base),
    )
    G["base_warp"] = to_photo(base)
    if getattr(G.get("tl"), "portal_on", False):
        from portal import Portal
        G["portal"] = Portal(base, G)


def to_photo(panel):
    return cv2.remap(panel, G["mx"], G["my"], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


# ------------------------------------------------------------- effects --

def rotate_region(dst, src, cx, cy, r0, r1, ang, dang):
    """Paste src rotated by ang degrees about (cx, cy) into the r0..r1 annulus."""
    pad = r1 + 3
    x0, x1 = max(0, int(cx - pad)), min(S, int(cx + pad) + 1)
    y0, y1 = max(0, int(cy - pad)), min(S, int(cy + pad) + 1)
    # 180-degree shutter: smear across half the frame's angular travel.
    n = int(min(9, abs(dang) * 0.5 * math.pi / 180 * r1 / 2.5)) + 1
    acc = None
    for k in range(n):
        a = math.radians(ang + (dang * 0.5 * (k / (n - 1) - 0.5) if n > 1 else 0.0))
        c, s = math.cos(a), math.sin(a)
        A = np.array([[c, s], [-s, c]])
        b = np.array([cx, cy]) - A @ np.array([cx, cy]) + A @ np.array([x0, y0])
        M = np.hstack([A, b[:, None]])
        r = cv2.warpAffine(src, M, (x1 - x0, y1 - y0),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REFLECT)
        acc = r if acc is None else acc + r
    rot = acc / n
    d = G["rad"][y0:y1, x0:x1] if (cx, cy) == (C, C) else np.hypot(
        G["xx"][y0:y1, x0:x1] - cx, G["yy"][y0:y1, x0:x1] - cy)
    m = np.clip(r1 - d + 0.5, 0, 1)
    if r0 > 0:
        m *= np.clip(d - r0 + 0.5, 0, 1)
    sub = dst[y0:y1, x0:x1]
    sub[:] = sub * (1 - m[..., None]) + rot * m[..., None]
    # Dark groove at the cut, deeper the faster its rim is moving.
    groove = min(1.0, abs(dang) * math.pi / 180 * r1 / 3.0) * 0.4
    if groove > 0.01:
        g = np.exp(-((d - r1) / 1.4) ** 2)
        if r0 > 0:
            g += np.exp(-((d - r0) / 1.4) ** 2)
        sub *= (1 - groove * np.clip(g, 0, 1))[..., None]


def resonance_wave(t):
    """(front radius, displacement) of every ripple travelling at time t."""
    tl = G["tl"]
    fronts = [(r0 + tl.lag(t, t0) * speed, amp) for t0, r0, speed, amp in tl.waves
              if tl.lag(t, t0) >= 0]
    return [(r, a) for r, a in fronts if -60 < r < 1200]


def apply_waves(img, t):
    fronts = resonance_wave(t)
    if not fronts:
        return img, None
    rad = G["rad"]
    disp = np.zeros_like(rad)
    glow = np.zeros_like(rad)
    for front, amp in fronts:
        u = (rad - front) / 42.0
        g = np.exp(-u * u)
        fade = np.clip((1250 - front) / 300, 0, 1) * np.clip((front + 60) / 120, 0, 1)
        disp += amp * fade * u * g * 1.6
        glow += g * fade
    inv = 1.0 / np.maximum(rad, 1e-3)
    mapx = G["xx"] - disp * (G["xx"] - C) * inv
    mapy = G["yy"] - disp * (G["yy"] - C) * inv
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), glow


def pulse_env(t):
    e = 0.0
    for tp, k in G["tl"].pulses:
        dt = G["tl"].lag(t, tp)
        if -0.05 < dt < 2.0:
            e += k * (min(1.0, (dt + 0.05) / 0.05) if dt < 0 else math.exp(-dt / 0.35))
    return e


def glitch_state(t):
    for t0, d, k, hiss in G["tl"].glitches:
        if t0 <= t < t0 + d:
            return k, hiss
    return 0.0, False


def roll_inner(a, dx, m=8):
    a[:, m:S - m] = np.roll(a[:, m:S - m], dx, axis=1)


def hiss_map(img):
    lum = img @ np.float32([0.114, 0.587, 0.299])
    lum = np.clip(lum, 0, 1)[..., None]
    dark = np.float32([0.02, 0.01, 0.08])
    mid = np.float32([0.05, 0.03, 0.80])
    hi = np.float32([0.55, 0.60, 1.0])
    lo = np.clip(lum * 2, 0, 1)
    hi_w = np.clip(lum * 2 - 1, 0, 1)
    return dark * (1 - lo) + mid * lo * (1 - hi_w) + hi * hi_w


def apply_glitch(img, k, hiss, rng):
    out = img.copy()
    s = int(round(2 + 10 * k * rng.uniform(0.5, 1.0)))
    roll_inner(out[..., 2], s)
    roll_inner(out[..., 0], -s)

    for _ in range(int(2 + 11 * k)):
        h = int(rng.integers(4, 12 + int(110 * k)))
        y = int(rng.integers(8, S - 8 - h))
        dx = int(rng.normal(0, 75 * k))
        band = out[y:y + h]
        roll_inner(band, dx)
        if rng.random() < 0.3:
            ch = int(rng.integers(0, 3))
            roll_inner(band[..., ch], int(rng.integers(-25, 26)))
        if rng.random() < 0.35:
            band *= rng.uniform(0.75, 1.3)

    for _ in range(int(rng.integers(0, 2 + int(7 * k)))):
        bw = int(rng.integers(5, 5 + int(28 * k))) * 8
        bh = int(rng.integers(3, 3 + int(18 * k))) * 8
        x, y = int(rng.integers(8, S - 8 - bw)), int(rng.integers(8, S - 8 - bh))
        sx, sy = int(rng.integers(8, S - 8 - bw)), int(rng.integers(8, S - 8 - bh))
        blk = img[sy:sy + bh, sx:sx + bw].copy()
        if rng.random() < 0.55:
            f = int(rng.integers(8, 17))
            small = cv2.resize(blk, (max(1, bw // f), max(1, bh // f)), interpolation=cv2.INTER_AREA)
            blk = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
        if hiss and rng.random() < 0.5:
            blk = hiss_map(blk)
        out[y:y + bh, x:x + bw] = blk

    if hiss:
        amt = rng.uniform(0.25, 0.65) * k
        out = out * (1 - amt) + hiss_map(out) * amt

    out[::3] *= 1 - 0.12 * k
    return out


# ------------------------------------------------------------- render --

def panel_at(t):
    """Lit panel (float BGR) with all mechanical motion at time t."""
    dt = 1.0 / FPS
    src = G["albedo"]
    alb = src.copy()
    tl = G["tl"]
    for x, y, r0, r1, spec in tl.pieces(G["stars"]):
        a, a2 = tl.angle(t, spec), tl.angle(t + dt, spec)
        rotate_region(alb, src, float(x), float(y), r0, r1, a, a2 - a)

    alb, glow = apply_waves(alb, t)
    img = alb * G["light"]
    red = np.float32([0.10, 0.18, 1.0])
    if glow is not None:
        img = img * (1 + 0.10 * glow[..., None]) + glow[..., None] * red * 0.07

    e = pulse_env(t)
    k, hiss = glitch_state(t)
    ring = min(1.5, e + (0.6 * k if hiss else 0.0) + tl.beat_glow(t))
    if ring > 0.01:
        img = img * (1 + 0.45 * ring * G["ring_band"][..., None])
        img = img + G["ring_glow"][..., None] * red * 0.35 * ring
    if e > 0.01:
        core = np.exp(-(G["rad"] / 38.0) ** 2)
        img = img + core[..., None] * red * 0.7 * e
    state = tl.portal(t)
    if state:
        img = G["portal"].blend(img, state, t)
    return img


def render_frame(n):
    t = n / FPS
    rng = np.random.default_rng(1000 + n)
    k, hiss = glitch_state(t)
    if k > 0 and rng.random() < 0.2:
        k = 0.0  # flicker: some frames inside a burst come through clean
    t_eff = t
    if hiss and k > 0:
        t_eff = t - float(rng.choice([0, 0, 1, 2, 4])) / FPS  # stutter

    img = panel_at(G["tl"].wrap(t_eff))
    if hiss and k > 0 and rng.random() < 0.6:
        ghost = panel_at(G["tl"].wrap(t_eff - 0.1))
        M = np.float32([[1, 0, rng.integers(-14, 15)], [0, 1, rng.integers(-6, 7)]])
        ghost = cv2.warpAffine(ghost, M, (S, S), borderMode=cv2.BORDER_REFLECT)
        img = img * 0.72 + hiss_map(ghost) * 0.28
    if k > 0:
        img = apply_glitch(img, k, hiss, rng)

    x0, y0, x1, y1 = G["roi"]
    out = G["photo"].astype(np.float32)
    delta = (to_photo(img) - G["base_warp"]) * 255.0
    out[y0:y1, x0:x1] += G["keep"] * delta
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("photo")
    ap.add_argument("output")
    ap.add_argument("--mask", help="person matte (white = person); computed with rembg if omitted")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--frames", help="only dump these frame numbers as PNGs, e.g. 30,190")
    ap.add_argument("--music", help="song to sync to; its clip is muxed into the output")
    ap.add_argument("--start", type=float, default=0.0, help="clip start in the song (s)")
    ap.add_argument("--end", type=float, help="clip end in the song (s)")
    ap.add_argument("--portal", action="store_true",
                    help="with --music: open the medallion into a trailer-style kaleidoscope")
    args = ap.parse_args()

    if args.music:
        from music_sync import MusicTimeline
        G["tl"] = MusicTimeline(args.music, args.start, args.end, portal=args.portal)
    else:
        G["tl"] = LoopTimeline()
    tl = G["tl"]
    nframes = int(round(tl.duration * FPS))
    load_assets(args.photo, args.mask)
    h, w = G["photo"].shape[:2]

    if args.frames:
        for n in map(int, args.frames.split(",")):
            cv2.imwrite(f"{os.path.splitext(args.output)[0]}_{n:03d}.png", render_frame(n))
        return

    audio = []
    if args.music:
        d = tl.duration
        audio = ["-ss", f"{tl.audio_start:.3f}", "-t", f"{d:.3f}", "-i", args.music,
                 "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k",
                 "-af", f"afade=t=in:d=0.25,afade=t=out:st={d - 1.1:.3f}:d=1.1", "-shortest"]
        print(f"song clip {tl.audio_start:.3f}s + {d:.3f}s, drop at {tl.beats[tl.drop]:.3f}s")
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", *audio,
        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", args.output,
    ], stdin=subprocess.PIPE)
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, frame in enumerate(pool.imap(render_frame, range(nframes), chunksize=2)):
            ff.stdin.write(frame.tobytes())
            if i % 30 == 0:
                print(f"frame {i}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()
