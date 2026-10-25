"""Kaleidoscope portal inspired by the end of the CONTROL Resonant trailer.

The mosaic's own glass is folded into a mirrored N-fold kaleidoscope and
streamed through a log-polar tunnel, in the trailer's three looks: a cold
blue tunnel with a dark core, a white mandala on black, and a Hiss-red
eclipse with liquid smearing. The portal opens from the medallion centre as
an iris, so outside it the real mosaic keeps turning.
"""
import math

import cv2
import numpy as np

import glitch_portrait as gp

BAND = (math.log(60.0), math.log(760.0))  # radial band of the panel the tunnel loops through
EDGE = {"blue": np.float32([1.0, 0.85, 0.6]), "white": np.float32([1.0, 1.0, 1.0]),
        "red": np.float32([0.1, 0.15, 1.0])}


def gradient(lum, stops):
    """Map luminance through (position, BGR colour) stops."""
    pos = [p for p, _ in stops]
    return np.stack([np.interp(lum, pos, [c[i] for _, c in stops]) for i in range(3)],
                    -1).astype(np.float32)


def bloom(img, sigma, k):
    return img + k * cv2.GaussianBlur(np.clip(img - 0.55, 0, None), (0, 0), sigma)


def smooth_noise(rng, sigma):
    n = cv2.GaussianBlur(rng.standard_normal((gp.S, gp.S)).astype(np.float32), (0, 0), sigma)
    return n / np.abs(n).max()


class Portal:
    def __init__(self, base, grid):
        self.g = grid  # the renderer's shared arrays (xx, yy, rad)
        lum = np.clip(base @ np.float32([0.114, 0.587, 0.299]), 0, 1)
        blue = gradient(lum, [(0, (0.07, 0.02, 0.01)), (0.3, (0.45, 0.15, 0.05)),
                              (0.6, (0.9, 0.55, 0.25)), (0.85, (1.0, 0.85, 0.7)),
                              (1.0, (1.0, 1.0, 1.0))])
        white = np.clip((lum - 0.5) * 2.8, 0, 1) ** 1.3
        self.src = {
            "blue": bloom(0.75 * blue + 0.25 * base, 12, 0.8),
            "white": bloom(white[..., None] * np.float32([0.92, 0.9, 1.0]), 10, 1.2),
            "red": bloom(gradient(lum, [(0, (0, 0, 0)), (0.35, (0.01, 0.01, 0.22)),
                                        (0.7, (0.03, 0.03, 0.6)), (1.0, (0.35, 0.38, 0.85))]), 12, 0.6),
        }
        rng = np.random.default_rng(3)
        self.n1, self.n2 = smooth_noise(rng, 70), smooth_noise(rng, 70)
        # The trailer's red frames smear hardest on one side.
        self.side = (0.4 + 0.6 * np.clip((grid["xx"] - 300) / 1000, 0, 1)).astype(np.float32)

    def kaleidoscope(self, s, t):
        x, y = self.g["xx"], self.g["yy"]
        if s.get("liquid"):
            w = 2 * math.pi * 0.35 * t
            a = s["liquid"] * self.side
            x = x + a * (self.n1 * math.cos(w) + self.n2 * math.sin(w))
            y = y + a * (self.n2 * math.cos(w) - self.n1 * math.sin(w))
        r = np.maximum(np.hypot(x - gp.C, y - gp.C), 1.0)
        wedge = 2 * math.pi / s["sectors"]
        phi = (np.arctan2(y - gp.C, x - gp.C) - s["spin"]) % wedge
        th = np.minimum(phi, wedge - phi) + s["slice"]  # mirror each wedge
        lr = np.log(r) - s["zoom"]
        rs = np.exp(BAND[0] + (lr - BAND[0]) % (BAND[1] - BAND[0]))
        mx = (gp.C + rs * np.cos(th)).astype(np.float32)
        my = (gp.C + rs * np.sin(th)).astype(np.float32)
        sample = lambda look: cv2.remap(self.src[look], mx, my, cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)
        img = sample(s["look"])
        if s.get("flood"):  # red seeping in from the outside
            m = np.clip((r - s["flood"]) / 160 + 0.5, 0, 1)[..., None]
            img = img * (1 - m) + sample("red") * m
        return img, r

    def blend(self, panel, s, t):
        img, r = self.kaleidoscope(s, t)
        rad = self.g["rad"]
        if s.get("void"):  # dark core with a thin lit rim
            v = s["void"]
            img *= np.clip((r - v) / 12, 0.04, 1)[..., None]
            img += np.exp(-((r - v - 6) / 5) ** 2)[..., None] * EDGE[s["look"]] * 0.6
        if s.get("eclipse"):
            e = s["eclipse"]
            img *= (0.16 + 0.84 * np.clip((r - e) / 30, 0, 1))[..., None]
            img += np.exp(-((r - e - 22) / 14) ** 2)[..., None] * EDGE["red"] * 0.5
        img *= 1 + s.get("flash", 0.0)

        iris = s["iris"]
        m = np.clip((iris - rad) / 6 + 0.5, 0, 1)[..., None]
        out = panel * (1 - m) + img * m
        if iris < 1100:
            out += np.exp(-((rad - iris) / 9) ** 2)[..., None] * EDGE[s["look"]] * 0.9
        return out
