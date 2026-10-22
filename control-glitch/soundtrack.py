#!/usr/bin/env python3
"""Synthesize a seamless-looping soundtrack synced to glitch_portrait's timeline.

Low drone, stone-grind while pieces turn, clunks when they lock, a resonant
boom on each pulse and digital crackle on every glitch burst. Everything is
placed and filtered circularly over one loop, so the audio wraps cleanly.

Usage:
    python soundtrack.py out.wav
"""
import math
import sys
import wave

import numpy as np

import glitch_portrait as gp

SR = 48000
N = int(SR * gp.LOOP)
rng = np.random.default_rng(7)
mix = np.zeros((2, N))


def add(sig, start, pan=0.0, gain=1.0):
    """Mix a mono signal in at `start` seconds, wrapping past the loop end."""
    i0 = int(round(start * SR)) % N
    sig = sig[:N] * gain
    lr = np.array([math.sqrt(0.5 * (1 - pan)), math.sqrt(0.5 * (1 + pan))])[:, None]
    first = min(len(sig), N - i0)
    mix[:, i0:i0 + first] += lr * sig[:first]
    if first < len(sig):
        mix[:, :len(sig) - first] += lr * sig[first:]


def band(x, lo, hi):
    """Circular FFT band-pass with soft edges."""
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = 1 / (1 + (lo / np.maximum(f, 1e-3)) ** 4) / (1 + (f / hi) ** 4)
    return np.fft.irfft(np.fft.rfft(x) * g, len(x))


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def thunk(freq, dur=0.9, metal=0.25):
    t = tt(dur)
    body = np.sin(2 * np.pi * freq * t * (1 + 0.6 * np.exp(-t / 0.03))) * np.exp(-t / 0.20)
    click = band(rng.standard_normal(len(t)), 1500, 7000) * np.exp(-t / 0.006)
    ring = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / d)
               for f, a, d in [(412, 1.0, 0.35), (1131, 0.5, 0.25), (2207, 0.25, 0.16)])
    return body + 0.35 * click / np.abs(click).max() + metal * ring


def tick(freq):
    t = tt(0.35)
    click = band(rng.standard_normal(len(t)), 2500, 9000) * np.exp(-t / 0.004)
    ping = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.09)
    return 0.5 * click / np.abs(click).max() + 0.3 * ping


def boom():
    t = tt(3.0)
    sub = np.sin(2 * np.pi * (40 * t + 50 * 0.12 * (1 - np.exp(-t / 0.12)))) * np.exp(-t / 0.9)
    chord = sum(np.sin(2 * np.pi * f * t) + np.sin(2 * np.pi * (f + 0.45) * t)
                for f in (110.0, 165.0, 220.0, 330.0)) / 8
    chord *= np.minimum(1, t / 0.04) * np.exp(-t / 1.4)
    return sub + 0.6 * chord


def swell(dur):
    t = tt(dur)
    x = band(rng.standard_normal(len(t)), 300, 5000) * np.exp(-t / (dur / 3))
    return (x / np.abs(x).max())[::-1]


def crackle(dur, k, hiss):
    n = int(dur * SR)
    out = np.zeros(n)
    i = 0
    while i < n:  # sample-and-hold digital noise
        L = int(rng.integers(20, 600))
        out[i:i + L] = rng.uniform(-1, 1) * (rng.random() < 0.8)
        i += L
    t = np.arange(n) / SR
    buzz = np.sign(np.sin(2 * np.pi * rng.uniform(60, 180) * t))
    gate = np.repeat(rng.random(n // 1440 + 1) < 0.6, 1440)[:n]
    sig = 0.5 * out + 0.35 * buzz * gate
    if hiss:
        air = band(rng.standard_normal(n), 3000, 9000)
        air *= (0.6 + 0.4 * np.sin(2 * np.pi * 13 * t)) / np.abs(air).max()
        growl = ((55 * t) % 1 - 0.5) + ((55.7 * t) % 1 - 0.5)
        sig += 0.6 * air + 0.4 * growl
    fade = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / 120)
    return sig * fade * k


def grind(events, count, lo, hi, gain):
    """Stone-on-stone noise following each element's angular speed."""
    frames = np.arange(int(gp.LOOP * gp.FPS) + 1) / gp.FPS
    speed = np.zeros_like(frames)
    for i in range(count):
        a = np.array([gp.angle_at(events, i, t) for t in frames])
        speed += np.abs(np.gradient(a)) * gp.FPS
    env = np.interp(np.arange(N) / SR, frames, speed / max(speed.max(), 1e-6))
    return band(rng.standard_normal(N), lo, hi) * env * gain


def main(path):
    t = np.arange(N) / SR

    # Drone: frequencies are multiples of 1/LOOP Hz so it wraps seamlessly.
    drone = (0.10 * np.sin(2 * np.pi * 55 * t) + 0.05 * np.sin(2 * np.pi * 110 * t)
             + 0.04 * np.sin(2 * np.pi * 82.5 * t)) * (1 + 0.3 * np.sin(2 * np.pi * t / 6))
    air = band(rng.standard_normal(N), 40, 400)
    mix[:] += drone + 0.25 * air / np.abs(air).max()

    mix[:] += grind(gp.RING_EVENTS, len(gp.RINGS), 120, 700, 0.12)
    mix[:] += grind(gp.ROSETTE_EVENTS, len(gp.ROSETTES), 500, 2500, 0.05)

    ring_freq = [95, 70, 52]
    for ring, t0, d, _ in gp.RING_EVENTS:
        add(thunk(ring_freq[ring] * 1.3, 0.5, 0.15), t0, gain=0.25)
        add(thunk(ring_freq[ring]), t0 + d, gain=0.6)

    for i, t0, d, _ in gp.ROSETTE_EVENTS:
        pan = (gp.ROSETTES[i][0] - gp.C) / gp.C * 0.8
        add(tick(rng.uniform(1700, 2600)), t0 + d, pan=pan, gain=0.25)

    for tp, k in gp.PULSES:
        add(boom(), tp, gain=0.7 * k)
    add(swell(1.2), gp.PULSES[1][0] - 1.2, gain=0.35)

    for t0, d, k, hiss in gp.GLITCHES:
        add(crackle(d, k, hiss), t0, pan=rng.uniform(-0.4, 0.4), gain=0.35)

    # Circular stereo reverb.
    ir_t = tt(1.8)
    for c in range(2):
        ir = rng.standard_normal(len(ir_t)) * np.exp(-ir_t / 0.45)
        ir = np.pad(ir / np.sqrt((ir ** 2).sum()), (0, N - len(ir)))
        mix[c] += 0.35 * np.fft.irfft(np.fft.rfft(mix[c]) * np.fft.rfft(ir), N)

    out = np.tanh(1.3 * mix)
    out *= 0.89 / np.abs(out).max()
    pcm = (out.T * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main(sys.argv[1])
