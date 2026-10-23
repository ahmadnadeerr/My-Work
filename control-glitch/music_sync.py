"""Drive the glitch animation from a song instead of the built-in loop.

Beats set the mechanical ticks, loudness sets how far each tick turns, the
biggest jump in loudness becomes "the drop" (a heavy Hiss burst with ripples
rushing in and out of the medallion), every 8 beats after it starts a new
phrase with its own ripple and glitch, and the hardest accents get short
glitch hits.
"""
import math
import subprocess

import numpy as np

import glitch_portrait as gp

SR = 22050
HOP = 512
DEG_PER_BEAT = 15.0
LEAD = 1 / 30  # land visual hits a frame early so they meet the attack, not trail it


def decode(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def ease_out_cubic(x):
    return 1 - (1 - x) ** 3


class MusicTimeline:
    def __init__(self, audio_path, start, end):
        import librosa

        y = decode(audio_path)
        onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP)
        beats = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
        rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
        frame_t = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)

        def beat_stat(arr, b, reduce):
            m = (frame_t >= b - 0.05) & (frame_t < b + 0.25)
            return reduce(arr[m]) if m.any() else 0.0

        db = np.array([20 * math.log10(beat_stat(rms, b, np.mean) + 1e-9) for b in beats])
        hits = np.array([beat_stat(onset, b, np.max) for b in beats])

        # Snap the clip to beats; beat index 0 is the clip's first frame.
        k0 = int(np.argmin(np.abs(beats - start)))
        k1 = int(np.argmin(np.abs(beats - end)))
        self.audio_start = float(beats[k0])
        self.duration = float(beats[k1] - beats[k0])
        self.ibi = float(np.median(np.diff(beats)))
        self.beats = beats - beats[k0] - LEAD
        self.idx = np.arange(len(beats)) - k0

        energy = np.clip((db + 22) / 12, 0, 1)
        jump = np.array([db[k] - db[max(0, k - 4):k].max() if k else 0 for k in range(len(db))])
        in_clip = (self.idx > 0) & (self.idx < k1 - k0)
        self.drop = int(np.argmax(np.where(in_clip, jump, -1e9)))
        loud = energy > 0.6

        accent = np.ones(len(beats))
        accent[self.drop] = 3.0
        phrases = [k for k in range(self.drop, k1 + 1, 8)]
        accent[phrases[1:]] = 1.6
        self.step = DEG_PER_BEAT * (0.3 + 0.7 * energy) * accent
        self.step[:k0] = 0
        self.mix = 0.45 + 0.4 * energy
        self.cum = np.concatenate([[0.0], np.cumsum(self.step)])
        self.energy = energy

        t = self.beats
        drop_t = t[self.drop]
        self.waves = [(t[k], 0.0, 950.0, 3.0) for k in range(k0, self.drop - 4, 8)]
        self.waves += [(drop_t - 1150 / 950, 1150.0, -950.0, 9.0), (drop_t, 0.0, 1150.0, 11.0)]
        for k in phrases[1:]:
            self.waves.append((t[k], 0.0, 950.0, 6.0))
        for k in range(self.drop + 4, k1 + 1, 8):
            self.waves.append((t[k] - 1150 / 950, 1150.0, -950.0, 5.0))

        self.pulses = [(drop_t, 1.4)] + [(t[k], 0.8) for k in phrases[1:]]
        self.pulses += [(t[k], 0.4) for k in range(self.drop + 4, k1 + 1, 8)]

        g = [(t[self.drop - 4 + i] - 0.02, d, k, False)
             for i, (d, k) in enumerate([(0.08, 0.25), (0.10, 0.35), (0.12, 0.5), (0.2, 0.65)])]
        g.append((drop_t - 0.02, 0.70, 1.0, True))
        for k in phrases[1:]:
            big = (k - self.drop) % 16 == 0
            g.append((t[k] - 0.02, 0.32 if big else 0.2, 0.85 if big else 0.6, big))
        taken = {self.drop - 4 + i for i in range(5)} | set(phrases)
        loud_hits = hits[k0:k1][loud[k0:k1]]
        cut = np.percentile(loud_hits, 75) if len(loud_hits) else np.inf
        for k in range(k0 + 1, k1):
            near = any(abs(k - j) <= 1 for j in taken)
            if not near and ((loud[k] and hits[k] >= cut) or (not loud[k] and hits[k] >= cut * 0.9)):
                g.append((t[k] - 0.02, 0.07, 0.4 if loud[k] else 0.3, False))
                taken.add(k)
        self.glitches = sorted(g)

    # -- interface used by glitch_portrait.panel_at / render_frame --

    def lag(self, t, t0):
        return t - t0

    def wrap(self, t):
        return max(0.0, t)

    def position(self, t):
        """Musical time in beats since the clip start (linear past the ends)."""
        b, i = self.beats, self.idx
        if t < b[0]:
            return i[0] + (t - b[0]) / self.ibi
        if t > b[-1]:
            return i[-1] + (t - b[-1]) / self.ibi
        return float(np.interp(t, b, i))

    def turned(self, p):
        """Degrees turned by a unit piece at musical position p; each beat kicks it on."""
        if p <= 0:
            return 0.0
        k = int(math.floor(p))
        f = p - k
        j = k - self.idx[0]
        if j >= len(self.step):
            return self.cum[-1] + (p - len(self.step) - self.idx[0]) * self.step[-1]
        m = self.mix[j]
        return self.cum[j] + self.step[j] * ((1 - m) * f + m * ease_out_cubic(f))

    def pieces(self, stars):
        ripple = lambda x, y: max(0.0, gp.ripple_delay(x, y)) / self.ibi
        p = [(gp.C, gp.C, r0, r1, (turns, 0.25 * i))
             for i, (r0, r1, turns, _) in enumerate(gp.RINGS)]
        p += [(x, y, 0, r, (1 if x < gp.C else -1, ripple(x, y))) for x, y, r in gp.ROSETTES]
        p += [(x, y, 0, r, (-2 if x < gp.C else 2, ripple(x, y) + 0.5)) for x, y, r in stars]
        return p

    def angle(self, t, spec):
        mult, delay = spec
        return mult * self.turned(self.position(t) - delay)

    def beat_glow(self, t):
        p = self.position(t)
        k = int(math.floor(p)) - self.idx[0]
        if not 0 <= k < len(self.beats):
            return 0.0
        since = t - self.beats[k]
        return 0.4 * self.energy[k] ** 2 * math.exp(-since / 0.2)
