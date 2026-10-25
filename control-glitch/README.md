# Control-style glitch portrait

Animates the mosaic panel behind the subject so the pattern "shifts" like the
Oldest House in *Control*, while the person stays pixel-for-pixel untouched.

The pattern never stops moving:

- Medallion rings turn continuously in alternating directions, lurching on a
  one-second mechanical tick.
- Star rosettes and the small blue stars spin like counter-rotating gears,
  mirrored left/right, with their ticks rippling out from the centre.
- Resonance ripples run out of and back into the medallion.
- Hiss-style glitch bursts: RGB split, tears, macroblocks, red tint, stutter.
- 16 s seamless loop. Every piece turns a whole number of times per loop.

```sh
pip install opencv-python-headless numpy "rembg[cpu]"   # rembg only if no --mask
python glitch_portrait.py photo.jpg video.mp4 [--mask person_mask.png]
python soundtrack.py sound.wav
ffmpeg -i video.mp4 -i sound.wav -c:v copy -c:a aac -b:a 192k final.mp4
```

To sync to a song instead of the built-in loop, pass the track and the part
to use (snapped to beats). Beats drive the ticks, loudness drives how far each
tick turns, the biggest jump in loudness becomes the drop (heavy Hiss burst and
ripples), and every 8 beats after it starts a new phrase. The song clip is
muxed into the output (needs `pip install librosa`):

```sh
python glitch_portrait.py photo.jpg video.mp4 --music song.mp3 --start 25.1 --end 54.8
```

Add `--portal` to also open the medallion into a kaleidoscope, like the end
of the CONTROL Resonant trailer (`portal.py`): the mosaic's own glass folded
into a mirrored 16/24-fold kaleidoscope and streamed through a log-polar
tunnel, cutting on the beat between a blue tunnel, a white mandala on black
and a Hiss-red eclipse with liquid smearing. It stays inside the panel.

The panel corners, medallion centre and rosette positions in
`glitch_portrait.py` were measured on the original 1890x2576 photo. Another
photo needs those constants re-measured.

Source photos, songs and rendered videos are git-ignored and not committed.
