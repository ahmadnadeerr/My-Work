# Control-style glitch portrait

Animates the mosaic panel behind the subject so the pattern "shifts" like the
Oldest House in *Control*, while the person stays pixel-for-pixel untouched.

- Medallion rings counter-rotate and lock into place on 22.5° steps.
- Star rosettes click round in 30° steps, triggered by resonance ripples.
- Hiss-style glitch bursts: RGB split, tears, macroblocks, red tint, stutter.
- 12 s seamless loop. Every motion nets to zero, so it loops cleanly.

```sh
pip install opencv-python-headless numpy "rembg[cpu]"   # rembg only if no --mask
python glitch_portrait.py photo.jpg video.mp4 [--mask person_mask.png]
python soundtrack.py sound.wav
ffmpeg -i video.mp4 -i sound.wav -c:v copy -c:a aac -b:a 192k final.mp4
```

The panel corners, medallion centre and rosette positions in
`glitch_portrait.py` were measured on the original 1890x2576 photo. Another
photo needs those constants re-measured.

Source photos and rendered videos are git-ignored and not committed.
