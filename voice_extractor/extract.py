"""Extract clean speech from a video or audio file.

Usage:
    voice-extractor movie_clip.mp4
    voice-extractor clip.mp4 --denoise -o results/
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .denoise import denoise_file
from .models import DEFAULT_MODEL
from .separate import separate_file


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def to_wav(input_path, wav_path):
    result = subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(input_path), "-vn", "-acodec", "pcm_s16le",
         "-ar", "44100", "-ac", "2", str(wav_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"error: could not read audio from {input_path}\n{result.stderr.strip().splitlines()[-1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract vocals and accompaniment from a video or audio file.")
    parser.add_argument("input", help="path to a video or audio file (mp4, mp3, wav, ...)")
    parser.add_argument("--denoise", action="store_true",
                        help="remove environmental noise before separating; useful for "
                             "poorly recorded audio, may reduce quality on clean recordings")
    parser.add_argument("-o", "--output-dir", default=".",
                        help="directory for vocals.wav and accompaniment.wav (default: current)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="noise-classifier checkpoint used with --denoise")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"error: input file not found: {input_path}")
    if args.denoise and not Path(args.model).is_file():
        sys.exit(f"error: model not found: {args.model}\nTrain one with: python -m voice_extractor.train")

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio.wav"
        to_wav(input_path, audio)
        if args.denoise:
            audio = denoise_file(audio, Path(tmp) / "denoised.wav", args.model)
        vocals, accompaniment = separate_file(audio, args.output_dir)
    print(f"written: {vocals}\nwritten: {accompaniment}")


if __name__ == "__main__":
    main()
