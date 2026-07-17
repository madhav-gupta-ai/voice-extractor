"""Vocal / accompaniment separation with Demucs (two-stems mode)."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def separate_file(input_path, output_dir):
    """Split ``input_path`` into vocals.wav and accompaniment.wav in ``output_dir``."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-o", tmp, str(input_path)],
            check=True,
        )
        stems = next(Path(tmp).glob(f"*/{input_path.stem}"))
        vocals = output_dir / "vocals.wav"
        accompaniment = output_dir / "accompaniment.wav"
        shutil.move(stems / "vocals.wav", vocals)
        shutil.move(stems / "no_vocals.wav", accompaniment)
    return vocals, accompaniment
