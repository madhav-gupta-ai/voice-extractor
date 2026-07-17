# Data

## features/

Pre-extracted audio features for all 8,732 clips of the
[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset,
split into 7,895 training and 837 test samples:

- `train_data.csv` / `test_data.csv` — one row per clip: a 40x5 feature matrix
  flattened to 200 values. The five columns are 40-coefficient vectors of MFCC,
  mel spectrogram, chroma-STFT, chroma-CQT and chroma-CENS (each averaged over
  time with librosa).
- `train_labels.csv` / `test_labels.csv` — the UrbanSound8K class ID (0-9) per clip.
- `UrbanSound8K.csv` — the dataset's original metadata table.

These files are committed so the classifier can be retrained without
downloading the ~6 GB dataset; regenerating them means downloading
UrbanSound8K and re-running the same librosa feature extraction as in
`voice_extractor/denoise.py`.

## Noise profiles

The reference noise recordings used for spectral gating (two per noise class;
none exist for `gun_shot`) ship inside the package at
`voice_extractor/assets/noise/`. They originate from UrbanSound8K via the
[Denoiser](https://github.com/immohann/Denoiser) project.

## License

UrbanSound8K is distributed under the Creative Commons
Attribution-NonCommercial 3.0 license, for non-commercial use only. If you use
this data in academic work, the dataset authors request citing:

> J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban
> Sound Research", 22nd ACM International Conference on Multimedia, Orlando,
> USA, Nov. 2014.
