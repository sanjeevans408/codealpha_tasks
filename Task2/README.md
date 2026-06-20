# Emotion Recognition — Task2

This project generates synthetic emotional speech signals, extracts audio features, trains neural models using TensorFlow/Keras, and saves a visual performance report.

## Files
- `Emotion_recognition.py` — main script
- `requirements.txt` — Python dependencies

## Prerequisites
- Python 3.11 or 3.12 is required for compatibility with TensorFlow 2.21.
- Python 3.13 is not supported for this project because TensorFlow 2.x does not offer stable prebuilt binaries for it.
- A virtual environment is strongly recommended.

## Setup (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```powershell
.\.venv\Scripts\python Emotion_recognition.py
```

## Troubleshooting
- If TensorFlow fails to load with a DLL or AVX-related error, your CPU may not support the installed TensorFlow binary or the required Microsoft Visual C++ runtime may be missing.
- For Windows, install the latest Visual C++ 2015-2022 Redistributable from Microsoft.
- If the error persists, uninstall `tensorflow` and install the CPU-only package:
  ```powershell
  pip uninstall tensorflow tensorflow-cpu
  pip install "tensorflow-cpu>=2.12,<2.22"
  ```
- If you still see native runtime initialization failures, create the venv with Python 3.11 or 3.12 instead of Python 3.13.
  ```powershell
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

## Notes
- The script depends on TensorFlow/Keras and librosa for audio feature extraction and model training.
- `requirements.txt` now recommends the CPU-only TensorFlow package to improve compatibility on Windows.
