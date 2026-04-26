# multilingual-tts-eval 🗣️

A lightweight evaluation framework for comparing Text-to-Speech (TTS) systems
across Arabic and English, using a MOS-style (Mean Opinion Score) rubric and
automated visual reporting.

## Overview

This project benchmarks 2–3 TTS engines side-by-side on the same text samples
in both Arabic and English. Each output is scored on clarity and naturalness,
results are aggregated per language and system, and findings are visualized for
clear comparison.

## TTS Systems Compared

| System              | Type                          |
|---------------------|-------------------------------|
| gTTS                | Google Text-to-Speech (cloud) |
| Coqui TTS           | Open-source neural TTS        |
| SpeechT5 (HuggingFace) | Transformer-based TTS      |

## Features

- Generates audio outputs from identical text prompts across all TTS systems
- Supports Arabic (`ar`) and English (`en`) natively
- MOS-style scoring script (manual rubric + optional automated proxy metrics)
- Pandas-powered aggregation of scores per system/language
- Bar charts and radar plots for intuitive comparison

## Tech Stack

- Python 3.10+
- `gTTS`, `TTS` (Coqui), `transformers` (SpeechT5)
- `pandas` — scoring tables and aggregation
- `matplotlib` / `seaborn` — visualization
- `soundfile`, `librosa` — audio handling

## Project Structure

```
multilingual-tts-eval/
├── samples/
│   ├── en_samples.txt      # English text prompts
│   └── ar_samples.txt      # Arabic text prompts
├── outputs/
│   ├── audio/              # Generated TTS audio files
│   └── plots/              # Comparison charts
├── scores/
│   └── results.csv         # Aggregated MOS-style scores
├── src/
│   ├── generate.py         # TTS generation across systems
│   ├── score.py            # Scoring rubric and aggregation
│   └── visualize.py        # Chart generation
├── main.py
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone https://github.com/your-username/multilingual-tts-eval.git
cd multilingual-tts-eval
pip install -r requirements.txt

python main.py --lang ar en --systems gtts coqui speecht5
```

## Scoring Rubric

Each audio output is scored 1–5 on:

| Dimension           | Description                                              |
|---------------------|----------------------------------------------------------|
| **Clarity**         | Is the speech intelligible and artifact-free?            |
| **Naturalness**     | Does it sound like a human speaker?                      |
| **Arabic Accuracy** | Are Arabic phonemes and stress correct? *(ar only)*      |

## Sample Results

| System    | Language | Clarity | Naturalness | Avg  |
|-----------|----------|---------|-------------|------|
| Coqui TTS | Arabic   | 4.1     | 3.8         | 3.95 |
| gTTS      | Arabic   | 3.2     | 2.9         | 3.05 |
| SpeechT5  | English  | 4.5     | 4.3         | 4.40 |

## License

MIT
