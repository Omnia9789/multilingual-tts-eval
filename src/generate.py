"""
generate.py – synthesise audio for every (system, language, sample) triple.
"""
import argparse, os, time
from pathlib import Path

OUTPUT_DIR = Path("outputs/audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILES = {"en": "samples/en_samples.txt", "ar": "samples/ar_samples.txt"}


def load_samples(lang: str) -> list[str]:
    with open(SAMPLE_FILES[lang], encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


# ── gTTS ─────────────────────────────────────────────────────────────────────
def generate_gtts(text: str, lang: str, out_path: Path) -> None:
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang)
    tts.save(str(out_path))


# ── Coqui TTS ─────────────────────────────────────────────────────────────────
def generate_coqui(text: str, lang: str, out_path: Path, model_name: str | None = None) -> None:
    from TTS.api import TTS
    model = model_name or (
        "tts_models/ar/css10/vits" if lang == "ar" else "tts_models/en/ljspeech/vits"
    )
    tts = TTS(model_name=model)
    tts.tts_to_file(text=text, file_path=str(out_path))


# ── SpeechT5 (HuggingFace) ────────────────────────────────────────────────────
def generate_speecht5(text: str, lang: str, out_path: Path) -> None:
    import torch, soundfile as sf
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    from datasets import load_dataset

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write(str(out_path), speech.numpy(), samplerate=16000)


GENERATORS = {
    "gtts": generate_gtts,
    "coqui": generate_coqui,
    "speecht5": generate_speecht5,
}


def generate_all(langs: list[str], systems: list[str]) -> None:
    for lang in langs:
        samples = load_samples(lang)
        for system in systems:
            gen_fn = GENERATORS[system]
            print(f"\n  [INFO] Generating audio for {len(samples)} {lang.upper()} prompts "
                  f"with {system}…")
            for idx, text in enumerate(samples, 1):
                out = OUTPUT_DIR / f"{system}_{lang}_{idx:02d}.wav"
                if out.exists():
                    print(f"         [{idx:02d}/{len(samples)}] skip (cached)")
                    continue
                try:
                    gen_fn(text, lang, out)
                    bar = "█" * idx + "░" * (len(samples) - idx)
                    print(f"  [{bar}] {idx}/{len(samples)}", end="\r")
                except Exception as exc:
                    print(f"\n  [WARN] {system}/{lang}/{idx} failed: {exc}")
            print(f"\n  [✓] {system}/{lang} done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lang", nargs="+", default=["en", "ar"])
    p.add_argument("--systems", nargs="+", default=["gtts", "coqui", "speecht5"])
    args = p.parse_args()
    generate_all(args.lang, args.systems)
