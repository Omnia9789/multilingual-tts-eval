"""
score.py – MOS-style rubric scoring + pandas aggregation.
"""
import csv, itertools
from pathlib import Path

import pandas as pd

SCORES_DIR = Path("scores")
SCORES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = SCORES_DIR / "results.csv"

DIMENSIONS = ["clarity", "naturalness", "arabic_accuracy"]  # arabic_accuracy: ar only


def prompt_score(system: str, lang: str, idx: int) -> dict:
    """Interactive scorer for one audio file (run in terminal after listening)."""
    print(f"\n  ▶ {system} / {lang} / sample {idx:02d}  "
          f"(outputs/audio/{system}_{lang}_{idx:02d}.wav)")
    row = {"system": system, "lang": lang, "sample": idx}
    for dim in DIMENSIONS:
        if dim == "arabic_accuracy" and lang != "ar":
            row[dim] = None
            continue
        while True:
            raw = input(f"    {dim} [1-5]: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= 5:
                row[dim] = int(raw)
                break
            print("    Please enter a number between 1 and 5.")
    return row


def load_or_create_results() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    cols = ["system", "lang", "sample"] + DIMENSIONS
    return pd.DataFrame(columns=cols)


def save_results(df: pd.DataFrame) -> None:
    df.to_csv(RESULTS_CSV, index=False)
    print(f"  [✓] Results saved → {RESULTS_CSV}")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-(system, lang) mean scores."""
    num = df[DIMENSIONS].apply(pd.to_numeric, errors="coerce")
    df[DIMENSIONS] = num
    grp = df.groupby(["system", "lang"])[DIMENSIONS].mean().round(2)
    grp["avg"] = grp[["clarity", "naturalness"]].mean(axis=1).round(2)
    return grp.reset_index()


def print_table(agg: pd.DataFrame) -> None:
    print("\n  System        Lang   Clarity  Naturalness  Avg")
    print("  " + "─" * 50)
    for _, r in agg.iterrows():
        ar_acc = f"{r['arabic_accuracy']:.1f}" if pd.notna(r.get("arabic_accuracy")) else "  —  "
        print(f"  {r['system']:<13} {r['lang']:<6} {r['clarity']:<8.1f} "
              f"{r['naturalness']:<12.1f} {r['avg']:.2f}")


def run_interactive_scoring(systems: list[str], langs: list[str], n_samples: int = 10) -> None:
    df = load_or_create_results()
    for system, lang in itertools.product(systems, langs):
        for idx in range(1, n_samples + 1):
            exists = ((df["system"] == system) & (df["lang"] == lang) &
                      (df["sample"] == idx)).any()
            if exists:
                continue
            row = prompt_score(system, lang, idx)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_results(df)
    agg = aggregate(df)
    print_table(agg)
    return agg


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=["gtts", "coqui", "speecht5"])
    p.add_argument("--lang", nargs="+", default=["en", "ar"])
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    run_interactive_scoring(args.systems, args.lang, args.n)
