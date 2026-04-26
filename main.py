"""
main.py – orchestrator for the multilingual TTS evaluation pipeline.

Usage:
    python main.py --lang ar en --systems gtts coqui speecht5
    python main.py --lang en   --systems gtts speecht5 --skip-score
    python main.py --plots-only
"""
import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Multilingual TTS Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lang", nargs="+", default=["en", "ar"],
                   choices=["en", "ar"], metavar="LANG",
                   help="Languages to evaluate (default: en ar)")
    p.add_argument("--systems", nargs="+", default=["gtts", "coqui", "speecht5"],
                   choices=["gtts", "coqui", "speecht5"], metavar="SYS",
                   help="TTS systems to benchmark (default: all)")
    p.add_argument("--n-samples", type=int, default=10,
                   help="Number of text samples per language (default: 10)")
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip audio generation (use cached outputs)")
    p.add_argument("--skip-score", action="store_true",
                   help="Skip interactive scoring (use existing results.csv)")
    p.add_argument("--plots-only", action="store_true",
                   help="Only regenerate plots from existing results.csv")
    args = p.parse_args()

    if args.plots_only:
        from src.visualize import generate_all_plots
        generate_all_plots()
        return

    # 1. Generate audio
    if not args.skip_generate:
        print("\n[STEP 1/3] Generating audio…")
        from src.generate import generate_all
        generate_all(args.lang, args.systems)
    else:
        print("\n[STEP 1/3] Skipping audio generation.")

    # 2. Score
    if not args.skip_score:
        print("\n[STEP 2/3] Scoring (MOS rubric)…")
        from src.score import run_interactive_scoring
        agg = run_interactive_scoring(args.systems, args.lang, args.n_samples)
    else:
        print("\n[STEP 2/3] Skipping scoring.")
        import pandas as pd
        from src.score import aggregate, print_table
        df = pd.read_csv("scores/results.csv")
        agg = aggregate(df)
        print_table(agg)

    # 3. Visualize
    print("\n[STEP 3/3] Generating plots…")
    from src.visualize import plot_bar, plot_radar
    plot_bar(agg)
    plot_radar(agg)
    print("\n✅  Evaluation complete.")


if __name__ == "__main__":
    main()
