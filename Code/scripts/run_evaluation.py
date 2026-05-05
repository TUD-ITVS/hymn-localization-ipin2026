"""Entry point for the HYMN positioning evaluation.

Usage (from Code/):
    python scripts/run_evaluation.py            # full run
    python scripts/run_evaluation.py --quick    # UWB + ILS only, no ResNet (smoke test)
    python scripts/run_evaluation.py --no-resnet
or
    python -m scripts.run_evaluation [args]
"""
import argparse
import os
import sys

# Ensure Code/ is on sys.path so `import hymn` works regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from hymn.config import EVAL, OUT_DIR
from hymn.evaluation import data_interface as di
from hymn.evaluation import resnet_ingest, runner, stats
from hymn.plotting import figures as plotting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="UWB + ILS only, no ResNet")
    ap.add_argument("--no-resnet", action="store_true", help="skip ResNet ingestion")
    ap.add_argument("--no-plots", action="store_true", help="skip plot generation")
    ap.add_argument("--skip-classical", action="store_true",
                    help="reuse existing results_long.csv, re-ingest ResNet + re-plot")
    args = ap.parse_args()

    cfg = EVAL

    if args.skip_classical:
        out_path = os.path.join(OUT_DIR, "results_long.csv")
        long_df = pd.read_csv(out_path)
        classical = long_df[long_df["method"].isin(cfg["methods"])].copy()
        if not args.no_resnet:
            resnet_df = resnet_ingest.ingest_all(cfg)
            if resnet_df is not None:
                long_df = pd.concat([classical, resnet_df], ignore_index=True)
            else:
                long_df = classical
        else:
            long_df = classical
        long_df.to_csv(out_path, index=False)
        print(f"[run] rewrote {len(long_df)} rows -> {out_path}")
    elif args.quick:
        long_df = runner.run_all(cfg=cfg, techs=["uwb"], methods=["ILS"], include_resnet=False)
    else:
        long_df = runner.run_all(cfg=cfg, include_resnet=not args.no_resnet)

    stats_techs = ["uwb"] if args.quick else None
    stats_df = stats.write_stats(long_df, techs=stats_techs)
    print("\n=== stats_table.csv ===")
    print(stats_df.to_string(index=False))

    if not args.quick:
        rng_df = di.compute_ranging_long(techs=("ble", "uwb", "wifi"))
        rng_path = os.path.join(OUT_DIR, "ranging_long.csv")
        rng_df.to_csv(rng_path, index=False)
        rng_stats = stats.write_ranging_stats(rng_df)
        print("\n=== ranging_stats.csv ===")
        print(rng_stats.to_string(index=False))
    else:
        rng_df = None

    if not args.no_plots:
        paths = plotting.plot_all(long_df, cfg)
        if rng_df is not None:
            paths.append(plotting.plot_ecdf_ranging(rng_df, cfg=cfg))
        print("\n=== figures ===")
        for p in paths:
            print(" ", p)


if __name__ == "__main__":
    main()
