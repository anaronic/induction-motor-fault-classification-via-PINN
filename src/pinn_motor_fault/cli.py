"""Command line interface for the Paderborn PINN pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .paderborn import DatasetError, download_archives, extract_archives
from .results import generate_results
from .train import make_synthetic_dataset, run_grouped_experiment, train_from_paderborn, train_from_windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Physics-informed motor fault classifier for Paderborn data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download official Paderborn archive files.")
    download.add_argument("--codes", nargs="+", default=["K001", "KA01", "KI01", "KB23"], help="Bearing-state archive codes.")
    download.add_argument("--raw-dir", type=Path, default=Path("data") / "paderborn" / "raw")
    download.add_argument("--extract", action="store_true", help="Extract downloaded RAR archives.")
    download.add_argument("--extract-dir", type=Path, default=Path("data") / "paderborn" / "extracted")
    download.add_argument("--overwrite", action="store_true")

    train = subparsers.add_parser("train", help="Train on extracted Paderborn MATLAB files.")
    train.add_argument("--data-dir", type=Path, default=Path("data") / "paderborn" / "extracted")
    train.add_argument("--output", type=Path, default=Path("models") / "paderborn_pinn.npz")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--window-size", type=int, default=4096)
    train.add_argument("--stride", type=int, default=2048)
    train.add_argument("--signal", choices=["current", "vibration"], default="current")
    train.add_argument("--max-windows-per-file", type=int, default=20)
    train.add_argument("--synthetic-if-empty", action="store_true")

    smoke = subparsers.add_parser("smoke", help="Run an end-to-end synthetic smoke test.")
    smoke.add_argument("--epochs", type=int, default=10)
    smoke.add_argument("--output", type=Path, default=Path("models") / "synthetic_pinn.npz")

    results = subparsers.add_parser("results", help="Evaluate a trained model and generate report artifacts.")
    results.add_argument("--data-dir", type=Path, default=Path("data") / "paderborn" / "extracted")
    results.add_argument("--model", type=Path, default=Path("models") / "paderborn_subset_pinn.npz")
    results.add_argument("--output-dir", type=Path, default=Path("reports") / "results")
    results.add_argument("--window-size", type=int, default=4096)
    results.add_argument("--stride", type=int, default=2048)
    results.add_argument("--signal", choices=["current", "vibration"], default="current")
    results.add_argument("--max-windows-per-file", type=int, default=8)

    experiment = subparsers.add_parser("experiment", help="Train and evaluate with a held-out file-group split.")
    experiment.add_argument("--data-dir", type=Path, default=Path("data") / "paderborn" / "extracted")
    experiment.add_argument("--model", type=Path, default=Path("models") / "paderborn_final_grouped_pinn.npz")
    experiment.add_argument("--output-dir", type=Path, default=Path("reports") / "results_final")
    experiment.add_argument("--epochs", type=int, default=60)
    experiment.add_argument("--window-size", type=int, default=4096)
    experiment.add_argument("--stride", type=int, default=2048)
    experiment.add_argument("--signal", choices=["current", "vibration"], default="vibration")
    experiment.add_argument("--max-windows-per-file", type=int, default=16)
    experiment.add_argument("--test-fraction", type=float, default=0.25)

    args = parser.parse_args(argv)
    try:
        # Dispatch CLI subcommands to the appropriate pipeline function.
        if args.command == "download":
            archives = download_archives(args.codes, args.raw_dir, overwrite=args.overwrite)
            for archive in archives:
                print(f"{archive.code}: {archive.path}")
            if args.extract:
                extracted = extract_archives(archives, args.extract_dir)
                for path in extracted:
                    print(f"extracted: {path}")
            return 0

        if args.command == "train":
            # Train model from extracted Paderborn files and optionally save weights.
            result = train_from_paderborn(
                data_dir=args.data_dir,
                output_path=args.output,
                epochs=args.epochs,
                window_size=args.window_size,
                stride=args.stride,
                signal_preference=args.signal,
                max_windows_per_file=args.max_windows_per_file,
                synthetic_if_empty=args.synthetic_if_empty,
            )
            print(f"train_accuracy={result.train_accuracy:.3f} test_accuracy={result.test_accuracy:.3f}")
            if result.output_path:
                print(f"saved_model={result.output_path}")
            return 0

        if args.command == "smoke":
            # Run a synthetic smoke test to verify training and saving flow.
            windows, labels, sources = make_synthetic_dataset(samples_per_class=16)
            result = train_from_windows(windows, labels, sources, output_path=args.output, epochs=args.epochs)
            print(f"train_accuracy={result.train_accuracy:.3f} test_accuracy={result.test_accuracy:.3f}")
            print(f"saved_model={result.output_path}")
            return 0

        if args.command == "results":
            # Load a trained model and emit evaluation artifacts for a dataset.
            metrics = generate_results(
                data_dir=args.data_dir,
                model_path=args.model,
                output_dir=args.output_dir,
                window_size=args.window_size,
                stride=args.stride,
                signal_preference=args.signal,
                max_windows_per_file=args.max_windows_per_file,
            )
            print(f"accuracy={metrics['accuracy']:.3f}")
            print(f"sample_count={metrics['sample_count']}")
            print(f"results_dir={args.output_dir}")
            return 0

        if args.command == "experiment":
            # Run grouped experiment with a held-out source-file split.
            result = run_grouped_experiment(
                data_dir=args.data_dir,
                model_path=args.model,
                results_dir=args.output_dir,
                epochs=args.epochs,
                window_size=args.window_size,
                stride=args.stride,
                signal_preference=args.signal,
                max_windows_per_file=args.max_windows_per_file,
                test_fraction=args.test_fraction,
            )
            print(f"train_accuracy={result.train_accuracy:.3f} test_accuracy={result.test_accuracy:.3f}")
            print(f"train_windows={result.train_count} test_windows={result.test_count}")
            print(f"saved_model={result.output_path}")
            print(f"results_dir={args.output_dir}")
            return 0
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
