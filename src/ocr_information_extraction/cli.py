"""Command-line extraction, local dataset preparation, training and evaluation."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .dataset import prepare_dataset, read_json, write_json
from .evaluation import comparison_markdown, evaluate_dataset
from .ml import load_model, train_model
from .pipeline import process_document
from .preprocessing import PreprocessOptions


def _ocr_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--language", default="eng", help="Installed Tesseract language(s), default eng"
    )
    parser.add_argument("--psm", type=int, choices=range(3, 14), default=6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="Extract a single image or multi-page PDF")
    extract.add_argument("document", type=Path)
    extract.add_argument("--output", type=Path, help="Write JSON here; defaults to stdout")
    extract.add_argument("--method", choices=["regex", "svm"], default="regex")
    extract.add_argument("--model", type=Path)
    extract.add_argument("--preview-dir", type=Path, help="Save original/processed page PNGs")
    for option in ("grayscale", "denoise", "threshold", "deskew"):
        extract.add_argument(f"--{option}", action="store_true")
    _ocr_arguments(extract)
    prepare = commands.add_parser(
        "prepare", help="Load local SROIE, freeze splits and run original OCR"
    )
    prepare.add_argument("dataset", type=Path)
    prepare.add_argument("--output", type=Path, default=Path("data/processed"))
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument(
        "--deduplicate",
        action="store_true",
        help="Keep one copy of identical images, prefer test copies, and audit exclusions",
    )
    prepare.add_argument(
        "--exclude-document",
        action="append",
        default=[],
        help="Exclude an explicit dataset-relative image ID before splitting; repeat as needed",
    )
    _ocr_arguments(prepare)
    train = commands.add_parser("train", help="Align training lines and fit TF-IDF + LinearSVC")
    train.add_argument("prepared", type=Path, help="prepared.json from prepare")
    train.add_argument("--model", type=Path, default=Path("models/receipt-svm.joblib"))
    train.add_argument("--c", type=float, default=1.0, help="Tune using validation only")
    evaluate = commands.add_parser(
        "evaluate", help="Compare extractors and preprocessing on frozen test documents"
    )
    evaluate.add_argument("prepared", type=Path)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--model", type=Path)
    evaluate.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    evaluate.add_argument(
        "--variants",
        nargs="*",
        choices=["grayscale", "threshold", "denoise", "deskew"],
        default=["grayscale", "threshold"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            model = load_model(args.model) if args.method == "svm" and args.model else None
            options = PreprocessOptions(
                **{
                    key: getattr(args, key)
                    for key in ("grayscale", "denoise", "threshold", "deskew")
                }
            )
            result = process_document(
                args.document,
                options=options,
                language=args.language,
                psm=args.psm,
                method=args.method,
                model=model,
                preview_dir=args.preview_dir,
            )
            if args.output:
                write_json(args.output, result.model_dump())
                print(f"Saved {len(result.pages)} page(s) to {args.output}")
            else:
                print(result.model_dump_json(indent=2))
        elif args.command == "prepare":
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            prepared = prepare_dataset(
                args.dataset,
                args.output,
                seed=args.seed,
                language=args.language,
                psm=args.psm,
                deduplicate=args.deduplicate,
                exclude_documents=args.exclude_document,
            )
            print(
                f"Saved {len(prepared['documents'])} documents to {args.output / 'prepared.json'}"
            )
            print(json.dumps({key: len(ids) for key, ids in prepared["splits"].items()}))
        elif args.command == "train":
            if not args.prepared.is_file():
                raise ValueError(
                    f"Prepared dataset not found: {args.prepared}. "
                    f"Run 'ocr-extract prepare <SROIE-directory> --output {args.prepared.parent}' "
                    "first and wait until preparation completes."
                )
            report = train_model(read_json(args.prepared), args.model, c=args.c)
            print(f"Saved model to {args.model}; aligned training lines: {report['train_lines']}")
            print(
                f"Inspect {args.model.with_suffix('.alignment.json')} and {args.model.with_suffix('.report.json')}"
            )
        elif args.command == "evaluate":
            if args.output.suffix.lower() != ".json":
                raise ValueError(
                    "Evaluation output must end in .json; a companion .md table is also written."
                )
            model = load_model(args.model) if args.model else None
            report = evaluate_dataset(
                read_json(args.prepared), args.dataset, model=model, variants=args.variants
            )
            write_json(args.output, report)
            table = comparison_markdown(report)
            args.output.with_suffix(".md").write_text(table, encoding="utf-8")
            print(table)
            print(f"Saved per-field scores, predictions and failures to {args.output}")
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
