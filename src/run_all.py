
from __future__ import annotations

import argparse

from src.run_medgemma import run_pipeline as run_medgemma
from src.run_transformers import run_pipeline as run_conventional


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete reproducible study.")
    parser.add_argument("--force", action="store_true", help="Rerun completed stages.")
    args = parser.parse_args()

    print("=" * 78)
    print("PART 1/2 — CONVENTIONAL NER + METHODOLOGICAL CONTROLS")
    print("=" * 78)
    run_conventional(force=args.force)

    print("\n" + "=" * 78)
    print("PART 2/2 — MEDGEMMA")
    print("=" * 78)
    run_medgemma(force=args.force)

    print("\nCOMPLETE STUDY FINISHED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
