"""Regenerate the submission CSV from the tactical bundle.

    export MTAH_DATA_DIR=/content/drive/MyDrive/MoreThanAHack/data
    python scripts/build_submission.py --out evaluation/results/submission.csv
    python evaluation/validate_submission.py evaluation/results/submission.csv

The public five-minute score is a wiring check. Do not pick these constants
by climbing that number.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from mtah.build import build_labels, summarise, write_submission  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "evaluation", "results", "submission.csv"),
        help="where to write the frame,label CSV",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="bundle root (default: MTAH_DATA_DIR or ./data)",
    )
    args = parser.parse_args()

    labels = build_labels(args.data_dir)
    path = write_submission(args.out, labels)
    print("wrote {}".format(os.path.relpath(path, ROOT)))
    print(summarise(labels))
    print("\nNext: python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
