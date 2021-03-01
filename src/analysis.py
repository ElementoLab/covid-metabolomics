#!/usr/bin/env python

"""
Analysis of Olink data from COVID-19 patients.
"""

import sys
import argparse
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn

from imc.types import Path, DataFrame


figkws = dict(dpi=300, bbox_inches="tight")

metadata_dir = Path("metadata")
data_dir = Path("data")
results_dir = Path("results")

X_FILE = data_dir / "assay_data.csv"
Y_FILE = data_dir / "metadata.csv"


cli = None


def main(cli: Sequence[str] = None) -> int:
    args = get_parser().parse_args(cli)

    x, y = get_x_y()

    return 0


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    return parser


def get_x_y() -> Tuple[DataFrame, DataFrame]:
    x = pd.read_csv(X_FILE, index_col=0)
    y = pd.read_csv(Y_FILE, index_col=0)
    y.columns = y.columns.str.replace(".", "_", regex=False)
    assert (x.index == y.index).all()
    assert y["Sample_id"].nunique() == y.shape[0]
    x.index = y["Sample_id"]
    y.index = y["Sample_id"]
    y = y.drop("Sample_id", axis=1)
    return x, y


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
