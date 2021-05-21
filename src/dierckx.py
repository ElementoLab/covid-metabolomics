#!/usr/bin/env python

"""
Analysis of NMR data of metabolites from blood serum of a Belgian cohort
(10.1101/2020.11.09.20228221).
"""

import sys
import typing as tp

import pandas as pd
import numpy as np

from imc.types import DataFrame

cli = None


def main(cli: tp.Sequence[str] = None) -> int:
    """The main function to run the analysis."""

    get_signatures()

    # Fin
    return 0


def get_signatures() -> DataFrame:
    """
    Extract sigantures of progression (day 7 vs admission),
    and remission (discharge vs day 7)
    """
    from urlpath import URL

    root = URL("https://www.medrxiv.org/content/medrxiv/")
    url = (
        root / "early/2020/11/12/2020.11.09.20228221/DC5/embed/media-5.xlsx?download=true"
    )

    df = pd.read_excel(url, index_col=0)
    df = df.drop("InfectiousDiseaseScore")
    for time in ["D0", "D7", "Dis"]:
        df[f"{time}_mean"] = (
            df[f"{time}_IQR"].str.split("-").apply(lambda x: float(x[1]) - float(x[0]))
        )

    sigs = pd.DataFrame(index=df.index)
    sigs["progression"] = np.log(df["D7_mean"] / df["D0_mean"]).sort_values()
    sigs["resolution"] = np.log(df["Dis_mean"] / df["D7_mean"]).sort_values()
    return sigs.sort_index()


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
