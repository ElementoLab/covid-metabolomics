#!/usr/bin/env python

"""
Configuration, constants and helper functions for the project.
"""

import numpy as np
import seaborn as sns

from imc.types import Path, DataFrame  # type: ignore


class PyMDE:
    import pymde

    def fit_transform(self, x, embedding_dim=2, **kwargs):
        if isinstance(x, pd.DataFrame):
            x = x.values
        embedding = (
            pymde.preserve_neighbors(x, embedding_dim=embedding_dim, **kwargs)
            .embed()
            .numpy()
        )
        return embedding


class DiffMap:
    from anndata import AnnData

    def fit_transform(self, x, embedding_dim=2, **kwargs):
        a = AnnData(x)
        sc.pp.neighbors(a, use_rep="X")
        sc.tl.diffmap(a)
        return a.obsm["X_diffmap"][:, 1:3]


figkws = dict(dpi=300, bbox_inches="tight")

metadata_dir = Path("metadata")
data_dir = Path("data")
results_dir = Path("results")

X_FILE = data_dir / "assay_data.csv"
Y_FILE = data_dir / "metadata.csv"


attributes = [
    "race",
    "age",
    "sex",
    "obesity",
    "bmi",
    "hospitalized",
    "patient_group",
    "WHO_score_sample",
    "WHO_score_patient",
    "alive",
]

palettes = dict(
    race=sns.color_palette("tab10"),
    sex=sns.color_palette("Pastel1")[3:5],
    obesity=sns.color_palette("tab10")[3:5],
    hospitalized=sns.color_palette("Set2")[:2],
    patient_group=np.asarray(sns.color_palette("Set1"))[
        [2, 1, 7, 3, 0]
    ].tolist(),
    WHO_score_sample=sns.color_palette("inferno", 9),
    WHO_score_patient=sns.color_palette("inferno", 9),
    alive=sns.color_palette("Dark2")[:2],
)
cmaps = dict(age="winter_r", bmi="copper")
