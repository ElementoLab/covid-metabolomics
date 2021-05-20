#!/usr/bin/env python

"""
Configuration, constants and helper functions for the project.
"""
from __future__ import annotations
import typing as tp

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import scanpy as sc
import pymde

from imc.types import Path, DataFrame  # type: ignore


class PyMDE:
    def fit_anndata(self, anndata, config="default") -> PyMDE:
        if config == "default":
            anndata.obsm["X_pymde"] = self.fit_transform(anndata.X)
        elif config == "alternate":
            anndata.obsm["X_pymde_alt"] = self.fit_transform(
                anndata.X,
                embedding_dim=2,
                attractive_penalty=pymde.penalties.Quadratic,
                repulsive_penalty=None,
            )
        return self

    def fit_transform(self, x, embedding_dim: int = 2, **kwargs):
        if isinstance(x, pd.DataFrame):
            x = x.values
        embedding = (
            pymde.preserve_neighbors(x, embedding_dim=embedding_dim, **kwargs)
            .embed()
            .numpy()
        )
        return embedding


class DiffMap:
    def fit_transform(self, x, embedding_dim: int = 2, **kwargs):
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
    "days_since_symptoms",
    "days_since_hospitalization",
]

palettes = dict(
    race=sns.color_palette("tab10"),
    sex=sns.color_palette("Pastel1")[3:5],
    obesity=sns.color_palette("tab10")[3:5],
    hospitalized=sns.color_palette("Set2")[:2],
    patient_group=np.asarray(sns.color_palette("Set1"))[[2, 1, 7, 3, 0]].tolist(),
    WHO_score_sample=sns.color_palette("inferno", 9),
    WHO_score_patient=sns.color_palette("inferno", 9),
    alive=sns.color_palette("Dark2")[:2],
)
cmaps = dict(
    age="winter_r",
    bmi="copper",
    days_since_symptoms="cividis",
    days_since_hospitalization="cividis",
)

_q = np.random.choice(range(40), 40, replace=False)
tab40 = matplotlib.colors.ListedColormap(
    colors=np.concatenate(
        [
            plt.get_cmap("tab20c")(range(20)),
            plt.get_cmap("tab20b")(range(20)),
        ]
    )[_q],
    name="tab40",
)
