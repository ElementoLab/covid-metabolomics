from __future__ import annotations
from dataclasses import dataclass, field
import typing as tp

import pandas as pd
import pymde
import scanpy as sc
from anndata import AnnData

from imc.types import DataFrame, Path

__all__ = ["DataSet", "PyMDE", "DiffMap"]


@dataclass
class DataSet:
    x: DataFrame
    obs: DataFrame = None
    var: DataFrame = None
    attributes: tp.Sequence[str] = ()
    cmaps: tp.Dict[str, str] = field(default_factory=dict)
    palettes: tp.Dict[str, tp.Sequence[tp.Tuple[float, float, float, float]]] = field(
        default_factory=dict
    )
    metadata_dir: Path = Path("metadata").resolve()
    data_dir: Path = Path("data").resolve()
    results_dir: Path = Path("results").resolve()

    def __repr__(self):
        return f"DataSet with {self.n_obs} observations and {self.n_vars} variables."

    @property
    def n_obs(self):
        return self.x.shape[0]

    @property
    def n_vars(self):
        return self.x.shape[1]

    @property
    def shape(self):
        return self.x.shape

    # @x.setter
    # def x(self, df: DataFrame) -> None:
    #     self.x = df
    #     if self.obs is not None:
    #         self.obs = self.obs.reindex(index=self.x.index)
    #     if self.var is not None:
    #         self.var = self.var.reindex(index=self.x.columns)


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
        return a.obsm["X_diffmap"][:, 1 : 1 + embedding_dim]
