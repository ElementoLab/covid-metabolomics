#!/usr/bin/env python

"""
Analysis of Olink data from COVID-19 patients.
"""

import sys, io, argparse
from typing import Sequence, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import sklearn  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.decomposition import PCA, NMF  # type: ignore
from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding  # type: ignore
from umap import UMAP  # type: ignore
import statsmodels.api as sm  # type: ignore
import statsmodels.formula.api as smf  # type: ignore

from imc.types import Path, DataFrame  # type: ignore
from seaborn_extensions import clustermap, swarmboxenplot  # type: ignore
from imc.utils import z_score  # type: ignore[import]

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
    "hospitalized",
    "WHO_classification",
    "alive",
]

palettes = dict(
    race=sns.color_palette("tab10"),
    sex=sns.color_palette("Pastel1")[3:5],
    obesity=sns.color_palette("tab10")[3:5],
    hospitalized=sns.color_palette("Set2")[:2],
    WHO_classification=np.asarray(sns.color_palette("Set1"))[
        [2, 1, 7, 3, 4, 0]
    ].tolist(),
    alive=sns.color_palette("Dark2")[:2],
)
cmaps = dict(age="winter_r")

cli = None


def main(cli: Sequence[str] = None) -> int:
    args = get_parser().parse_args(cli)

    x, y = get_x_y()

    unsupervised(x, y)

    get_explanatory_variables(x, y)

    return 0


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    return parser


def get_x_y() -> Tuple[DataFrame, DataFrame]:
    na_values = ["na", "unk", "unkn", "not applicable", "not available"]
    x = pd.read_csv(X_FILE, index_col=0, na_values=na_values)
    y = pd.read_csv(Y_FILE, index_col=0, na_values=na_values)
    y.columns = y.columns.str.replace(".", "_", regex=False)
    assert (x.index == y.index).all()
    assert y["Sample_id"].nunique() == y.shape[0]
    x.index = y["Sample_id"]
    y.index = y["Sample_id"]
    y = y.drop("Sample_id", axis=1)

    y["race"] = pd.Categorical(y["race"])
    y["age"] = y["age"].astype(float)
    y["sex"] = pd.Categorical(y["sex"].str.capitalize())
    y["obesity"] = pd.Categorical(
        y["obesity"].replace("overwheight", "overweight"),
        ordered=True,
        categories=["nonobese", "overweight", "obese"],
    )
    y["hospitalized"] = pd.Categorical(
        y["hospitalized"].replace({"no": False, "yes": True})
    )
    y["WHO_classification"] = pd.Categorical(
        y["WHO_classification"].replace("mid", "mild"),
        ordered=True,
        categories=["uninfected", "low", "mild", "moderate", "severe"],
    )
    y["alive"] = pd.Categorical(
        y["alive"], ordered=True, categories=["alive", "dead"]
    )

    return x, y


def unsupervised(x, y, attributes: List[str] = []) -> None:

    output_dir = (results_dir / "unsupervised").mkdir()

    ## Clustermaps
    for c in ["abs", "z"]:
        grid = clustermap(
            x, row_colors=y[attributes], config=c, rasterized=True
        )
        grid.savefig(
            output_dir / f"unsupervised.clustering.clustermap.{c}.svg",
            **figkws,
        )
    kws = dict(
        cmap="RdBu_r",
        rasterized=True,
        cbar_kws=dict(label="Pearson correlation"),
        xticklabels=False,
        yticklabels=False,
    )
    grid = clustermap(z_score(x).corr(), center=0, **kws)
    grid.savefig(
        output_dir / f"unsupervised.correlation_variable.clustermap.svg",
        **figkws,
    )
    grid = clustermap(z_score(x).T.corr(), **kws, row_colors=y[attributes])
    grid.savefig(
        output_dir / f"unsupervised.correlation_samples.clustermap.svg",
        **figkws,
    )

    ## Dimres
    for model, pkwargs, mkwargs in [
        (PCA, dict(), dict()),
        (NMF, dict(), dict()),
        (MDS, dict(n_dims=1), dict()),
        (TSNE, dict(n_dims=1), dict()),
        (Isomap, dict(n_dims=1), dict()),
        (UMAP, dict(n_dims=1), dict(random_state=0)),
        (SpectralEmbedding, dict(n_dims=1), dict()),
    ][::-1]:
        name = str(model).split(".")[-1].split("'")[0]
        model_inst = model(**mkwargs)

        for transf, label in [(lambda x: x, ""), (z_score, "Z-score.")]:
            try:  #  this will occur for example in NMF with Z-score transform
                res = pd.DataFrame(
                    model_inst.fit_transform(transf(x)), index=x.index
                )
            except ValueError:
                continue

            fig = _plot_projection(  # type: ignore[misc]
                res,
                y,
                factors=attributes,
                algo_name=name,
                **pkwargs,
            )
            fig.savefig(
                output_dir / f"unsupervised.dimres.{name}.{label}svg",
                **figkws,
            )
            plt.close(fig)


def get_explanatory_variables(x, y) -> None:

    output_dir = (results_dir / "unsupervised").mkdir()

    res = pd.DataFrame(
        SpectralEmbedding().fit_transform(z_score(x)),
        index=x.index,
        columns=["SE1", "SE2"],
    )
    corr_mat = res.join(z_score(x)).corr().loc[x.columns, res.columns]
    corr_mat.to_csv(
        output_dir
        / "unsupervised.variable_contibution_SpectralEmbedding.correlation.csv"
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # # plot samples
    ax.scatter(*res.T.values)
    # # plot variables as vectors
    for i in corr_mat.index:
        ax.plot(
            (0, corr_mat.loc[i, "SE1"] / 10), (0, corr_mat.loc[i, "SE2"] / 10)
        )
    fig.savefig(
        output_dir
        / "unsupervised.variable_contibution_SpectralEmbedding.correlation.scatter_vectors.svg",
        **figkws,
    )

    xz = res.join(z_score(x))
    _coefs = list()
    for var in x.columns:
        res1 = smf.ols(f"SE1 ~ {var}", data=xz).fit()
        res2 = smf.ols(f"SE2 ~ {var}", data=xz).fit()
        _coefs.append(
            res1.summary2()
            .tables[1]
            .assign(var="SE1")
            .append(res2.summary2().tables[1].assign(var="SE2"))
        )
    coefs = pd.concat(_coefs).drop("Intercept").rename_axis(index="variable")
    coefs2 = coefs.pivot_table(index="variable", columns="var", values="Coef.")
    coefs2.to_csv(
        output_dir
        / "unsupervised.variable_contibution_SpectralEmbedding.regression.csv"
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # # plot samples
    ax.scatter(*res.T.values)
    # # plot variables as vectors
    cmap = plt.get_cmap("inferno")
    vmin = coefs2.abs().sum(1).min()
    vmax = coefs2.abs().sum(1).max()
    for i in coefs2.index:
        c = coefs2.loc[i].abs().sum()
        ax.plot(
            (0, coefs2.loc[i, "SE1"] * 2),
            (0, coefs2.loc[i, "SE2"] * 2),
            c=cmap((c - vmin) / (vmax - vmin)),
        )
        if c > 0.027:
            ax.text(coefs2.loc[i, "SE1"] * 2, coefs2.loc[i, "SE2"] * 2, s=i)
    fig.savefig(
        output_dir
        / "unsupervised.variable_contibution_SpectralEmbedding.regression.scatter_vectors.svg",
        **figkws,
    )

    sample_order = res.sort_values("SE1").index
    var_order = coefs2.sort_values("SE1").index
    grid = clustermap(
        z_score(x).loc[sample_order, var_order],
        col_cluster=False,
        row_cluster=False,
        center=0,
        cmap="RdBu_r",
        robust=True,
        row_colors=y[attributes].join(res["SE1"]),
        figsize=(16, 6),
        rasterized=True,
    )
    grid.savefig(
        output_dir
        / "unsupervised.variable_contibution_SpectralEmbedding.regression.ordered.clustermap.svg",
        **figkws,
    )


def supervised():
    output_dir = (results_dir / "supervised").mkdir()

    # first just get the stats
    fig, stats = swarmboxenplot(
        data=x.join(y),
        x="WHO_classification",
        y=x.columns,
        swarm=False,
        boxen=False,
    )
    plt.close(fig)

    # now plot top variables
    fig, s2 = swarmboxenplot(
        data=x.join(y),
        x="WHO_classification",
        y=stats.sort_values("p-unc")["Variable"].head(20).unique(),
        plot_kws=dict(palette=palettes.get("WHO_classification")),
    )
    fig.savefig(
        output_dir / "supervised.top_differential.swarmboxenplot.svg",
        **figkws,
    )


def _plot_projection(x_df, y_df, factors, n_dims=4, algo_name="PCA"):
    from seaborn_extensions.annotated_clustermap import to_color_series  # type: ignore[import]
    from seaborn_extensions.annotated_clustermap import is_numeric

    factors = [c for c in factors if c in y_df.columns]
    n = len(factors)
    fig, axes = plt.subplots(
        n,
        n_dims,
        figsize=(4 * n_dims, 4 * n),
        sharex="col",
        sharey="col",
        squeeze=False,
    )

    for i, factor in enumerate(factors):
        numeric = is_numeric(y_df[factor])
        try:
            colors = pd.Series(palettes.get(factor), dtype="object").reindex(
                y_df[factor].dropna().cat.codes
            )
            colors.index = y_df[factor].dropna().index
        except AttributeError:  # not a categorical
            try:
                colors = to_color_series(
                    y_df[factor].dropna(), palettes.get(cat)
                )
            except (TypeError, ValueError):
                colors = to_color_series(y_df[factor].dropna())
        for pc in x_df.columns[:n_dims]:
            ax = axes[i, pc]
            if numeric:
                m = ax.scatter(
                    x_df.loc[:, pc],
                    x_df.loc[:, pc + 1],
                    c=y_df[factor],
                    cmap=cmaps.get(factor),
                )
                if pc == 0:
                    bb = ax.get_position()
                    cax = fig.add_axes(
                        (bb.xmax, bb.ymin, bb.width * 0.05, bb.height)
                    )
                    cbar = fig.colorbar(m, label=factor, cax=cax)
            else:
                for value in y_df[factor].dropna().unique():
                    idx = y_df[factor].isin([value])  # to handle nan correctly
                    ax.scatter(
                        x_df.loc[idx, pc],
                        x_df.loc[idx, pc + 1],
                        c=colors.loc[idx] if not numeric else None,
                        cmap=cmaps.get(factor) if numeric else None,
                        label=value,
                    )
            if pc == 0:
                ax.legend(
                    title=factor,
                    loc="center right",
                    bbox_to_anchor=(-0.15, 0.5),
                )
            ax.set_ylabel(algo_name + str(pc + 2))

    for i, ax in enumerate(axes[-1, :]):
        ax.set_xlabel(algo_name + str(i + 1))
    return fig


if __name__ == "__main__":
    from typing import List

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
