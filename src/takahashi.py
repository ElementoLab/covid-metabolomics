#!/usr/bin/env python

"""
Analysis of immunological data from a COVID-19 cohort
(10.1038/s41586-020-2700-3).
"""

import sys
import typing as tp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imc.types import DataFrame, Array
from imc.graphics import close_plots
from seaborn_extensions import clustermap, swarmboxenplot

from src.config import data_dir, metadata_dir, results_dir, figkws
from src.analysis import plot_projection


attributes = ["Age", "Sex", "BMI", "DFSO", "ICU", "Clinical_score"]
palettes = dict(
    Sex=sns.color_palette("Pastel1")[3:5],
    ICU=[(0.1, 0.1, 0.1)] + sns.color_palette("Set2")[:2],
    Clinical_score=np.asarray(sns.color_palette("Set1"))[[2, 1, 7, 3, 4, 0]].tolist(),
)
cmaps = dict(
    Age="winter_r",
    BMI="copper",
    DFSO="cividis",
)
output_dir = (results_dir / "takahashi").mkdir()


def main() -> int:
    """The main function to run the analysis."""

    x, y = get_suppl_data()

    summary_plots(x, y)

    x_imp = impute_x(np.log1p(x))

    unsupervised(x_imp, y.reindex(x_imp.index), attributes)

    # Fin
    return 0


def get_suppl_data() -> tp.Tuple[DataFrame, DataFrame]:
    """
    Download and prepare data from supplementary material of original manuscript.
    """
    from urlpath import URL

    root = URL(
        "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-2700-3/MediaObjects/"
    )
    url = root / "41586_2020_2700_MOESM1_ESM.xlsx"

    df = pd.read_excel(url, index_col=0, skiprows=23)
    df.columns = df.columns.str.strip()

    var = get_feature_annotations(df)
    # dft = df.T.join(var)

    # Unlogaritmize cytokine_chemokine
    sel = var["feature_type"] == "cytokine_chemokine"
    df.loc[:, sel] = 10 ** df.loc[:, sel]

    # Split into clinical and lab variables
    sel = var["feature_type"].isin(["demographic", "disease"])
    x = df.loc[:, ~sel].astype(float).copy()
    y = df.loc[:, sel].copy()
    y["individual_id"] = y.index.str.replace(r"\..*", "", regex=True)
    y["patient"] = (
        y["individual_id"]
        .str.startswith("Pt")
        .replace({False: "healthy", True: "COVID-19"})
    )
    for col in y.columns:
        try:
            y[col] = y[col].astype(float)
        except ValueError:
            y[col] = y[col].astype(pd.CategoricalDtype())
    y["Clinical_score"] = pd.Categorical(y["Clinical score"].astype(float), ordered=True)
    y = y.drop("Clinical score", axis=1)
    y["Age"] = y["Age"].replace("≥ 90", 95).astype(pd.Int64Dtype())
    y["Ethnicity"] = y["Ethnicity"].replace(98, np.nan).astype(pd.CategoricalDtype())
    y = y.assign(Sex=y["sex"]).drop("sex", 1)
    y["ICU"] = y["ICU"].astype(pd.CategoricalDtype())

    return x, y


def get_feature_annotations(x: DataFrame) -> DataFrame:
    import json

    var_annot = json.load(open(metadata_dir / "takahashi.variable_classes.json"))
    var = pd.Series({x: k for k, v in var_annot.items() for x in v}, name="feature_type")
    return var.to_frame().reindex(x.columns).astype(pd.CategoricalDtype())


@close_plots
def summary_plots(x: DataFrame, y: DataFrame) -> None:

    # Clinical data
    fig, axes = plt.subplots(1, 2, figsize=(2 * 3, 2))
    sns.histplot(y.groupby("individual_id").size(), ax=axes[0])
    sns.histplot(
        y.query("patient == 'COVID-19'").groupby("individual_id").size(), ax=axes[1]
    )
    axes[0].set(title="All individuals", xlabel="Number of samples", ylabel="Frequency")
    axes[1].set(title="COVID-19 patients", xlabel="Number of samples", ylabel="Frequency")
    fig.savefig(
        output_dir / "summary_plots.samples_per_individual.histplot.svg", **figkws
    )

    fig, axes = plt.subplots(1, 2, figsize=(2 * 3, 2))
    sns.histplot(y["DFSO"], ax=axes[0])
    sns.histplot(
        y.query("patient == 'COVID-19'").groupby("individual_id")["DFSO"].diff(),
        ax=axes[1],
    )
    axes[0].set(xlabel="Days from symptom onset", ylabel="Frequency")
    axes[1].set(xlabel="Interval between sampled timepoints (days)", ylabel="Frequency")
    fig.savefig(output_dir / "summary_plots.frequency_of_sampling.histplot.svg", **figkws)

    fig, axes = plt.subplots(3, 3, figsize=(3 * 4, 3 * 4))
    factors = ["Sex", "Clinical_score", "ICU"]
    for factor, axs in zip(factors, axes.T):
        swarmboxenplot(data=y, x=factor, y=["DFSO", "BMI", "Age"], ax=axs)
    for ax, label in zip(axes[-1], factors):
        ax.set(xlabel=label)
    for ax in axes[0]:
        ax.set(title="Days from symptom onset")
    fig.savefig(
        output_dir / "summary_plots.time_bmi_age_severity.swarmboxenplot.svg", **figkws
    )

    # Data
    var = get_feature_annotations(x)

    # # Missing data
    fig, axes = plt.subplots(1, 2, figsize=(2 * 3, 1 * 3), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, ["Feature", "Sample"])):
        sns.histplot(x.isnull().sum(i) / x.shape[i] * 100, bins=25, ax=ax)
        ax.set(xlabel=f"Missing data per {label} (%)", ylabel="Frequency")
    axes[0].axvline(50, linestyle="--", color="grey")
    axes[1].axvline(80, linestyle="--", color="grey")
    fig.savefig(output_dir / "summary_plots.missing_data.histogram.svg", **figkws)

    # # Mean/variance relationship
    fig, axes = plt.subplots(1, 2, figsize=(2 * 4, 1 * 4))
    ax = axes[0]
    for cat in var["feature_type"].unique():
        _x = x.loc[:, var["feature_type"] == cat]
        ax.scatter(_x.mean(), _x.std(), label=cat, alpha=0.5)
    ax.loglog()
    ax.legend()
    vmin = min(x.mean().min(), x.std().min())
    vmax = max(x.mean().max(), x.std().max())
    ax.plot((vmin, vmax), (vmin, vmax), linestyle="--", color="grey")
    ax.set(xlabel="Mean", ylabel="Standard deviation")

    ax = axes[1]
    for cat in var["feature_type"].unique():
        _x = x.loc[:, var["feature_type"] == cat]
        ax.scatter(_x.mean(), _x.std() / _x.mean(), label=cat, alpha=0.5)
    ax.loglog()
    ax.legend()
    vmin = min(x.mean().min(), x.std().min())
    vmax = max(x.mean().max(), x.std().max())
    ax.plot((vmin, vmax), (vmin, vmax), linestyle="--", color="grey")
    ax.set(xlabel="Mean", ylabel="Coefficient of variation")
    fig.savefig(output_dir / "summary_plots.data_variance.scatterplot.svg", **figkws)


def impute_x(
    x: DataFrame,
    frac_obs: float = 0.8,
    frac_var: float = 0.5,
    method="factorization",
    save: bool = True,
) -> DataFrame:
    from fancyimpute import MatrixFactorization, KNN

    x_file = data_dir / "takahashi.imputed.csv"

    if not x_file.exists():
        null = x.isnull()
        missing = (null.values.sum() / x.size) * 100
        print(f"Dataset has {missing:.3f}% missing data.")  # 27.936%

        # First remove samples/vars with too many missing values
        x2 = x.loc[
            null.sum(1) / x.shape[1] < frac_obs, null.sum(0) / x.shape[0] < frac_var
        ]
        if method == "factorization":
            model = MatrixFactorization(learning_rate=0.01, epochs=500)
        elif method == "knn":
            model = KNN(15)
        x_imp = pd.DataFrame(
            model.fit_transform(x2),
            index=x2.index,
            columns=x2.columns,
        )
        # clip variables to zero
        if save:
            x_imp.clip(lower=0).to_csv(x_file)
    x_imp = pd.read_csv(x_file, index_col=0)

    return x_imp


@close_plots
def unsupervised(x: DataFrame, y: DataFrame, factors: tp.Sequence[str]) -> None:
    from imc.utils import z_score

    from sklearn.decomposition import PCA, NMF
    from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding
    from umap import UMAP
    from src.config import PyMDE, DiffMap

    var = get_feature_annotations(x)
    grid = clustermap(
        x, config="z", row_colors=y[factors], col_colors=var, figsize=(20, 12)
    )
    grid.fig.savefig(output_dir / "unsupervised.clustermap.svg", **figkws)

    ## Clustermaps
    for c in ["abs", "z"]:
        grid = clustermap(
            x,
            row_colors=y[factors],
            col_colors=var,
            config=c,
            rasterized=True,
        )
        grid.savefig(
            output_dir / f"unsupervised.clustering.clustermap.{c}.svg",
            **figkws,
        )
        plt.close(grid.fig)
    kws = dict(
        cmap="RdBu_r",
        rasterized=True,
        cbar_kws=dict(label="Pearson correlation"),
        xticklabels=False,
        yticklabels=False,
    )
    grid = clustermap(z_score(x).corr(), center=0, **kws, row_colors=var)
    grid.savefig(
        output_dir / "unsupervised.correlation_variable.clustermap.svg",
        **figkws,
    )
    plt.close(grid.fig)

    grid = clustermap(z_score(x).T.corr(), **kws, row_colors=y[factors])
    grid.savefig(
        output_dir / "unsupervised.correlation_samples.clustermap.svg",
        **figkws,
    )
    plt.close(grid.fig)

    ## Dimres
    for model, pkwargs, mkwargs in [
        (PCA, dict(), dict()),
        (NMF, dict(), dict()),
        (MDS, dict(n_dims=1), dict()),
        (TSNE, dict(n_dims=1), dict()),
        (Isomap, dict(n_dims=1), dict()),
        (UMAP, dict(n_dims=1), dict(random_state=0)),
        (DiffMap, dict(n_dims=1), dict()),
        (PyMDE, dict(n_dims=1), dict()),
        (SpectralEmbedding, dict(n_dims=1), dict()),
    ][::-1]:
        # model, pkwargs, mkwargs = (PyMDE, dict(), dict())
        name = str(model).split(".")[-1].split("'")[0]
        model_inst = model(**mkwargs)

        for transf, label in [(lambda x: x, ""), (z_score, "Z-score.")]:
            try:  #  this will occur for example in NMF with Z-score transform
                res = pd.DataFrame(model_inst.fit_transform(transf(x)), index=x.index)
            except ValueError:
                continue

            fig = plot_projection(
                res, y.reindex(res.index), factors=factors, algo_name=name, **pkwargs
            )
            fig.savefig(
                output_dir / f"unsupervised.dimres.{name}.{label}svg",
                **figkws,
            )
            plt.close(fig)


def supervised(x: DataFrame, y: DataFrame) -> None:

    _, stats = swarmboxenplot(
        data=x.join(y), x="Clinical_score", y=x.columns, swarm=False, boxen=False
    )
    stats.to_csv(output_dir / "supervised.stats.csv")

    fig, _stats = swarmboxenplot(
        data=np.log(x).join(y),
        x="Clinical_score",
        y=stats.sort_values("p-unc").head(35)["Variable"].unique(),
    )
    fig.savefig(output_dir / "supervised.top_changing.svg", **figkws)
    plt.close(fig)


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
