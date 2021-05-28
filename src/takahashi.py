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

from imc.types import DataFrame
from imc.graphics import close_plots
from seaborn_extensions import clustermap, swarmboxenplot

from src.config import data_dir, metadata_dir, results_dir, figkws
from src.models import DataSet
from src.ops import unsupervised, overlay_individuals_over_global


attributes = ["Age", "Sex", "BMI", "DFSO", "ICU", "Clinical_score", "Outcome"]
palettes = dict(
    Sex=sns.color_palette("Pastel1")[3:5],
    ICU=[(0.1, 0.1, 0.1)] + sns.color_palette("Set2")[:2],
    Clinical_score=np.asarray(sns.color_palette("Set1"))[[2, 1, 7, 3, 4, 0]].tolist(),
    WHO_score_sample=np.asarray(sns.color_palette("Set1"))[[2, 1, 7, 3, 4, 0]].tolist(),
    Outcome=np.asarray(sns.color_palette("Set1"))[[3, 4, 0, 2]].tolist(),
)
cmaps = dict(
    Age="winter_r",
    BMI="copper",
    DFSO="cividis",
)
output_dir = (results_dir / "takahashi").mkdir()


def main() -> int:
    """The main function to run the analysis."""
    d = get_dataset()

    summary_plots(d)

    d = impute(d, frac_obs=0.25, frac_var=0.3)

    unsupervised(d.x, d.obs, attributes=attributes, data_type="takahashi")
    unsupervised2(d.x, d.obs, attributes)

    overlay_individuals_over_global(d.x, d.obs, data_type="takahashi")

    sel = d.obs.index.str.startswith("Pt")
    unsupervised(
        d.x.loc[sel],
        d.obs.loc[sel],
        attributes=attributes,
        data_type="takahashi",
        suffix="only_patients",
    )
    overlay_individuals_over_global(
        d.x.loc[sel], d.obs.loc[sel], data_type="takahashi", suffix="only_patients"
    )

    # Fin
    return 0


def get_dataset() -> DataSet:
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

    # for compatibility with NMR data:
    date_diagnosis = pd.to_datetime("2020/02/01")
    y["date_sample"] = date_diagnosis + pd.to_timedelta(y["DFSO"], unit="day")
    y["patient_code"] = y["individual_id"]
    y["accession"] = y.index
    y["WHO_score_sample"] = y["Clinical_score"]

    # The supplementary material from the Lucas et al paper has one more column of interest:
    root = URL(
        "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-2588-y"
    )
    url = root / "MediaObjects/41586_2020_2588_MOESM3_ESM.xlsx"
    df2 = pd.read_excel(url, index_col=0, skiprows=26)
    y["Outcome"] = pd.Categorical(
        y.join(df2["LatestOutcome"])["LatestOutcome"].replace(
            {0: "still admitted", 1: "discharged", 2: "deceased", 3: "CMO/hospice"}
        ),
        ordered=True,
        categories=["still admitted", "CMO/hospice", "deceased", "discharged"],
    )
    # # for compatibility with NMR data:
    y["alive"] = y["Outcome"]

    return DataSet(
        x=x,
        obs=y,
        var=var,
        name="takahashi",
        data_type="immune",
        attributes=attributes,
        metadata_dir=metadata_dir,
        data_dir=data_dir,
        results_dir=results_dir,
        palettes=palettes,
        cmaps=cmaps,
    )


def get_feature_annotations(x: DataFrame) -> DataFrame:
    import json

    with open(metadata_dir / "takahashi.variable_classes.json") as handle:
        var_annot = json.load(handle)
    var = pd.Series({x: k for k, v in var_annot.items() for x in v}, name="feature_type")
    return var.to_frame().reindex(x.columns).astype(pd.CategoricalDtype())


@close_plots
def summary_plots(d: DataSet) -> None:
    output_prefix = output_dir / f"{d.name}.{d.data_type}.summary_plots."

    # Clinical data
    fig, axes = plt.subplots(1, 2, figsize=(2 * 3, 2))
    sns.histplot(d.obs.groupby("individual_id").size(), ax=axes[0])
    sns.histplot(
        d.obs.query("patient == 'COVID-19'").groupby("individual_id").size(), ax=axes[1]
    )
    axes[0].set(title="All individuals", xlabel="Number of samples", ylabel="Frequency")
    axes[1].set(title="COVID-19 patients", xlabel="Number of samples", ylabel="Frequency")
    fig.savefig(output_prefix + "samples_per_individual.histplot.svg", **figkws)

    fig, axes = plt.subplots(1, 2, figsize=(2 * 3, 2))
    sns.histplot(d.obs["DFSO"], ax=axes[0])
    sns.histplot(
        d.obs.query("patient == 'COVID-19'").groupby("individual_id")["DFSO"].diff(),
        ax=axes[1],
    )
    axes[0].set(xlabel="Days from symptom onset", ylabel="Frequency")
    axes[1].set(xlabel="Interval between sampled timepoints (days)", ylabel="Frequency")
    fig.savefig(output_prefix + "frequency_of_sampling.histplot.svg", **figkws)

    fig, axes = plt.subplots(3, 3, figsize=(3 * 4, 3 * 4))
    factors = ["Sex", "Clinical_score", "ICU"]
    for factor, axs in zip(factors, axes.T):
        swarmboxenplot(data=d.obs, x=factor, y=["DFSO", "BMI", "Age"], ax=axs)
    for ax, label in zip(axes[-1], factors):
        ax.set(xlabel=label)
    for ax in axes[0]:
        ax.set(title="Days from symptom onset")
    fig.savefig(output_prefix + "time_bmi_age_severity.swarmboxenplot.svg", **figkws)

    # Data
    var = get_feature_annotations(d.x)

    # # Missing data
    fig, axes = plt.subplots(1, 2, figsize=(2 * 3, 1 * 3), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, ["Feature", "Sample"])):
        sns.histplot(d.x.isnull().sum(i) / d.x.shape[i] * 100, bins=25, ax=ax)
        ax.set(xlabel=f"Missing data per {label} (%)", ylabel="Frequency")
    axes[0].axvline(50, linestyle="--", color="grey")
    axes[1].axvline(80, linestyle="--", color="grey")
    fig.savefig(output_prefix + "missing_data.histogram.svg", **figkws)

    # # Mean/variance relationship
    fig, axes = plt.subplots(1, 2, figsize=(2 * 4, 1 * 4))
    ax = axes[0]
    for cat in var["feature_type"].unique():
        _x = d.x.loc[:, var["feature_type"] == cat]
        ax.scatter(_x.mean(), _x.std(), label=cat, alpha=0.5)
    ax.loglog()
    ax.legend()
    vmin = min(d.x.mean().min(), d.x.std().min())
    vmax = max(d.x.mean().max(), d.x.std().max())
    ax.plot((vmin, vmax), (vmin, vmax), linestyle="--", color="grey")
    ax.set(xlabel="Mean", ylabel="Standard deviation")

    ax = axes[1]
    for cat in var["feature_type"].unique():
        _x = d.x.loc[:, var["feature_type"] == cat]
        ax.scatter(_x.mean(), _x.std() / _x.mean(), label=cat, alpha=0.5)
    ax.loglog()
    ax.legend()
    vmin = min(d.x.mean().min(), d.x.std().min())
    vmax = max(d.x.mean().max(), d.x.std().max())
    ax.plot((vmin, vmax), (vmin, vmax), linestyle="--", color="grey")
    ax.set(xlabel="Mean", ylabel="Coefficient of variation")
    fig.savefig(output_prefix + "data_variance.scatterplot.svg", **figkws)


def impute(
    d: DataSet,
    log: bool = True,
    frac_obs: float = 0.8,
    frac_var: float = 0.5,
    method="factorization",
    save: bool = True,
) -> DataSet:
    from fancyimpute import MatrixFactorization, KNN

    x_file = data_dir / f"{d.name}.{d.data_type}.imputed.csv"

    if not x_file.exists():
        null = d.x.isnull()
        missing = (null.values.sum() / d.x.size) * 100
        print(f"Dataset has {missing:.3f}% missing data.")  # 27.936%

        # First remove samples/vars with too many missing values
        x2 = d.x.loc[
            null.sum(1) / d.x.shape[1] < frac_obs, null.sum(0) / d.x.shape[0] < frac_var
        ]
        if log:
            x2 = np.log1p(x2)
        if method == "factorization":
            model = MatrixFactorization(learning_rate=0.01, epochs=500, verbose=False)
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
    else:
        print("Using pre-exising dataset.")
    d.x = pd.read_csv(x_file, index_col=0)

    d.obs = d.obs.reindex(index=d.x.index)
    d.var = d.var.reindex(index=d.x.columns)
    return d


@close_plots
def unsupervised2(d: DataSet) -> None:
    var = get_feature_annotations(d.x)
    grid = clustermap(
        d.x, config="z", row_colors=d.obs[d.attributes], col_colors=var, figsize=(20, 12)
    )
    grid.fig.savefig(output_dir / "unsupervised.clustermap.svg", **figkws)


def supervised(d: DataSet) -> None:

    _, stats = swarmboxenplot(
        data=d.x.join(d.obs), x="Clinical_score", y=d.x.columns, swarm=False, boxen=False
    )
    stats.to_csv(output_dir / "supervised.stats.csv")

    fig, _stats = swarmboxenplot(
        data=d.x.join(d.obs),
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
