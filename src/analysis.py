#!/usr/bin/env python

"""
Analysis of Olink data from COVID-19 patients.
"""

import sys, io, argparse
from typing import Sequence, Tuple
from functools import partial

from tqdm import tqdm
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import sklearn  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.decomposition import PCA, NMF  # type: ignore
from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding  # type: ignore
from umap import UMAP  # type: ignore
import scipy
import pingouin as pg  # type: ignore
import statsmodels.api as sm  # type: ignore
import statsmodels.formula.api as smf  # type: ignore
import pymde  # type: ignore
from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore

from imc.types import Path, DataFrame  # type: ignore
from imc.utils import z_score  # type: ignore[import]
from seaborn_extensions import clustermap, swarmboxenplot, volcano_plot  # type: ignore

from src.config import *


cli = None


def main(cli: Sequence[str] = None) -> int:
    args = get_parser().parse_args(cli)

    x, y = get_x_y()
    feature_annotations = get_feature_annotations(x)

    unsupervised(x, y, attributes)

    get_explanatory_variables(x, y)

    overlay_individuals_over_global(x, y)

    supervised(x, y, [a for a in attributes if a in palettes])

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
    y["bmi"] = (
        y["bmi"].str.replace("\xa0", "").replace("30-45", "30.45")
    ).astype(float)
    y["obesity"] = pd.Categorical(
        y["obesity"].replace("overwheight", "overweight"),
        ordered=True,
        categories=["nonobese", "overweight", "obese"],
    )
    y["underlying_pulm_disease"] = pd.Categorical(
        y["Underlying_Pulm_disease"].replace(
            {"no": False, "yes": True, "Yes": True}
        )
    )
    y["hospitalized"] = pd.Categorical(
        y["hospitalized"].replace({"no": False, "yes": True})
    )
    y["patient_group"] = pd.Categorical(
        y["WHO_classification"].replace("mid", "mild"),
        ordered=True,
        categories=["uninfected", "low", "mild", "moderate", "severe"],
    )
    y["WHO_score_patient"] = pd.Categorical(
        y["WHO_score__based_on_the_entire_course_"].astype(pd.Int64Dtype()),
        ordered=True,
    )
    y["WHO_score_sample"] = pd.Categorical(
        y["WHO_classification_at_the_specific_day"].astype(pd.Int64Dtype()),
        ordered=True,
    )
    y = y.drop(
        [
            "WHO_classification",
            "WHO_classification_at_the_specific_day",
            "WHO_score__based_on_the_entire_course_",
        ],
        axis=1,
    )

    y["alive"] = pd.Categorical(
        y["alive"], ordered=True, categories=["alive", "dead"]
    )
    y["date_sample"] = pd.to_datetime(y["date_sample"])
    y["patient_code"] = y["patient_code"].astype(pd.Int64Dtype())

    # reorder columns
    y = y.reindex(
        columns=(
            y.columns[y.columns.tolist().index("date_sample") :].tolist()
            + y.columns[: y.columns.tolist().index("date_sample")].tolist()
        )
    )

    return x, y


def get_feature_annotations(x):
    global feature_annotation

    # Lipids
    # # categories
    lipids = ["P", "L", "PL", "C", "CE", "FC", "TG"]

    # # size
    "XS",
    "S",
    "M",
    "L",
    "XL",
    "XXL",

    # # ...?
    "VLDL",
    "LDL",
    "HDL",

    # aminoacids
    aas = [
        "Ala",
        "Gln",
        "Gly",
        "His",
        "Total_BCAA",
        "Ile",
        "Leu",
        "Val",
        "Phe",
        "Tyr",
    ]

    # Sugars
    sugars = [
        "Glucose",
        "Lactate",
        "Pyruvate",
        "Citrate",
        "bOHbutyrate",
        "Acetate",
        "Acetoacetate",
        "Acetone",
        "Creatinine",
        "Albumin",
        "GlycA",
    ]

    # Type of measurement/transformation
    "_pct"

    #
    feature_annotation = pd.DataFrame(
        dict(
            lipid=x.columns.str.contains("|".join([f"_{x}" for x in lipids])),
            # XS_lipid=x.columns.str.contains("|".join([f"_{x}" for x in lipids])),
            aminocid=x.columns.str.contains("|".join(aas)),
            sugars=x.columns.str.contains("|".join(sugars)),
        ),
        index=x.columns,
    )
    return feature_annotation


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
    grid = clustermap(
        z_score(x).corr(), center=0, **kws, row_colors=feature_annotation
    )
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
        (DiffMap, dict(n_dims=1), dict()),
        (PyMDE, dict(n_dims=1), dict()),
        (SpectralEmbedding, dict(n_dims=1), dict()),
    ][::-1]:
        # model, pkwargs, mkwargs = (PyMDE, dict(), dict())
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


def diffusion(x, y) -> None:
    from anndata import AnnData
    import scanpy as sc

    a = AnnData(x, obs=y)
    sc.pp.scale(a)
    sc.pp.neighbors(a, use_rep="X")
    sc.tl.diffmap(a)
    a.uns["iroot"] = np.flatnonzero(a.obs["WHO_score_patient"] == 0)[0]
    sc.tl.dpt(a)
    # fix for https://github.com/theislab/scanpy/issues/409:
    a.obs["dpt_order_indices"] = a.obs["dpt_pseudotime"].argsort()
    a.uns["dpt_changepoints"] = np.ones(a.obs["dpt_order_indices"].shape[0] - 1)


def get_explanatory_variables(x, y) -> None:
    """
    Find variables explaining the latent space discovered unsupervisedly.
    """

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
    lx = z_score(x).loc[sample_order, var_order]
    # # apply some smoothing
    lxs = pd.DataFrame(
        scipy.ndimage.gaussian_filter(lx, 3, mode="mirror"),
        lx.index,
        lx.columns,
    )
    grid = clustermap(
        lxs,
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


def overlay_individuals_over_global(x, y) -> None:
    """
    Find variables explaining the latent space discovered unsupervisedly.
    """

    """
    TODO: idea:
        Derive the vector field of the latent space(s):
         - get sparse vector field based on observed patient movement
            (x, y) <- coordinates of vector origin
            (u, v) <- direction 
         - plt.quiver(x, y, u, v)
         - smooth or interpolate sparse field -> general field
    """
    from scipy.spatial.distance import euclidean, pdist, squareform
    from scipy import interpolate

    output_dir = (results_dir / "unsupervised").mkdir()

    _joint_metrics = list()
    for model, pkwargs, mkwargs in [
        (PCA, dict(), dict()),
        # (NMF, dict(), dict()),
        (MDS, dict(n_dims=1), dict()),
        (TSNE, dict(n_dims=1), dict()),
        (Isomap, dict(n_dims=1), dict()),
        (UMAP, dict(n_dims=1), dict(random_state=0)),
        (DiffMap, dict(n_dims=1), dict()),
        (PyMDE, dict(n_dims=1), dict()),
        (SpectralEmbedding, dict(n_dims=1), dict()),
    ][::-1]:
        name = str(model).split(".")[-1].split("'")[0]
        model_inst = model(**mkwargs)

        res = (
            pd.DataFrame(
                model_inst.fit_transform(z_score(x))[:, :2],
                index=x.index,
                columns=["SE1", "SE2"],
            )
            * 1e3
        )

        dists = pd.DataFrame(
            squareform(pdist(res)), index=res.index, columns=res.index
        )

        patient_timepoints = y.groupby("patient_code")["accession"].nunique()
        patients = patient_timepoints[patient_timepoints > 1].index

        # Metrics to calculate:
        # # Total distance "run" over time
        # # Overall direction (axis1 difference END - START)
        _vector_field = list()
        _metrics = list()
        for patient in patients:
            y2 = y.loc[y["patient_code"] == patient].sort_values(
                ["date_sample"]
            )
            last = y2.iloc[-1].name
            first = y2.iloc[0].name
            # res.loc[y2.index].diff().abs().sum()

            for s1, s2 in zip(y2.index[:-1], y2.index[1:]):
                _vector_field.append(
                    [*res.loc[s1]] + [*res.loc[s2] - res.loc[s1]]
                )

            _dists = pd.Series(np.diag(dists.loc[y2.index[:-1], y2.index[1:]]))

            _metrics.append(
                pd.Series(
                    dict(
                        # step_distance=res.loc[y2.index].diff(),
                        # step_time=y2['date_sample'].diff(),
                        n_timepoints=y2.shape[0],
                        total_distance=_dists.sum(),
                        dislocation=dists.loc[first, last],
                        timedelta=y2.loc[last, "date_sample"]
                        - y2.loc[first, "date_sample"],
                    ),
                    name=int(patient),
                )
            )
        vf = np.asarray(_vector_field)

        metrics = pd.DataFrame(_metrics)
        metrics.index.name = "patient_code"
        metrics["time_days"] = metrics["timedelta"].apply(lambda x: x.days)
        metrics["velo"] = metrics["total_distance"].abs() / metrics["time_days"]
        metrics["velo_dir"] = metrics["total_distance"] / metrics["time_days"]
        _joint_metrics.append(metrics.assign(method=name))

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.scatter(*res.values.T, alpha=0.25, color="grey")

        # # add lowess
        fit = lowess(res["SE1"], res["SE2"])
        ax.plot(*fit.T, color="black", linestyle="--")

        colors = sns.color_palette("tab20", len(patients)) + sns.color_palette(
            "Accent", len(patients)
        )
        for i, patient in enumerate(patients):
            y2 = y.loc[y["patient_code"] == patient].sort_values(
                ["date_sample"]
            )

            color = colors[i]
            seve_color = palettes["WHO_score_sample"][
                y2["WHO_score_sample"].cat.codes[0]
            ]
            outcome = y2["alive"].iloc[0]

            v = res.loc[y2.index].values
            ax.text(
                *v[0],
                s=f"P{str(patient).zfill(2)} - start",
                color=color,
                ha="center",
                va="center",
            )
            ax.text(
                *v[-1],
                s=f"P{str(patient).zfill(2)} - end: {outcome}",
                color=color,
                ha="center",
                va="center",
            )
            ax.scatter(*v[0], s=12, color=color)
            ax.scatter(*v[-1], s=12, color=color, marker="^")
            for l1, l2 in zip(v[:-1], v[1:]):
                ax.annotate(
                    "",
                    xy=l2,
                    xytext=l1,
                    arrowprops=dict(arrowstyle="->", color=color),
                )
        fig.savefig(
            output_dir
            / f"unsupervised.{name}.patient_walk_in_space.scatter_arrow.svg",
            **figkws,
        )
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 4))
        axes[0].scatter(metrics["time_days"], metrics["total_distance"])
        axes[0].set(xlabel="Course (days)", ylabel="Distance (abs)")

        sns.swarmplot(x=metrics["velo_dir"], ax=axes[1])
        axes[1].set(xlabel="Overall velocity (distance/day)")

        axes[2].scatter(metrics["time_days"], metrics["dislocation"])
        axes[2].set(
            xlabel="Course (days)", ylabel="Total dislocation (end - start)"
        )

        v = metrics["total_distance"].max()
        v -= v * 0.4
        axes[3].scatter(
            metrics["dislocation"],
            metrics["total_distance"],
            c=metrics["n_timepoints"],
        )
        axes[3].plot((-v, 0), (v, 0), linestyle="--", color="grey")
        axes[3].plot((0, v), (0, v), linestyle="--", color="grey")
        axes[3].set(xlabel="Total dislocation (end - start)", ylabel="Distance")
        fig.savefig(
            output_dir
            / f"unsupervised.{name}.patient_walk_in_space.metrics.svg",
            **figkws,
        )
        plt.close(fig)

        # Reconstruct vector field
        fig, axes = plt.subplots(1, 2, figsize=(2 * 6, 4))
        axes[0].scatter(*res.values.T, alpha=0.25, color="grey")
        axes[0].quiver(
            *np.asarray(_vector_field).T, color=sns.color_palette("tab10")[0]
        )
        axes[0].set(title="Original")
        axes[1].set(title="Interpolated")

        m = abs(vf[:, 0:2].max())
        xx = np.linspace(-m, m, 100)
        yy = np.linspace(-m, m, 100)
        xx, yy = np.meshgrid(xx, yy)
        u_interp = interpolate.griddata(
            vf[:, 0:2], vf[:, 2], (xx, yy), method="cubic"
        )
        v_interp = interpolate.griddata(
            vf[:, 0:2], vf[:, 3], (xx, yy), method="cubic"
        )
        axes[1].scatter(*res.values.T, alpha=0.25, color="grey")
        axes[1].quiver(
            *np.asarray(_vector_field).T, color=sns.color_palette("tab10")[0]
        )
        axes[1].quiver(xx, yy, u_interp, v_interp)
        axes[1].set(xlim=axes[0].get_xlim(), ylim=axes[0].get_ylim())
        fig.savefig(
            output_dir
            / f"unsupervised.{name}.patient_walk_in_space.quiver.svg",
            **figkws,
        )
        plt.close(fig)

    # Consensus
    joint_metrics = pd.concat(_joint_metrics)
    joint_metrics.to_csv(
        output_dir
        / f"unsupervised.all_methods.patient_walk_in_space.metrics.csv"
    )

    joint_metrics = pd.read_csv(
        output_dir
        / f"unsupervised.all_methods.patient_walk_in_space.metrics.csv",
        index_col=0,
    )
    joint_metricsz = (
        joint_metrics.groupby("method")[
            ["total_distance", "dislocation", "velo", "velo_dir"]
        ]
        .apply(z_score)
        .join(
            joint_metrics.groupby(level=0)[["n_timepoints", "time_days"]].apply(
                np.mean
            )
        )
        .groupby(level=0)
        .mean()
    )

    fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 4))
    axes[0].scatter(
        joint_metricsz["time_days"],
        joint_metricsz["total_distance"],
        c=joint_metricsz["n_timepoints"],
    )
    axes[0].set(xlabel="Course (days)", ylabel="Distance (Z-score, abs)")

    axes[1].scatter(
        np.random.random(joint_metricsz.shape[0]) / 10,
        joint_metricsz["velo_dir"],
        c=joint_metricsz["n_timepoints"],
    )
    # sns.swarmplot(x=joint_metricsz["velo_dir"], ax=axes[1])
    axes[1].set(ylabel="Overall velocity (distance/day)", xlim=(-1, 1))

    axes[2].scatter(
        joint_metricsz["time_days"],
        joint_metricsz["dislocation"],
        c=joint_metricsz["n_timepoints"],
    )
    axes[2].set(
        xlabel="Course (days)",
        ylabel="Total dislocation (Z-score, end - start)",
    )

    v = joint_metricsz["total_distance"].max()
    v -= v * 0.4
    axes[3].scatter(
        joint_metricsz["dislocation"],
        joint_metricsz["total_distance"],
        c=joint_metricsz["n_timepoints"],
    )
    axes[3].plot((-v, v), (-v, v), linestyle="--", color="grey")
    axes[3].set(
        xlabel="Total dislocation (Z-score, end - start)",
        ylabel="Distance (Z-score)",
    )
    fig.savefig(
        output_dir
        / f"unsupervised.mean_methods.patient_walk_in_space.metrics.svg",
        **figkws,
    )

    # See what velocity is related with
    _stats = list()
    for attribute in [a for a in attributes if a in palettes]:
        df = (
            joint_metricsz.join(y.set_index("patient_code")[[attribute]])
            .dropna()
            .drop_duplicates()
        )

        fig, _ = swarmboxenplot(data=df, x=attribute, y="velo")
        fig.savefig(
            output_dir
            / f"unsupervised.mean_methods.patient_walk_in_space.velocity_vs_{attribute}.svg",
            **figkws,
        )

        if (
            pg.homoscedasticity(data=df, dv="velo", group=attribute)[
                "equal_var"
            ].squeeze()
            != True
        ):
            continue
        if (
            pg.anova(data=df, dv="velo", between=attribute)["p-unc"].squeeze()
            >= 0.05
        ):
            # continue
            pass
        _stats.append(
            pg.pairwise_tukey(data=df, dv="velo", between=attribute).assign(
                attribute=attribute
            )
        )

    stats = pd.concat(_stats)
    stats.pivot_table(index="A", columns="B", values="hedges")
    stats.pivot_table(index="A", columns="B", values="diff")

    stats = stats.query("attribute == 'WHO_score_patient'")


def supervised(x, y, attributes, plot_all: bool = True) -> None:
    import statsmodels.formula.api as smf

    output_dir = (results_dir / "supervised").mkdir()

    for attribute in attributes:
        # first just get the stats
        fig, stats = swarmboxenplot(
            data=x.join(y),
            x=attribute,
            y=x.columns,
            swarm=False,
            boxen=False,
        )
        plt.close(fig)
        stats.to_csv(
            output_dir / f"supervised.{attribute}.all_variables.stats.csv",
            index=False,
        )

        if plot_all:
            fig, stats = swarmboxenplot(
                data=x.join(y),
                x=attribute,
                y=x.columns,
            )
            fig.savefig(
                output_dir
                / f"supervised.{attribute}.all_variables.swarmboxenplot.svg",
                **figkws,
            )

        # now plot top variables
        fig, s2 = swarmboxenplot(
            data=x.join(y),
            x=attribute,
            y=stats.sort_values("p-unc")["Variable"].head(20).unique(),
            plot_kws=dict(palette=palettes.get(attribute)),
        )
        fig.savefig(
            output_dir
            / f"supervised.{attribute}.top_differential.swarmboxenplot.svg",
            **figkws,
        )

    # Use also a MLM and compare to GLM
    attrs = [
        "race",
        "age",
        "sex",
        "obesity",
        "bmi",
        "hospitalized",
        "patient_group",
        "alive",
    ]
    for attribute in attrs:
        stats_f = output_dir / f"supervised.{attribute}.model_fits.csv"

        if not stats_f.exists():
            # # Mixed effect
            data = (
                z_score(x)
                .join(y)
                .dropna(subset=[attribute, "WHO_score_sample"])
            )
            data["WHO_score_sample"] = data["WHO_score_sample"].cat.codes
            _res_mlm = list()
            for feat in tqdm(x.columns):
                mdf = smf.mixedlm(
                    f"{feat} ~ {attribute} * WHO_score_sample",
                    data,
                    groups=data["patient_code"],
                ).fit()
                res = (
                    mdf.params.to_frame("coefs")
                    .join(mdf.pvalues.rename("pvalues"))
                    .assign(feature=feat)
                )
                res = res.loc[res.index.str.contains(attribute)]
                _res_mlm.append(res)
            res_mlm = pd.concat(_res_mlm)
            # res_mlm['qvalues'] = pg.multicomp(res_mlm['pvalues'].values, method="fdr_bh")[1]
            # res_mlm.loc[res_mlm.index.str.contains(":")].sort_values("pvalues")

            # # GLM
            _res_glm = list()
            for feat in tqdm(x.columns):
                mdf = smf.glm(
                    f"{feat} ~ {attribute} + WHO_score_sample", data
                ).fit()
                res = (
                    mdf.params.to_frame("coefs")
                    .join(mdf.pvalues.rename("pvalues"))
                    .assign(feature=feat)
                )
                res = res.loc[res.index.str.contains(attribute)]
                _res_glm.append(res)
            res_glm = pd.concat(_res_glm)

            res = pd.concat(
                [res_mlm.assign(model="mlm"), res_glm.assign(model="glm")]
            )
            res.to_csv(stats_f)

    all_stats = pd.concat(
        [
            pd.read_csv(
                output_dir / f"supervised.{attribute}.model_fits.csv",
                index_col=0,
            ).assign(attribute=attribute)
            for attribute in attrs
        ]
    )

    coef_mat = all_stats.reset_index().pivot_table(
        index="feature", columns="index", values="coefs"
    )

    grid = clustermap(
        coef_mat, config="abs", cmap="RdBu_r", center=0, xticklabels=True
    )

    for attribute in attrs:
        stats_f = output_dir / f"supervised.{attribute}.model_fits.csv"
        res = pd.read_csv(stats_f, index_col=0)

        res_glm = res.query("model == 'glm'")
        res_mlm = res.query("model == 'mlm'")

        cglm = res_glm.set_index("feature")["coefs"].rename("GLM")
        cmlm = res_mlm.set_index("feature")["coefs"].rename("MLM")
        c = cglm.to_frame().join(cmlm)
        pglm = res_glm.set_index("feature")["pvalues"].rename("GLM")
        pmlm = res_mlm.set_index("feature")["pvalues"].rename("MLM")
        p = pglm.to_frame().join(pmlm)
        q = p.copy()
        for col in q:
            q[col] = pg.multicomp(q[col].values, method="fdr_bh")[1]

        v = c.abs().max()
        v += v * 0.1

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot((-v, v), (-v, v), linestyle="--", color="grey", zorder=-2)
        ax.scatter(
            c["GLM"],
            c["MLM"],
            c=p.mean(1),
            cmap="Reds_r",
            vmin=0,
            vmax=1.5,
            s=5,
            alpha=0.5,
        )
        ax.set(
            title=attribute,
            xlabel=r"$\beta$   (GLM)",
            ylabel=r"$\beta$   (MLM)",
        )
        fig.savefig(
            output_dir
            / f"supervised.{attribute}.General_vs_Mixed_effect_model_comparison.scatter.svg",
            **figkws,
        )

        # now plot top variables
        if y[attribute].dtype.name in ["object", "category"]:
            fig = swarmboxenplot(
                data=x.join(y),
                x=attribute,
                y=res_mlm.sort_values("pvalues").head(20)["feature"],
                plot_kws=dict(palette=palettes.get(attribute)),
                test=False,
            )
            fig.savefig(
                output_dir
                / f"supervised.{attribute}.top_differential-Mixed_effect_models.swarmboxenplot.svg",
                **figkws,
            )
        else:
            from imc.graphics import get_grid_dims

            feats = res_mlm.sort_values("pvalues").head(20)["feature"].tolist()
            fig = get_grid_dims(feats, return_fig=True, sharex=True)
            for feat, ax in zip(feats, fig.axes):
                sns.regplot(data=x.join(y), x=attribute, y=feat, ax=ax)
                ax.set(
                    title=feat + f"; FDR = {p.loc[feat, 'MLM']:.2e}",
                    ylabel=None,
                )
            fig.savefig(
                output_dir
                / f"supervised.{attribute}.top_differential-Mixed_effect_models.regplot.svg",
                **figkws,
            )

        # Plot volcano
        data = res_mlm.rename(
            columns={
                "feature": "Variable",
                "coefs": "hedges",
                "pvalues": "p-unc",
            }
        ).assign(A="A", B="B")
        data["p-cor"] = pg.multicomp(data["p-unc"].values, method="fdr_bh")[1]
        fig = volcano_plot(stats=data.reset_index(drop=True))
        fig.savefig(
            output_dir
            / f"supervised.{attribute}.Mixed_effect_models.volcano_plot.svg",
            **figkws,
        )

        # Plot heatmap of differential only
        # # TODO: for multiclass attributes (e.g. race need to disentangle)
        f = c.abs().sort_values("MLM").tail(50).index.drop_duplicates()
        f = c.loc[f].sort_values("MLM").index.drop_duplicates()
        stats = (
            (-np.log10(p["MLM"].to_frame("-log10(p-value)")))
            .join(c["MLM"].rename("Coefficient"))
            .join((q["MLM"] < 0.05).to_frame("Significant"))
            .groupby(level=0)
            .mean()
        )

        so = score_signature(x, diff=stats["Coefficient"]).sort_values()

        grid = clustermap(
            x.loc[so.index, f],
            config="z",
            row_colors=y[attribute].to_frame().join(so),
            col_colors=stats,
            yticklabels=False,
            cbar_kws=dict(label="Z-score"),
        )
        grid.ax_heatmap.set(
            xlabel=f"Features (top 50 features for '{attribute}'"
        )
        grid.fig.savefig(
            output_dir
            / f"supervised.{attribute}.Mixed_effect_models.clustermap.top_50.svg",
            **figkws,
        )

        grid = clustermap(
            x.loc[so.index, f],
            row_cluster=False,
            col_cluster=False,
            config="z",
            row_colors=y[attribute].to_frame().join(so),
            col_colors=stats,
            yticklabels=False,
            cbar_kws=dict(label="Z-score"),
        )
        grid.ax_heatmap.set(
            xlabel=f"Features (top 50 features for '{attribute}'"
        )
        grid.fig.savefig(
            output_dir
            / f"supervised.{attribute}.Mixed_effect_models.clustermap.top_50.sorted.svg",
            **figkws,
        )


def _plot_projection(
    x_df, y_df, factors, n_dims=4, algo_name="PCA", fit_lowess: bool = False
):
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
                    y_df[factor].dropna(), palettes.get(factor)
                )
            except (TypeError, ValueError):
                colors = to_color_series(y_df[factor].dropna())
        for pc in x_df.columns[:n_dims]:
            ax = axes[i, pc]

            if fit_lowess:
                fit = lowess(x_df.loc[:, pc], x_df.loc[:, pc + 1], frac=1.0)
                ax.plot(*fit.T, color="grey", linestyle="--")

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


import pandas as pd

from imc.types import Series, DataFrame
from imc.utils import z_score


def score_signature(x: DataFrame, diff: Series):
    """
    Score samples based on signature.
    """
    # Standardize and center
    xz = z_score(x)

    # Separate up/down genes
    if diff.dtype.name in ["categorical", "object"]:
        raise NotImplementedError
        # 1.1 get vectors for up- and down regulated genes
        cond1, cond2 = diff.drop_duplicates()
        u1 = xz.loc[:, xz.columns.str.startswith(cond1)].mean(axis=1)
        u2 = xz.loc[:, xz.columns.str.startswith(cond2)].mean(axis=1)
        extremes = pd.DataFrame([u1, u2], index=[cond1, cond2]).T
        up = extremes[extremes[cond1] > extremes[cond2]].index
        down = extremes[extremes[cond1] < extremes[cond2]].index
    else:
        up = diff[diff > 0].index
        down = diff[diff < 0].index

    # 1.2 Make score
    # get sum/mean intensities in either
    # weighted by each side contribution to the signature
    # sum the value of each side
    n = xz.shape[1]
    scores = (
        -(xz[up].mean(axis=1) * (float(up.size) / n))
        + (xz[down].mean(axis=1) * (float(down.size) / n))
    ).rename("signature_score")
    return scores


if __name__ == "__main__":
    from typing import List

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
