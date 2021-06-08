import typing as tp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.formula.api as smf

from imc.types import DataFrame, Figure
from imc.utils import z_score
from seaborn_extensions import clustermap, swarmboxenplot

from src.config import results_dir, figkws

__all__ = [
    "unsupervised",
    "get_explanatory_variables",
    "overlay_individuals_over_global",
    "plot_projection",
]


def unsupervised(
    x: DataFrame,
    y: DataFrame,
    var: DataFrame = None,
    attributes: tp.Sequence[str] = None,
    data_type: str = "NMR",
    suffix: str = "",
) -> None:
    """
    Unsupervised analysis of data using sample/feature correlations and
    dimentionality reduction and their visualization dependent on sample attributes.
    """
    from sklearn.decomposition import PCA, NMF
    from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding
    from umap import UMAP
    from src.models import PyMDE, DiffMap

    if attributes is None:
        attributes = list()

    output_dir = (results_dir / f"unsupervised_{data_type}{suffix}").mkdir()
    output_prefix = output_dir / f"unsupervised."

    ## Clustermaps
    for c in ["abs", "z"]:
        grid = clustermap(
            x,
            row_colors=y[attributes],
            col_colors=var,
            config=c,
            rasterized=True,
        )
        grid.savefig(output_prefix + f"clustering.clustermap.{c}.svg", **figkws)
        plt.close(grid.fig)
    kws = dict(
        cmap="RdBu_r",
        rasterized=True,
        cbar_kws=dict(label="Pearson correlation"),
        xticklabels=False,
        yticklabels=False,
    )
    grid = clustermap(z_score(x).corr(), center=0, **kws, row_colors=var)
    grid.savefig(output_prefix + "correlation_variable.clustermap.svg", **figkws)
    plt.close(grid.fig)

    grid = clustermap(z_score(x).T.corr(), **kws, row_colors=y[attributes])
    grid.savefig(output_prefix + "correlation_samples.clustermap.svg", **figkws)
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

            fig = plot_projection(res, y, factors=attributes, algo_name=name, **pkwargs)
            fig.savefig(output_prefix + f"dimres.{name}.{label}svg", **figkws)
            plt.close(fig)


def get_explanatory_variables(
    x, y, data_type: str, suffix: str = "", attributes: tp.Sequence[str] = []
) -> None:
    """
    Find variables explaining the latent space discovered unsupervisedly.
    """
    from sklearn.manifold import SpectralEmbedding
    import scipy
    from imc.utils import minmax_scale

    output_dir = (results_dir / f"unsupervised_{data_type}{suffix}").mkdir()
    output_prefix = output_dir / "unsupervised.variable_contribution_SpectralEmbedding."

    xz = z_score(x)

    res = pd.DataFrame(
        SpectralEmbedding().fit_transform(xz),
        index=x.index,
        columns=["SE1", "SE2"],
    )
    # res.to_csv(output_prefix + "sample_positions.csv")
    # Get order of variables along axes
    feat_res = (
        res[["SE1"]]
        .join((xz.loc[res["SE2"] < 0]))
        .corr()
        .loc[x.columns, "SE1"]
        .rename_axis(index="feature")
        .to_frame()
    )
    feat_res = feat_res.join(
        res[["SE2"]]
        .join((xz.loc[res["SE1"] > 0]))
        .corr()
        .loc[x.columns, "SE2"]
        .rename_axis(index="feature")
    )

    lat = minmax_scale(feat_res)
    feat_res["joint"] = minmax_scale(lat["SE1"]) * (lat["SE2"])
    feat_res.to_csv(output_prefix + "correlation.variable_ordering.csv")

    # Get order of samples based on variable order
    res["joint"] = xz.T.join(feat_res["joint"]).corr()["joint"].drop("joint")
    res.to_csv(output_prefix + "correlation.sample_ordering.csv")

    for text, label in [(False, "."), (True, ".with_text.")]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        # # plot sample scatter
        ax.scatter(res["SE1"], res["SE2"], c=res["joint"], cmap="RdBu_r")
        # # plot variables as vectors
        cmap = plt.get_cmap("inferno")
        vmin, vmax = feat_res["joint"].apply([min, max])
        for i in feat_res.index:
            c = feat_res.loc[i, "joint"]
            draw = c > 0.055
            ax.plot(
                (0, feat_res.loc[i, "SE1"] / 10),
                (0, feat_res.loc[i, "SE2"] / 10),
                c=cmap((c - vmin) / (vmax - vmin)),
                linewidth=0.5,
                alpha=0.5 if draw else 0.25,
            )
            if text and draw:
                ha = "left" if feat_res.loc[i, "SE1"] > 0 else "right"
                va = "bottom" if feat_res.loc[i, "SE2"] > 0 else "top"
                ax.text(feat_res.loc[i, "SE1"] / 10, feat_res.loc[i, "SE2"] / 10, s=i, ha=ha, va=va)
                ax.scatter(
                    feat_res.loc[i, "SE1"] / 10, feat_res.loc[i, "SE2"] / 10, s=1, color="green"
                )
        fig.savefig(output_prefix + f"correlation.scatter_vectors{label}svg", **figkws)

    # Heatmap ordered
    sample_order = res.sort_values("joint").index
    var_order = feat_res.sort_values("joint").index
    lx = z_score(x).loc[sample_order, var_order]
    # # apply some smoothing
    lxs = pd.DataFrame(
        scipy.ndimage.gaussian_filter(lx, 1, mode="mirror"),
        lx.index,
        lx.columns,
    )
    for df, label in [(lx, ""), (lxs, ".smoothed")]:
        grid = clustermap(
            df,
            col_cluster=False,
            row_cluster=False,
            center=0,
            cmap="RdBu_r",
            robust=True,
            row_colors=y[attributes].join(res),
            col_colors=feat_res,
            figsize=(16, 6),
            rasterized=True,
        )
        grid.savefig(
            output_prefix + f"ordered.clustermap{label}.svg",
            **figkws,
        )
        if "palettes" in locals():
            fig = plot_attribute_heatmap(
                y.reindex(sample_order), attributes, palettes, cmaps, figsize=(16, 6)
            )
            fig.savefig(
                output_prefix + f"ordered.clustermap{label}.colors.svg",
                **figkws,
            )

    # Plot variable values along space
    l = np.round(np.arange(-1, 1, 0.1), 1)
    p = pd.DataFrame(index=l, columns=var_order)
    for col in var_order:
        xn = res.loc[sample_order, "joint"]
        f = scipy.interpolate.interp1d(xn, xz.loc[sample_order, col], fill_value="extrapolate")
        p[col] = f(l)

    fig, ax = plt.subplots(figsize=(18, 2))
    sns.heatmap(p.loc[p.index >= -0.8].iloc[:, 1:], center=0, cmap="RdBu_r", ax=ax, robust=True)
    ax.set(ylabel="Pseudotime")
    fig.savefig(output_prefix + "ordered.interpolated.heatmap.svg", **figkws)

    # Compare with simple aggregation of severity score
    p = z_score(xz.groupby(y["WHO_score_sample"]).mean())
    fig, ax = plt.subplots(figsize=(18, 2))
    sns.heatmap(p[var_order].iloc[:, 1:], center=0, cmap="RdBu_r", ax=ax, robust=True)
    ax.set(ylabel="WHO_score_sample")
    fig.savefig(output_dir / "mean_per_WHO_score_sample.ordered.heatmap.svg", **figkws)


def overlay_individuals_over_global(
    x: DataFrame, y: DataFrame, data_type: str, suffix: str = ""
) -> None:
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
    from sklearn.decomposition import PCA, NMF
    from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding
    from umap import UMAP
    from src.models import PyMDE, DiffMap
    from scipy.spatial.distance import pdist, squareform
    from scipy import interpolate
    from statsmodels.nonparametric.smoothers_lowess import lowess

    output_dir = (results_dir / f"unsupervised_{data_type}{suffix}").mkdir()

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

        dists = pd.DataFrame(squareform(pdist(res)), index=res.index, columns=res.index)

        patient_timepoints = y.groupby("patient_code")["accession"].nunique()
        patients = patient_timepoints[patient_timepoints > 1].index

        # Metrics to calculate:
        # # Total distance "run" over time
        # # Overall direction (axis1 difference END - START)
        _vector_field = list()
        _metrics = list()
        for patient in patients:
            y2 = y.loc[y["patient_code"] == patient].sort_values(["date_sample"])
            last = y2.iloc[-1].name
            first = y2.iloc[0].name
            # res.loc[y2.index].diff().abs().sum()

            for s1, s2 in zip(y2.index[:-1], y2.index[1:]):
                _vector_field.append([*res.loc[s1]] + [*res.loc[s2] - res.loc[s1]])

            _dists = pd.Series(np.diag(dists.loc[y2.index[:-1], y2.index[1:]]))

            _metrics.append(
                pd.Series(
                    dict(
                        # step_distance=res.loc[y2.index].diff(),
                        # step_time=y2['date_sample'].diff(),
                        n_timepoints=y2.shape[0],
                        total_distance=_dists.sum(),
                        dislocation=dists.loc[first, last],
                        timedelta=y2.loc[last, "date_sample"] - y2.loc[first, "date_sample"],
                    ),
                    name=patient,
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
            y2 = y.loc[y["patient_code"] == patient].sort_values(["date_sample"])

            color = colors[i]
            seve_color = palettes["WHO_score_sample"][y2["WHO_score_sample"].cat.codes[0]]
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
            / f"unsupervised.{name}.patient_walk_in_space.scatter_arrow.P23_P24_detail.zoom.svg",
            **figkws,
        )
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 4))
        axes[0].scatter(metrics["time_days"], metrics["total_distance"])
        axes[0].set(xlabel="Course (days)", ylabel="Distance (abs)")

        sns.swarmplot(x=metrics["velo_dir"], ax=axes[1])
        axes[1].set(xlabel="Overall velocity (distance/day)")

        axes[2].scatter(metrics["time_days"], metrics["dislocation"])
        axes[2].set(xlabel="Course (days)", ylabel="Total dislocation (end - start)")

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
            output_dir / f"unsupervised.{name}.patient_walk_in_space.metrics.svg",
            **figkws,
        )
        plt.close(fig)

        # Reconstruct vector field
        fig, axes = plt.subplots(1, 2, figsize=(2 * 6, 4))
        axes[0].scatter(*res.values.T, alpha=0.25, color="grey")
        axes[0].quiver(*np.asarray(_vector_field).T, color=sns.color_palette("tab10")[0])
        axes[0].set(title="Original")
        axes[1].set(title="Interpolated")

        m = abs(vf[:, 0:2].max())
        xx = np.linspace(-m, m, 100)
        yy = np.linspace(-m, m, 100)
        xx, yy = np.meshgrid(xx, yy)
        u_interp = interpolate.griddata(vf[:, 0:2], vf[:, 2], (xx, yy), method="cubic")
        v_interp = interpolate.griddata(vf[:, 0:2], vf[:, 3], (xx, yy), method="cubic")
        axes[1].scatter(*res.values.T, alpha=0.25, color="grey")
        axes[1].quiver(*np.asarray(_vector_field).T, color=sns.color_palette("tab10")[0])
        axes[1].quiver(xx, yy, u_interp, v_interp)
        axes[1].set(xlim=axes[0].get_xlim(), ylim=axes[0].get_ylim())
        fig.savefig(
            output_dir / f"unsupervised.{name}.patient_walk_in_space.quiver.svg",
            **figkws,
        )
        plt.close(fig)

    # Consensus
    joint_metrics = pd.concat(_joint_metrics)
    joint_metrics.to_csv(output_dir / "unsupervised.all_methods.patient_walk_in_space.metrics.csv")

    joint_metrics = pd.read_csv(
        output_dir / "unsupervised.all_methods.patient_walk_in_space.metrics.csv",
        index_col=0,
    )
    joint_metricsz = (
        joint_metrics.groupby("method")[["total_distance", "dislocation", "velo", "velo_dir"]]
        .apply(z_score)
        .join(joint_metrics.groupby(level=0)[["n_timepoints", "time_days"]].apply(np.mean))
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
        output_dir / "unsupervised.mean_methods.patient_walk_in_space.metrics.svg",
        **figkws,
    )

    # See what velocity is related with
    _stats = list()
    for attribute in [a for a in attributes if a in palettes]:
        df = (
            joint_metricsz.join(y.set_index("patient_code")[[attribute]]).dropna().drop_duplicates()
        )

        fig, _ = swarmboxenplot(data=df, x=attribute, y="velo")
        fig.savefig(
            output_dir
            / f"unsupervised.mean_methods.patient_walk_in_space.velocity_vs_{attribute}.svg",
            **figkws,
        )

        if not pg.homoscedasticity(data=df, dv="velo", group=attribute)["equal_var"].squeeze():
            continue
        if pg.anova(data=df, dv="velo", between=attribute)["p-unc"].squeeze() >= 0.05:
            # continue
            pass
        _stats.append(
            pg.pairwise_tukey(data=df, dv="velo", between=attribute).assign(attribute=attribute)
        )

    stats = pd.concat(_stats)
    stats.pivot_table(index="A", columns="B", values="hedges")
    stats.pivot_table(index="A", columns="B", values="diff")

    stats = stats.query("attribute == 'WHO_score_patient'")


def plot_projection(
    x_df: DataFrame,
    y_df: DataFrame,
    factors: tp.Sequence[str],
    n_dims: int = 4,
    algo_name: str = "PCA",
    fit_lowess: bool = False,
) -> Figure:
    from seaborn_extensions.annotated_clustermap import to_color_series
    from seaborn_extensions.annotated_clustermap import is_numeric
    from statsmodels.nonparametric.smoothers_lowess import lowess

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
                colors = to_color_series(y_df[factor].dropna(), palettes.get(factor))
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
                    c=y_df[factor].astype(float),
                    cmap=cmaps.get(factor),
                )
                if pc == 0:
                    bb = ax.get_position()
                    cax = fig.add_axes((bb.xmax, bb.ymin, bb.width * 0.05, bb.height))
                    _ = fig.colorbar(m, label=factor, cax=cax)
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
