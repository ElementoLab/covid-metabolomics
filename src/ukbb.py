#!/usr/bin/env python

"""
Analysis of NMR data of metabolites from blood serum of UK biobank participants.
"""

import sys

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imc.types import DataFrame, Series
from imc.utils import z_score
from seaborn_extensions import clustermap, swarmboxenplot

from src.config import *
from src.analysis import get_nmr_feature_annotations


cli = None


def main(cli: tp.Sequence[str] = None) -> int:
    """The main function to run the analysis."""
    x, y = get_x_y_nmr()

    plot_global_stats(x)
    feature_abundance_by_physical_properties(x)

    unsupervised(x, y, attributes, data_type="NMR")
    supervised(x, y)

    # Investigate NMR features
    get_feature_network(x)

    # Fin
    return 0


def get_healthy_range():
    """
    Get range of variables in ukbb.
    Including relative variables!
    """

    def ci(x: Series) -> tp.Tuple[float, float]:
        import scipy.stats as st

        return st.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=st.sem(x))

    q, _ = get_x_y_nmr()
    ids = pd.read_csv(metadata_dir / "feature_names_ids.csv").set_index("Biomarker_name")
    x_var = pd.read_csv(metadata_dir / "ukbb_var_id_mapping.csv").set_index("ukbb_id")
    formulas = pd.read_table(
        "https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/Nightingale_ratio_calculation.tsv"
    )
    for _, row in formulas.iterrows():
        f1 = row["Numerator_Field_ID"]
        f2 = row["Denominator_Field_ID"]
        fn1 = x_var.loc[f1]["feature"]
        fn2 = x_var.loc[f2]["feature"]
        r = q[fn1] / q[fn2]
        if row["Unit"] == "%":
            r *= 100
        n = ids.loc[row["Biomarker_name"]]["Variable"]
        q[n] = r

    # # export mean/CI to later use
    s = (
        q.apply(ci)
        .rename(index={0: "ci_lower", 1: "ci_upper"})
        .T.join(q.mean().rename("mean"))
        .join(q.std().rename("std"))
    )
    s.to_csv(metadata_dir / "ukbb.feature_range.csv")


def get_feature_names_from_ukbbid(ids: tp.List[int]) -> tp.List[tp.Tuple[str, str]]:
    import aiohttp
    import asyncio
    import io

    from bs4 import BeautifulSoup

    async def fetch(session, url):
        async with session.get(url) as response:
            if response.status != 200:
                response.raise_for_status()
            return await response.text()

    async def fetch_all(session, urls):
        tasks = []
        for url in urls:
            task = asyncio.create_task(fetch(session, url))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results

    def get_name_from_html(html: str) -> tp.Tuple[str, str]:

        soup = BeautifulSoup(html, "html.parser")
        for table in soup.find_all("table"):
            if "summary" in table.attrs:
                if table.attrs["summary"] == "Identification":
                    tables = pd.read_html(io.StringIO(table.decode()), index_col=0)
                    desc = tuple(tables[0].squeeze())
                    return desc  # type: ignore[return-value]
        raise ValueError

    async def to_run(ids: tp.List[int]) -> tp.List[tp.Tuple[str, str]]:
        base_url = "https://biobank.ndph.ox.ac.uk/showcase/field.cgi"
        id_urls = [base_url + f"?id={id_value}" for id_value in ids]

        async with aiohttp.ClientSession() as session:
            htmls = await fetch_all(session, id_urls)

        return [get_name_from_html(html) for html in htmls]

    return asyncio.run(to_run(ids))


def get_x_y_nmr(transformation="imputed") -> tp.Tuple[DataFrame, DataFrame]:
    """
    Read NMR data and its metadata annotation.
    """
    if transformation is None:
        x_file = data_dir / "ukb46898.reharm.csv.gz"
    elif transformation == "imputed":
        x_file = data_dir / "ukb46898.reharm.imputed.csv.gz"
    else:
        raise ValueError

    if not x_file.exists():
        ukbiobank_f = data_dir / "ukb46898.csv.gz"
        df = pd.read_csv(ukbiobank_f, index_col=0)
        df = df.loc[:, ~df.isnull().all(0)]
        ids = df.columns.str.extract(r"(\d+)-.*")[0].astype(int).values
        _desc = get_feature_names_from_ukbbid(ids)
        desc = pd.DataFrame(_desc, index=df.columns, columns=["name", "feature_type"])
        feat_filter = desc["feature_type"].str.contains("Genomics|QC")
        select = desc.loc[~feat_filter].index
        df = df.loc[:, select]
        df = df.loc[~df.isnull().all(1), :]

        # Reduce variables
        # # variable seems repeated, each with two differen suffixes: "-0.0", "-1.0"
        # # each individual has values only for one of them though.
        # # in the absence of more info, I'll reduce them
        x = df.T.groupby(ids[~feat_filter]).mean().T
        x_var = (
            desc.assign(ukbb_id=ids).drop_duplicates().set_index("ukbb_id").loc[x.columns.values]
        )

        x_annot = get_nmr_feature_annotations().query("measurement_type == 'absolute'")
        x_var = pd.concat([x_var.reset_index(), x_annot.reset_index()], axis=1)
        x_var.to_csv(metadata_dir / "ukbb_var_id_mapping.csv", index=False)
        x.columns = x_var["feature"]
        x.to_csv(x_file)
    x = pd.read_csv(x_file, index_col=0)

    y = (
        pd.read_table(data_dir / "covid19_results.4.21.21.txt")
        .groupby("eid")[["result", "acute"]]
        .max()
    )
    y = y.reindex(x.index)
    birth = pd.read_csv(metadata_dir / "original" / "birth_info", index_col=0)
    birth.columns = ["year", "month"]
    age = ((birth["year"] - birth["year"].max()) - birth["month"] / 12) * -1
    y = y.join(age.rename("age"))
    return x, y


def plot_global_stats(x: DataFrame) -> None:
    mean = x.mean()
    var = x.var()
    std = x.std()
    cv2 = x.std() / x.mean()
    var_names = ["Variance", "Standard deviation", "Squared coefficient of variation"]

    annot = get_nmr_feature_annotations().reindex(x.columns)
    cat = annot["group"].astype(pd.CategoricalDtype())
    cmap = sns.color_palette("tab20")

    fig, axes = plt.subplots(2, 3, figsize=(3 * 4, 4 * 2))
    for axs, (_y, name) in zip(axes.T, zip([var, std, cv2], var_names)):
        for ax in axs:
            for n, c in enumerate(cat.cat.categories):
                f = cat == c
                ax.scatter(mean.loc[f], _y.loc[f], color=cmap[n], label=c, alpha=0.5)
                ax.set(xlabel="Mean", ylabel=name)
    for ax in axes.flat:
        v = min(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot((0, v), (0, v), linestyle="--", color="grey")
    for ax in axes[1, :]:
        ax.loglog()
    fig.savefig((results_dir / "nightingale_tech").mkdir() / "stat_properties.svg", **figkws)

    # # Plot examples across the range of mean
    # n_bins = 12
    # bins = pd.cut(mean[mean <= mean.quantile(0.95)], n_bins)

    # fig, axes = plt.subplots(1, n_bins, figsize=(3 * n_bins, 3))
    # for ax, _bin in zip(axes, bins.cat.categories):
    #     sel = bins.loc[bins == _bin].index[0]
    #     sns.histplot(x.loc[:, sel], ax=ax)
    #     ax.set_title(_bin)

    # n_bins = 10
    # n = 10000
    # var = 1
    # params = np.linspace(0, mean.quantile(0.95), n_bins)
    # fig, axes = plt.subplots(1, n_bins, figsize=(3 * n_bins, 3))
    # for p, ax in zip(params, axes):
    #     d = scipy.stats.norm(p, var).rvs(n)
    #     d[d < 0] = 0
    #     sns.histplot(d, ax=ax)


def feature_abundance_by_physical_properties(x: DataFrame) -> None:
    output_dir = (results_dir / "feature_network").mkdir()
    annot = get_nmr_feature_annotations()
    # Observe
    x2 = x[[annot.index]]

    group_x = (
        x.T.join(annot)
        .groupby(["lipid_density", "lipid_size"])
        .mean()
        .mean(1)
        .to_frame("value")
        .pivot_table(index="lipid_density", columns="lipid_size")["value"]
    )

    # Plot
    fig, ax = plt.subplots()
    sns.heatmap(group_x, ax=ax, square=True, vmin=0, vmax=1, annot=True)
    fig.savefig(
        output_dir / "NMR_features.mean.dependent_on_patient_attributes.ukbb.svg",
        **figkws,
    )


def model_data(x: DataFrame, y: DataFrame):
    """
    Exponentially modified Gaussian

    https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html#scipy.stats.exponnorm
    https://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.ExGaussian
    """
    from src.analysis import get_nmr_feature_technical_robustness
    import pymc3 as pm

    n_indiv = 2_000
    sel_indiv = (x.std(1) / x.mean(1)).sort_values().tail(n_indiv).index

    n = x.shape[1]
    with pm.Model() as exg_model:
        mu = pm.Normal("mu", sigma=20, shape=n)
        sigma = pm.HalfCauchy("sigma", beta=10, testval=1.0, shape=n)
        nu = pm.HalfCauchy("nu", beta=10, testval=1.0, shape=n)
        exg = pm.ExGaussian("exg", mu=mu, sigma=sigma, nu=nu, observed=x.loc[sel_indiv])

    with exg_model:
        approx = pm.fit()
        # trace = pm.sample()  # slooow

    # pm.plot_trace(trace)

    trace = approx.sample(10_000)
    mu = pd.Series(trace["mu"].mean(0), x.columns)
    sigma = pd.Series(trace["sigma"].mean(0), x.columns)
    nu = pd.Series(trace["nu"].mean(0), x.columns)

    rob = get_nmr_feature_technical_robustness().join(sigma.rename("sigma"))

    fig, ax = plt.subplots()
    ax.scatter(rob["CV"], rob["sigma"])

    import statsmodels.api as sm

    model = sm.GLM(endog=y["age"], exog=sm.add_constant(x))
    res = model.fit().summary2().tables[1]
    res2 = res.copy()
    res2["Coef."] = model.fit_regularized(L1_wt=0).params

    from ngs_toolkit.utils import log_pvalues

    fig, axes = plt.subplots(1, 2)
    axes[0].scatter(res["Coef."], log_pvalues(res["P>|z|"]), alpha=0.5, s=2)
    axes[1].scatter(res2["Coef."], log_pvalues(res2["P>|z|"]), alpha=0.5, s=2)

    axes[0].set(xlim=(-3, 30))
    axes[1].set(xlim=(-3, 30))


def impute_x(x: DataFrame, method="factorization", save: bool = True) -> DataFrame:
    from fancyimpute import MatrixFactorization, KNN

    x_file = data_dir / "ukb46898.reharm.imputed.csv.gz"

    if not x_file.exists():
        missing = (x.isnull().values.sum() / x.size) * 100
        print(f"Dataset has {missing:.3f}% missing data.")  # 0.050%
        if method == "factorization":
            model = MatrixFactorization(learning_rate=0.01, epochs=500)
        elif method == "knn":
            model = KNN(15)
        x_imp = pd.DataFrame(
            model.fit_transform(x),
            index=x.index,
            columns=x.columns,
        )
        if save:
            x_imp.clip(lower=0).to_csv(x_file)
    x_imp = pd.read_csv(x_file, index_col=0)

    return x_imp


def unsupervised(
    x: DataFrame,
    y: DataFrame,
    attributes: tp.Sequence[str] = None,
    data_type: str = "NMR",
) -> None:
    """
    Unsupervised analysis of data using sample/feature correlations and
    dimentionality reduction and their visualization dependent on sample attributes.
    """
    from src.analysis import plot_projection

    if attributes is None:
        attributes = list()

    output_dir = (results_dir / f"unsupervised_{data_type}_ukbb").mkdir()

    annot = get_nmr_feature_annotations()
    annot = annot.drop(
        ["abbreviation", "name", "description", "subgroup", "measurement_type"], axis=1
    )

    n_indiv = 20_000
    sel_indiv = (x.std(1) / x.mean(1)).sort_values().tail(n_indiv).index

    ## Clustermaps
    for c in ["abs", "z"]:
        grid = clustermap(
            x.loc[sel_indiv],
            col_colors=annot,
            config=c,
            rasterized=True,
        )
        grid.savefig(
            output_dir / f"unsupervised.clustering.clustermap.top_{n_indiv}_indiv.{c}.svg",
            **figkws,
        )
    kws = dict(
        cmap="RdBu_r",
        rasterized=True,
        cbar_kws=dict(label="Pearson correlation"),
        xticklabels=False,
        yticklabels=False,
    )
    grid = clustermap(z_score(x).corr(), center=0, **kws, row_colors=annot)
    grid.savefig(
        output_dir / "unsupervised.correlation_variable.clustermap.svg",
        **figkws,
    )

    n_indiv = 10_000
    sel_indiv = (x.std(1) / x.mean(1)).sort_values().tail(n_indiv).index

    grid = clustermap(z_score(x.loc[sel_indiv]).T.corr(), **kws)
    grid.savefig(
        output_dir / "unsupervised.correlation_samples.clustermap.svg",
        **figkws,
    )


def supervised(x, y):
    import pingouin as pg

    output_dir = results_dir / "supervised"

    def rank_genes_groups_df(adata, key="rank_genes_groups"):
        # create a data frame with columns from .uns['rank_genes_groups'] (eg. names,
        # logfoldchanges, pvals).
        # Ideally, the list of columns should be consistent between methods
        # but 'logreg' does not return logfoldchanges for example

        dd = []
        groupby = adata.uns["rank_genes_groups"]["params"]["groupby"]
        for group in adata.obs[groupby].cat.categories:
            cols = []
            # inner loop to make data frame by concatenating the columns per group
            for col in adata.uns[key].keys():
                if col != "params":
                    cols.append(pd.DataFrame(adata.uns[key][col][group], columns=[col]))

            df = pd.concat(cols, axis=1)
            df["group"] = group
            dd.append(df)

        # concatenate the individual group data frames into one long data frame
        rgg = pd.concat(dd)
        rgg["group"] = rgg["group"].astype("category")
        return rgg.set_index("group")

    #
    a = AnnData(x.values, obs=y, var=x.columns.to_frame())
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    sc.tl.diffmap(a)

    # sc.pl.pca(a)
    # sc.pl.umap(a)
    # sc.pl.diffmap(a)

    # compare results between cohorts
    a.obs["result"] = a.obs["result"].astype(str).astype(pd.CategoricalDtype())
    sc.tl.rank_genes_groups(a, "result")
    res = rank_genes_groups_df(a).loc["1.0"].set_index("names").sort_values("logfoldchanges")

    stats = pd.read_csv(
        results_dir / "supervised" / "supervised.alive.all_variables.stats.csv",
        index_col=0,
    )
    change = stats["hedges_g"].rename("alive") * -1

    p = res.join(change)

    fig, ax = plt.subplots()
    ax.scatter(p["alive"], p["logfoldchanges"])

    # Compare also using exact same test
    p = x.join(y["result"]).dropna()
    s = list()
    for var in tqdm(x.columns):
        s.append(pg.mwu(p["result"], p[var]).rename_axis(var))
    stats = pd.concat(s)
    stats.to_csv(
        output_dir / "supervised.ukbb.result.all_variables.stats.csv",
        index=False,
    )


def get_feature_network(x: DataFrame) -> DataFrame:
    # import networkx as nx
    from imc.graphics import rasterize_scanpy

    output_dir = (results_dir / "feature_network").mkdir()

    # With scanpy
    annot = get_nmr_feature_annotations()
    annot = annot.drop(["abbreviation", "name", "description", "subgroup"], axis=1)

    xx = z_score(x)

    annott = annot.loc[xx.columns, (annot.nunique() > 1) & ~annot.columns.str.contains("_")]

    stats = pd.read_csv(
        results_dir / "supervised" / "supervised.alive.all_variables.stats.csv",
    )
    change = stats.set_index("Variable")["hedges_g"].rename("alive") * -1

    a = AnnData(xx.T, obs=annott.join(change))
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X")
    sc.tl.umap(a, gamma=1)
    sc.tl.leiden(a)

    feats = annott.columns.tolist() + ["leiden", "alive"]
    fig, ax = plt.subplots(len(feats), 1, figsize=(4, len(feats) * 4), sharex=True, sharey=True)
    group_cmap = tab40(range(a.obs["group"].nunique()))[:, :3]
    size_cmap = sns.color_palette("inferno", a.obs["lipid_size"].nunique())
    density_cmap = sns.color_palette("inferno", a.obs["lipid_density"].nunique())
    cmaps = [group_cmap.tolist(), "Paired"] + [density_cmap, size_cmap] + ["tab10"] + ["coolwarm"]
    for ax, feat, cmap in zip(fig.axes, feats, cmaps):
        p = dict(cmap=cmap) if a.obs[feat].dtype.name.startswith("float") else dict(palette=cmap)
        sc.pl.umap(a, color=feat, **p, edges=True, ax=ax, show=False, s=50, alpha=0.5)
    for ax in fig.axes:
        ax.set(xlabel="", ylabel="")
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "feature_annotation.network.scanpy.ukbb.svg", **figkws)

    # Visualize as heatmaps as well
    grid = clustermap(
        z_score(xx),
        metric="cosine",
        cmap="coolwarm",
        center=0,
        col_colors=annott.join(a.obs["leiden"]),
        rasterized=True,
        robust=True,
    )
    ax = grid.ax_heatmap
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
    grid.savefig(output_dir / "feature_annotation.network.scanpy.clustermap.ukbb.svg", **figkws)

    corr = z_score(xx).corr()
    grid = clustermap(
        corr,
        metric="euclidean",
        cmap="coolwarm",
        center=0,
        row_colors=annott.join(a.obs["leiden"]),
        rasterized=True,
    )
    ax = grid.ax_heatmap
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
    grid.savefig(
        output_dir / "feature_annotation.network.scanpy.clustermap.symmetric.ukbb.svg",
        **figkws,
    )

    # # hack to get the side colors:
    attrs = ["metagroup", "group", "lipid_density", "lipid_size"]
    gcmap = matplotlib.colors.ListedColormap(colors=group_cmap, name="group")
    cmaps = ["Paired", gcmap, "inferno", "inferno"]
    p = annott.join(a.obs["leiden"])[attrs]
    p["metagroup"] = pd.Categorical(
        p["metagroup"]
    )  # , categories=p['metagroup'].value_counts().index)
    p["group"] = pd.Categorical(p["group"])  # , categories=p['group'].value_counts().index)
    p = p.iloc[grid.dendrogram_row.reordered_ind]
    fig, axes = plt.subplots(1, p.shape[1], figsize=(5, 10))
    for ax, at, cmap in zip(axes, p.columns, cmaps):
        sns.heatmap(
            p[at].cat.codes.to_frame(at).replace(-1, np.nan),
            cmap=cmap,
            ax=ax,
            yticklabels=False,
            rasterized=True,
            vmin=0,
        )
        fig.axes[-1].set_yticklabels(p[at].cat.categories)
    fig.savefig(
        output_dir / "feature_annotation.network.scanpy.clustermap.symmetric.ukbb.annotations.svg",
        **figkws,
    )

    # Per group only
    feats = annott.columns.tolist()[2:] + ["leiden", "alive"]
    obs = annott.query("group == 'Lipoprotein subclasses'").join(change)
    embeddings = ["draw_graph_fa", "umap", "tsne", "pymde", "pymde_alt"]
    n, m = len(feats), len(embeddings)
    fig, axes = plt.subplots(n, m, figsize=(4 * m, 4 * n), sharex="col", sharey="col")
    a = AnnData(xx.T.loc[obs.index], obs=obs)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X")
    sc.tl.umap(a, gamma=1)
    sc.tl.draw_graph(a)
    sc.tl.tsne(a, use_rep="X")
    sc.tl.leiden(a)
    PyMDE().fit_anndata(a).fit_anndata(a, "alternate")

    size_cmap = sns.color_palette("inferno", a.obs["lipid_size"].nunique())
    density_cmap = sns.color_palette("inferno", a.obs["lipid_density"].nunique())
    cmaps = [density_cmap, size_cmap] + ["tab10"] + ["coolwarm"]
    for ax, feat, cmap in zip(axes, feats, cmaps):
        p = dict(cmap=cmap) if a.obs[feat].dtype.name.startswith("float") else dict(palette=cmap)
        for j, embedding in enumerate(embeddings):
            sc.pl.embedding(
                a,
                basis=embedding,
                color=feat,
                **p,
                edges=True,
                ax=ax[j],
                show=False,
                s=50,
                alpha=0.5,
            )
    for ax in axes.flat:
        ax.set(xlabel="", ylabel="")
    fig.savefig(
        output_dir / "feature_annotation.network.scanpy.only_lipoproteins.ukbb.svg",
        **figkws,
    )

    import ringity
    import networkx as nx

    G = nx.from_scipy_sparse_matrix(a.obsp["connectivities"])
    dgm = ringity.diagram(G)

    fig = plt.figure(constrained_layout=False)
    fig.suptitle(f"Ringity score: {dgm.score:.3f}")
    gs = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48, wspace=0.05)
    ax1 = fig.add_subplot(gs[:-1, :])
    ax2 = fig.add_subplot(gs[-1, :-1])
    ax3 = fig.add_subplot(gs[-1, -1])
    ringity.plots.plot_nx(G, ax=ax1)
    ringity.plots.plot_seq(dgm, ax=ax2)
    ringity.plots.plot_dgm(dgm, ax=ax3)
    fig.savefig(
        output_dir / "feature_annotation.network.scanpy.ringity_analysis.ukbb.svg",
        **figkws,
    )

    # TODO: get empirical p-value
    # # https://gist.github.com/gotgenes/2770023


def get_signatures() -> DataFrame:
    """
    Extract sigantures of progression (day 7 vs admission),
    and remission (discharge vs day 7)
    """
    from urlpath import URL

    root = URL("https://www.medrxiv.org/highwire/filestream/")
    url = root / "88249/field_highwire_adjunct_files/1/2020.07.02.20143685-2.xlsx"

    df = pd.read_excel(url, index_col=0)
    annot = get_nmr_feature_annotations()
    df = annot.reset_index().set_index("abbreviation").join(df).reset_index().set_index("feature")

    sigs = pd.DataFrame(index=df.index)
    sigs["future_severe_pneumonia"] = df["Beta"]

    df = pd.read_csv(metadata_dir / "infectious_disease_score.csv", index_col=0)
    df = annot.reset_index().set_index("abbreviation").join(df).reset_index().set_index("feature")
    sigs["future_infectious_disease"] = df["infectious_disease_score_weight"]

    return sigs.sort_index()


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
