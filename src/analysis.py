#!/usr/bin/env python

"""
Analysis of NMR data of metabolites from blood serum of COVID-19 patients.
"""

import sys
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import MDS, Isomap, TSNE, SpectralEmbedding
from umap import UMAP
import scipy
import pingouin as pg
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

from imc.types import Path, Series, DataFrame, Figure
from imc.utils import z_score
from seaborn_extensions import clustermap, swarmboxenplot, volcano_plot

from src.config import *


cli = None


def main(cli: tp.Sequence[str] = None) -> int:
    """The main function to run the analysis."""
    # args = get_parser().parse_args(cli)

    # Inspect panel regardless of feature abundance
    get_nmr_feature_technical_robustness()
    plot_nmr_technical_robustness()
    # TODO: bring here plots on feature classes

    # Our cohort
    x1, y1 = get_x_y_nmr()

    unsupervised(x1, y1, attributes, data_type="NMR")

    get_explanatory_variables(x1, y1, "NMR")

    overlay_individuals_over_global(x1, y1, data_type="NMR")

    supervised(x1, y1, [a for a in attributes if a in palettes])
    # TODO: compare with signature from UK biobank: https://www.medrxiv.org/highwire/filestream/88249/field_highwire_adjunct_files/1/2020.07.02.20143685-2.xlsx

    # Investigate NMR features
    # # feature frequency
    plot_nmr_feature_annotations()
    # # Feature-feature relationships and their classes
    get_feature_network(x1, y1)
    feature_physical_aggregate_change(x1, y1)
    feature_properties_change(x1)
    feature_properties_pseudotime(x1)

    # Flow cytometry
    x2, y2 = get_x_y_flow()
    x1, x2, y = get_matched_nmr_and_flow(x1, y1, x2, y2)

    # unsupervised(x1, y, attributes, data_type="NMR", suffix="_strict")
    # overlay_individuals_over_global(x1, y, data_type="NMR", suffix="_strict")
    unsupervised(x2, y, attributes, data_type="flow_cytometry")
    overlay_individuals_over_global(x2, y, data_type="flow_cytometry")

    # Joint data types
    # # See how variables relate to each other
    cross_data_type_predictions()
    # # Develop common latent space
    integrate_nmr_flow()
    # # Predict disease severity based on combination of data
    predict_outcomes()

    # Fin
    return 0


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    return parser


def get_x_y_nmr() -> tp.Tuple[DataFrame, DataFrame]:
    """
    Read NMR data and its metadata annotation.
    """
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
    y["bmi"] = (y["bmi"].str.replace("\xa0", "").replace("30-45", "30.45")).astype(float)
    y["obesity"] = pd.Categorical(
        y["obesity"].replace("overwheight", "overweight"),
        ordered=True,
        categories=["nonobese", "overweight", "obese"],
    )
    y["underlying_pulm_disease"] = pd.Categorical(
        y["Underlying_Pulm_disease"].replace({"no": False, "yes": True, "Yes": True})
    )
    y["hospitalized"] = pd.Categorical(
        y["hospitalized"].replace({"no": False, "yes": True})
    )
    y["intubated"] = pd.Categorical(
        y["intubated"].replace({"not intubated": False, "intubated": True})
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

    y["alive"] = pd.Categorical(y["alive"], ordered=True, categories=["alive", "dead"])
    y["date_sample"] = pd.to_datetime(y["date_sample"])
    y["patient_code"] = y["patient_code"].astype(pd.Int64Dtype())
    q = y["accession"].str.strip().str.extract(r"P20-(\d+)")[0]
    y["accession"] = "P020-" + q.str.zfill(3)
    y["days_symptoms_before_admission"] = (
        y["days_of_symptoms_preceding_admission"]
        .replace("no admission", np.nan)
        .fillna(-1)
        .astype(int)
        .astype(pd.Int64Dtype())
        .replace(-1, np.nan)
    )
    y["days_since_hospitalization"] = (
        y["days_bw_hospitalization_and_sample"]
        .replace("not admitted", np.nan)
        .fillna(-1)
        .astype(int)
        .astype(pd.Int64Dtype())
        .replace(-1, np.nan)
    )
    y["days_since_symptoms"] = (
        y["days_BW_start_of_covid_symptom_and_sample"]
        .replace(">90", 90)
        .fillna(-1)
        .astype(int)
        .astype(pd.Int64Dtype())
        .replace(-1, np.nan)
    )

    ## labs
    labs = [
        "total_bilirubin",
        "ALT",
        "AST",
        "creatinine",
        "CRP",
        "hemoglobin",
        "hematocrit",
        "LDH",
        "RDWCV",
        "MCV",
    ]
    to_repl = {"not done": np.nan, "<0.4": 0.0, ">32": 32.0}
    for col in labs:
        y[col] = y[col].replace(to_repl).astype(str).str.replace(",", ".").astype(float)

    ## treatments
    treats = ["HCQ", "Remdesivir", "tosilizumab", "Steroids"]
    for treat in treats:
        y[treat] = pd.Categorical(y[treat], ordered=True, categories=["no", "yes"])

    # reorder columns
    y = y.reindex(
        columns=(
            y.columns[y.columns.tolist().index("date_sample") :].tolist()
            + y.columns[: y.columns.tolist().index("date_sample")].tolist()
        )
    )

    return x, y


def get_nmr_feature_technical_robustness() -> DataFrame:
    """
    Measure robustness of each variable based on repeated measuremnts or
    measurements of same individual.
    """
    nightingale_rep_csv_f = metadata_dir / "nightingale_feature_robustness.csv"
    if not nightingale_rep_csv_f.exists():
        from urlpath import URL
        import pdfplumber

        nightingale_rep_f = metadata_dir / "original" / "nmrm_app2.pdf"

        if not nightingale_rep_f.exists():
            url = URL(
                "https://biobank.ndph.ox.ac.uk/showcase/showcase/docs/nmrm_app2.pdf"
            )
            req = url.get()
            with open(nightingale_rep_f, "wb") as handle:
                handle.write(req.content)

        row_endings = ["mmol/l", "g/l", "ratio", "nm", "%", "degree"]
        pdf = pdfplumber.open(nightingale_rep_f)
        _res = list()
        for page in tqdm(pdf.pages):
            # last two lines are same for every page
            lines = page.extract_text().split("\n")[:-2]
            group = lines[0]

            # find the split in rows
            row_idxs = [0]
            for i, line in enumerate(lines[1:], 1):
                if all([q in row_endings for q in line.split(" ")]):
                    row_idxs.append(i)

            for start_line, end_line in zip(row_idxs[:-1], row_idxs[1:]):
                features = lines[start_line + 1].split(" ")
                # Skip false rows which start only with floats (page 35+)
                try:
                    _ = [float(x) for x in features]
                    continue
                except ValueError:
                    pass

                units = lines[end_line].split(" ")
                perf = [
                    r.strip().split(", R : ")
                    for r in lines[start_line + 3].split("CV: ")[1:]
                ]
                res = pd.DataFrame(perf, index=features, columns=["CV", "R"]).assign(
                    group=group
                )
                res["unit"] = units
                print(res)
                _res.append(res)
        pdf.close()
        res = pd.concat(_res).rename_axis(index="feature")
        res["CV"] = res["CV"].str.replace("%", "").astype(float) / 100
        res["R"] = res["R"].astype(float)
        res["group"] = res["group"].str.replace("−", "-")

        # # extract diameter from 'group' column
        # # (otherwise 'group' column is the same as annot['subgroup'])
        diameters = res["group"].str.extract(r".* \((.*)\)")[0]
        diameters = diameters.str.extract(r"(\d+ nm)")[0].str.replace(" nm", "")
        diameters.loc[res["group"].str.contains("upwards")] += "+"
        res["diameter_nm"] = diameters
        res["subgroup"] = res["group"].str.extract(r"(.*) \(")[0]
        res["subgroup"] = res["subgroup"].fillna(res["group"])

        # # Drop duplicates but pay attention to index
        res = (
            res.drop("group", axis=1).reset_index().drop_duplicates().set_index("feature")
        )
        res.to_csv(nightingale_rep_csv_f)
    robustness = pd.read_csv(nightingale_rep_csv_f, index_col=0)
    return robustness


def plot_nmr_technical_robustness() -> None:
    annot = get_nmr_feature_annotations()
    cat = annot["group"].astype(pd.CategoricalDtype())

    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = sns.color_palette("tab20")
    for n, c in enumerate(cat.cat.categories):
        p = annot.loc[cat == c]
        ax.scatter(p["CV"], p["R"], color=cmap[n], label=c, alpha=0.5)
    ax.set(xscale="log", xlabel="CV", ylabel="R^2")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    fig.savefig(
        (results_dir / "nightingale_tech").mkdir() / "assay_robustness.svg", **figkws
    )

    # # Relate technical and biological variability
    # x, _ = get_x_y_nmr()
    # cv2 = ((x.std() / x.mean()) ** 2).rename("CV2")

    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.scatter(annot["R"] ** 6, cv2.reindex(annot.index), alpha=0.5)


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
    fig.savefig(
        (results_dir / "nightingale_tech").mkdir() / "stat_properties.svg", **figkws
    )


def compare_clinical_and_panel():

    x, y = get_x_y_nmr()

    fig, ax = plt.subplots()
    ax.scatter(y["creatinine"], x["Creatinine"])
    ax.set(title="Creatinine", xlabel="Clinical labs", ylabel="NMR panel")


def get_feature_annotations(x: DataFrame, data_type: str) -> DataFrame:
    if data_type == "NMR":
        return get_nmr_feature_annotations()
    if data_type == "flow_cytometry":
        return get_flow_feature_annotations(x)
    raise ValueError("Data type not understood. Choose one of 'NMR' or 'flow_cytometry'.")


def get_flow_feature_annotations(x):
    import json

    projects_dir = Path("~/projects/archive").expanduser()
    project_dir = projects_dir / "covid-flowcyto"

    var_annot = json.load(open(project_dir / "metadata" / "panel_variables.json"))
    feature_annotation = pd.DataFrame(
        index=x.columns, columns=var_annot.keys(), dtype=bool
    )
    for group in var_annot:
        feature_annotation.loc[
            ~feature_annotation.index.isin(var_annot[group]), group
        ] = False
    return feature_annotation


def plot_all_signatures():
    from src.ukbb import get_signatures as ukbbsigs
    from src.dierckx import get_signatures as dierckxsigs

    ukbb_sigs = ukbbsigs().sort_index()
    dierckx_sigs = dierckxsigs().sort_index()

    p = ukbb_sigs.join(dierckx_sigs).drop("future_infectious_disease", 1).dropna()

    grid = clustermap(p, col_cluster=False, center=0, cmap="RdBu_r")
    plt.close(grid.fig)


def get_nmr_feature_annotations() -> DataFrame:
    """
    Annotate NMR features with broad category and some basic physical properties.
    """
    nightingale_annotation_f = metadata_dir / "df_NG_biomarker_metadata.csv"
    if not nightingale_annotation_f.exists():
        import tempfile

        from urlpath import URL
        import pyreadr

        # Another source could the the UK Biobank, e.g.:
        # https://biobank.ndph.ox.ac.uk/showcase/showcase/docs/Nightingale_biomarker_groups.txt

        # Reading these data from the ggforestplot R package
        nightingale_repo = URL("https://github.com/NightingaleHealth/ggforestplot")
        df_name = "df_NG_biomarker_metadata.rda"
        df_url = nightingale_repo / "raw" / "master" / "data" / df_name

        tmp_f = tempfile.NamedTemporaryFile(suffix=df_url.suffix)
        with df_url.get() as req:
            with open(tmp_f.name, "wb") as handle:
                if req.ok:
                    handle.write(req.content)
        result = pyreadr.read_r(tmp_f.name)
        cols = [
            "abbreviation",
            # "alternative_names",  # <- gives problems since it is a list for each row
            "name",
            "subgroup",
            "machine_readable_name",
            "description",
            "group",
            # "unit",
        ]
        result[df_name][cols].to_csv(nightingale_annotation_f, index=False)

    # read saved
    annot = (
        pd.read_csv(nightingale_annotation_f)
        .set_index("machine_readable_name")
        .rename_axis(index="feature")
    )

    # Further annotate lipoproteins with physical properties
    densities = {
        "VLDL": "Very low",
        "LDL": "Low",
        "IDL": "Intermediate",
        "HDL": "Heavy",
    }
    sizes = {
        "XS": "Extra small",
        "S": "Small",
        "M": "Medium",
        "L": "Large",
        "XL": "Extra large",
        "XXL": "Extra extra large",
    }
    lipoprots = {
        "P": "Particle concentration",
        "size": "Particle size",
        "L": "Total lipids",
        "C": "Cholesterol",
        "TG": "Triglycerides",
        "PL": "Phospholipids",
        "CE": "Cholesteryl esters",
        "FC": "Free cholesterol",
    }
    annot["metagroup"] = annot["group"]
    annot.loc[
        annot["group"].isin(lipoprots.keys())
        | annot["group"].str.contains("Lipo|lipo|lipid")
        & (annot["group"] != "Apolipoproteins"),
        "metagroup",
    ] = "Lipid"

    annot["lipid_class"] = np.nan
    for lipid_class in lipoprots:
        annot.loc[
            annot.index.str.endswith(lipid_class)
            | annot.index.str.endswith(lipid_class + "_pct"),
            "lipid_class",
        ] = lipoprots[lipid_class]
    annot["lipid_density"] = np.nan
    for density in densities:
        annot.loc[
            annot.index.str.contains(density + "_") & ~annot.index.str.startswith("non"),
            "lipid_density",
        ] = density
    annot["lipid_size"] = np.nan
    for size in sizes:
        annot.loc[annot.index.str.startswith(size + "_"), "lipid_size"] = size
    # fix VLDL/LDL mix
    annot.loc[annot.index.str.contains("VLDL"), "lipid_density"] = "VLDL"

    # # make ordered categories
    annot["lipid_density"] = pd.Categorical(
        annot["lipid_density"], ordered=True, categories=densities.keys()
    )
    annot["lipid_size"] = pd.Categorical(
        annot["lipid_size"], ordered=True, categories=sizes.keys()
    )

    # Type of measurement/transformation
    annot["measurement_type"] = "absolute"
    annot.loc[
        annot["description"].str.contains("ratio ", case=False, regex=False)
        | annot["description"].str.contains("ratios", case=False, regex=False),
        "measurement_type",
    ] = "relative"

    # Variables not in our dataset
    annot = annot.drop(["HDL2_C", "HDL3_C", "Glycerol"])

    # Join with robustness metrics
    rob = get_nmr_feature_technical_robustness().drop("subgroup", axis=1)
    annot = annot.join(rob)

    # annot.to_csv(metadata_dir / "NMR_feature_annot.csv")

    return annot


def plot_nmr_feature_annotations() -> None:
    output_dir = (results_dir / "feature_network").mkdir()

    annot = get_nmr_feature_annotations()
    annot = annot.query("measurement_type == 'absolute'")

    attrs = [
        "metagroup",
        "group",
        "subgroup",
        "lipid_class",
        "lipid_density",
        "lipid_size",
    ]

    cmaps = ["tab10", "tab20", tab40(range(40)), "inferno", "inferno"]

    fig, axes = plt.subplots(len(attrs), 1, figsize=(4, 4 * len(attrs)))
    for ax, attr, cmap in zip(axes, attrs, cmaps):
        p = annot[attr].value_counts()
        sns.barplot(x=p, y=p.index, ax=ax, palette=cmap)
        ax.set(xlabel="Number of features")
    fig.savefig(output_dir / "NMR_features.frequency.svg", **figkws)
    plt.close(fig)


def unsupervised(
    x: DataFrame,
    y: DataFrame,
    attributes: tp.Sequence[str] = None,
    data_type: str = "NMR",
    suffix: str = "",
) -> None:
    """
    Unsupervised analysis of data using sample/feature correlations and
    dimentionality reduction and their visualization dependent on sample attributes.
    """
    if attributes is None:
        attributes = list()

    output_dir = (results_dir / f"unsupervised_{data_type}{suffix}").mkdir()

    feature_annotation = get_feature_annotations(x, data_type=data_type)
    feature_annotation = feature_annotation.drop(
        ["abbreviation", "name", "description", "subgroup"], axis=1, errors="ignore"
    )

    ## Clustermaps
    for c in ["abs", "z"]:
        grid = clustermap(
            x,
            row_colors=y[attributes],
            col_colors=feature_annotation,
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
    grid = clustermap(z_score(x).corr(), center=0, **kws, row_colors=feature_annotation)
    grid.savefig(
        output_dir / "unsupervised.correlation_variable.clustermap.svg",
        **figkws,
    )
    plt.close(grid.fig)

    grid = clustermap(z_score(x).T.corr(), **kws, row_colors=y[attributes])
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

            fig = plot_projection(res, y, factors=attributes, algo_name=name, **pkwargs)
            fig.savefig(
                output_dir / f"unsupervised.dimres.{name}.{label}svg",
                **figkws,
            )
            plt.close(fig)


def get_feature_network_knn(x: DataFrame, k: int = 15, **kwargs) -> DataFrame:
    from umap.umap_ import nearest_neighbors
    from scanpy.neighbors import Neighbors, _compute_connectivities_umap

    kws = dict(
        metric="euclidean",
        metric_kwds={},
        angular=True,
        random_state=None,
        low_memory=True,
    )
    kws.update(kwargs)

    # Without `angular` option, there are often repetitions (i.e. same neighbor twice)
    knn_indices, knn_distances, _ = nearest_neighbors(z_score(x).T, k, **kws)
    distances, connectivities = _compute_connectivities_umap(
        knn_indices,
        knn_distances,
        x.shape[1],
        k,
    )
    net = (
        pd.DataFrame(connectivities.toarray(), x.columns, x.columns)
        .sort_index(0)
        .sort_index(1)
    )
    np.fill_diagonal(net.values, 1.0)
    return net


def get_feature_network_correlation(x: DataFrame) -> DataFrame:
    from imc.operations import get_probability_of_gaussian_mixture, get_population

    corr = z_score(x).corr()
    np.fill_diagonal(corr.values, np.nan)
    corr_s = (6 ** corr).stack()

    p_bin = get_population(corr_s)
    p = pd.Series(get_probability_of_gaussian_mixture(corr_s, 3, 2), index=corr_s.index)
    p.loc[p_bin == False] = 0

    net = (
        p.rename_axis(["a", "b"])
        .reset_index()
        .pivot_table(index="a", columns="b", values=0)
    )
    np.fill_diagonal(net.values, 1.0)
    return net


def get_feature_network_hierarchical(x: DataFrame) -> DataFrame:
    raise NotImplementedError
    corr = z_score(x).corr()
    grid = clustermap(corr, metric="correlation", cmap="RdBu_r", center=0)
    plt.close(grid.fig)

    return net


def get_feature_network(x: DataFrame, y: DataFrame, data_type: str = "NMR") -> DataFrame:
    # import networkx as nx
    from imc.graphics import rasterize_scanpy

    output_dir = (results_dir / "feature_network").mkdir()

    # if method == "knn":
    #     net = get_feature_network_knn(
    #         x.drop(x.columns[x.columns.str.endswith("_pct")], axis=1)
    #     )
    # elif method == "correlation":
    #     net = get_feature_network_correlation(x)
    # else:
    #     raise ValueError("Method not understood. Choose one of 'knn' or 'correlation'.")

    # G = nx.from_pandas_adjacency(net)

    # annot = get_feature_annotations(x, data_type)
    # nx.set_node_attributes(G, annot.T.to_dict())
    # nx.write_gexf(G, output_dir / f"network.{data_type}.gexf")

    # fig, ax = plt.subplots(figsize=(6, 4))
    # nx.draw_spectral(G, ax=ax)

    # corr = z_score(x).corr()
    # grid = clustermap(corr, metric="euclidean", cmap="RdBu_r", center=0)

    # With scanpy
    annot = get_feature_annotations(x, data_type)
    annot = annot.drop(["abbreviation", "name", "description", "subgroup"], axis=1)

    xx = z_score(x.drop(annot.query("measurement_type == 'relative'").index, axis=1))

    annott = annot.loc[
        xx.columns, (annot.nunique() > 1) & ~annot.columns.str.contains("_")
    ]

    stats = pd.read_csv(
        results_dir / "supervised" / "supervised.alive.all_variables.stats.csv",
    )
    change = stats.set_index("Variable")["hedges_g"].rename("alive") * -1

    a = AnnData(xx.T, obs=annott.join(change))
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X")
    sc.tl.umap(a, gamma=1)
    sc.tl.leiden(a)

    feats = annott.columns.tolist() + ["leiden", "alive"]
    fig, ax = plt.subplots(
        len(feats), 1, figsize=(4, len(feats) * 4), sharex=True, sharey=True
    )
    group_cmap = tab40(range(a.obs["group"].nunique()))[:, :3]
    size_cmap = sns.color_palette("inferno", a.obs["lipid_size"].nunique())
    density_cmap = sns.color_palette("inferno", a.obs["lipid_density"].nunique())
    cmaps = (
        [group_cmap.tolist(), "Paired"]
        + [density_cmap, size_cmap]
        + ["tab10"]
        + ["coolwarm"]
    )
    for ax, feat, cmap in zip(fig.axes, feats, cmaps):
        p = (
            dict(cmap=cmap)
            if a.obs[feat].dtype.name.startswith("float")
            else dict(palette=cmap)
        )
        sc.pl.umap(a, color=feat, **p, edges=True, ax=ax, show=False, s=50, alpha=0.5)
    for ax in fig.axes:
        ax.set(xlabel="", ylabel="")
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "feature_annotation.network.scanpy.svg", **figkws)
    plt.close(fig)

    # Visualize as heatmaps as well
    corr = z_score(xx)
    grid = clustermap(
        corr,
        metric="cosine",
        cmap="coolwarm",
        center=0,
        row_colors=y[attributes],
        col_colors=annott.join(a.obs["leiden"]),
        rasterized=True,
        robust=True,
    )
    ax = grid.ax_heatmap
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
    grid.savefig(
        output_dir / "feature_annotation.network.scanpy.clustermap.svg", **figkws
    )
    plt.close(grid.fig)

    corr = z_score(xx).corr()
    grid = clustermap(
        corr,
        metric="euclidean",
        cmap="coolwarm",
        center=0,
        row_colors=a.obs["leiden"],
        rasterized=True,
    )
    ax = grid.ax_heatmap
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
    grid.savefig(
        output_dir / "feature_annotation.network.scanpy.clustermap.symmetric.svg",
        **figkws,
    )
    plt.close(grid.fig)

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
        p = (
            dict(cmap=cmap)
            if a.obs[feat].dtype.name.startswith("float")
            else dict(palette=cmap)
        )
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
        output_dir / "feature_annotation.network.scanpy.only_lipoproteins.svg", **figkws
    )
    plt.close(fig)

    import ringity
    import networkx as nx

    G = nx.from_scipy_sparse_matrix(a.obsp["connectivities"])
    dgm = ringity.diagram(G)
    dgm.score

    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48, wspace=0.05)
    ax1 = fig.add_subplot(gs[:-1, :])
    ax2 = fig.add_subplot(gs[-1, :-1])
    ax3 = fig.add_subplot(gs[-1, -1])
    ringity.plots.plot_nx(G, ax=ax1)
    ringity.plots.plot_seq(dgm, ax=ax2)
    ringity.plots.plot_dgm(dgm, ax=ax3)
    fig.savefig(
        output_dir / "feature_annotation.network.scanpy.ringity_analysis.svg", **figkws
    )
    plt.close(fig)

    # Check the same on steady state only
    xx = x.loc[y.query("group == 'control'").index]
    xx = z_score(xx.drop(annot.query("measurement_type == 'relative'").index, axis=1))
    xx = xx.loc[:, annot.index[annot["subgroup"].str.endswith("DL")]]
    annott = annot.loc[
        xx.columns, (annot.nunique() > 1) & ~annot.columns.str.contains("_")
    ]

    an = AnnData(xx.T, obs=annott.join(change))
    sc.pp.neighbors(an, n_neighbors=15, use_rep="X")
    sc.tl.umap(an)
    sc.tl.leiden(an)

    feats = annott.columns.tolist() + ["leiden", "alive"]
    fig, ax = plt.subplots(
        len(feats), 1, figsize=(4, len(feats) * 4), sharex=True, sharey=True
    )
    size_cmap = sns.color_palette("inferno", an.obs["lipid_size"].nunique())
    density_cmap = sns.color_palette("inferno", an.obs["lipid_density"].nunique())
    cmaps = ["tab20b", "Paired"] + [density_cmap, size_cmap] + ["tab10"] + ["coolwarm"]
    for ax, feat, cmap in zip(fig.axes, feats, cmaps):
        p = (
            dict(cmap=cmap)
            if an.obs[feat].dtype.name.startswith("float")
            else dict(palette=cmap)
        )
        sc.pl.umap(an, color=feat, **p, edges=True, ax=ax, show=False, s=50, alpha=0.5)
    for ax in fig.axes:
        ax.set(xlabel="", ylabel="")
    rasterize_scanpy(fig)
    fig.savefig(
        output_dir / "feature_annotation.only_healthy.network.scanpy.svg", **figkws
    )
    plt.close(fig)

    corr = z_score(xx)
    grid = clustermap(
        corr,
        metric="cosine",
        cmap="coolwarm",
        center=0,
        col_colors=an.obs["leiden"],
        rasterized=True,
    )
    ax = grid.ax_heatmap
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
    grid.savefig(
        output_dir / "feature_annotation.only_healthy.network.scanpy.clustermap.svg",
        **figkws,
    )
    plt.close(grid.fig)

    corr = z_score(xx).corr()
    grid = clustermap(
        corr,
        metric="euclidean",
        cmap="coolwarm",
        center=0,
        row_colors=an.obs["leiden"],
        rasterized=True,
    )
    ax = grid.ax_heatmap
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
    grid.savefig(
        output_dir
        / "feature_annotation.only_healthy.network.scanpy.clustermap.symmetric.svg",
        **figkws,
    )
    plt.close(grid.fig)

    #

    # Make one visualization with all info condensed in one figure axis
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter
    # TODO:

    #


def feature_physical_aggregate_change(x: DataFrame, y: DataFrame) -> None:
    output_dir = (results_dir / "feature_network").mkdir()
    annot = get_feature_annotations(x, "NMR")
    # Observe
    x2 = x.loc[:, annot.index]
    attrs = [
        "race",
        "sex",
        "obesity",
        "hospitalized",
        "intubated",
        "patient_group",
        "WHO_score_sample",
        "alive",
    ]
    n = max([y[attr].nunique() for attr in attrs])

    # Get mean per condition/attribute
    means: tp.Dict[str, tp.List[DataFrame]] = dict()
    means_z: tp.Dict[str, tp.List[DataFrame]] = dict()
    for i, attr in enumerate(attrs):
        means[attr] = list()
        means_z[attr] = list()
        groups = y[attr].cat.categories
        for j, group in enumerate(groups):
            h = y[attr] == group
            group_x = (
                x2.loc[h]
                .T.join(annot)
                .groupby(["lipid_density", "lipid_size"])
                .mean()
                .mean(1)
                .to_frame("value")
                .pivot_table(index="lipid_density", columns="lipid_size")["value"]
            )
            if j == 0:
                ctrl = group_x
            means[attr].append(group_x)
            # fold over normal
            means_z[attr].append(np.log(group_x / ctrl))

    # Plot
    for label, z in [("abs", False), ("z_score", True)]:
        fig, axes = plt.subplots(n, len(attrs), figsize=(4 * len(attrs), 2.5 * n))
        for i, attr in enumerate(attrs):
            groups = y[attr].cat.categories
            axes[0, i].set(title=attr)
            for j, group in enumerate(groups):
                ax = axes[j, i]
                if z:
                    group_x = means_z[attr][j]
                    flat = group_x.stack().dropna().values
                    group_x = (group_x - flat.mean()) / flat.std()
                    kws = dict(vmin=-3, vmax=3, cmap="coolwarm")
                else:
                    group_x = means[attr][j]
                    kws = dict(vmin=0, vmax=1)
                sns.heatmap(group_x, ax=ax, **kws, square=True)
                ax.set(ylabel=group)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            for ax in axes[j + 1 :, i]:
                ax.axis("off")
        fig.savefig(
            output_dir / f"NMR_features.mean.dependent_on_patient_attributes.{label}.svg",
            **figkws,
        )
        plt.close(fig)


def feature_properties_change(x: DataFrame, data_type: str = "NMR"):
    """
    See relationship between feature properties and change with disease.
    """
    output_dir = (results_dir / "feature_network").mkdir()

    # Collect fold-changes
    _changes = list()
    for attr in ["hospitalized", "intubated", "alive", "patient_group"]:
        stats = pd.read_csv(
            results_dir / "supervised" / f"supervised.{attr}.all_variables.stats.csv",
        )
        stats["A"] = stats["A"].astype(str)
        stats["B"] = stats["B"].astype(str)
        _v = "hedges" if "hedges" in stats.columns else "hedges_g"

        for _, (a, b) in stats[["A", "B"]].drop_duplicates().iterrows():
            change = (
                stats.query(f"A == '{a}' & B == '{b}'")
                .set_index("Variable")[_v]
                .rename(f"{attr}: {b}-{a}")
                * -1
            )
            _changes.append(change)
    changes = pd.concat(_changes, 1)

    # Visualize all fold-changes
    grid = clustermap(changes, xticklabels=True, cmap="coolwarm", center=0)
    grid.savefig(
        output_dir / "disease_change.all_variables.clustermap.svg",
        **figkws,
    )
    plt.close(grid.fig)

    annot = get_feature_annotations(x, data_type)
    annot = annot.loc[annot["measurement_type"] == "absolute"]

    rows = ["lipid_size", "lipid_density"]
    cols = changes.columns
    n, m = (
        len(rows),
        len(cols),
    )
    fig, axes = plt.subplots(n, m, figsize=(4 * m, 4 * n), sharex="row", sharey="row")
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            s = swarmboxenplot(
                data=annot.join(changes),
                x=row,
                y=col,
                ax=axes[i, j],
                plot_kws=dict(palette="inferno"),
            )
            axes[i, j].axhline(0, linestyle="--", color="grey")
            axes[i, j].set(title=col, ylabel="log fold-change", ylim=(-2.5, 1.5))
    fig.savefig(
        output_dir / "disease_change.dependent_on_feature_properties.swarmboxenplot.svg",
        **figkws,
    )
    plt.close(fig)


def feature_properties_pseudotime(x: DataFrame, data_type: str = "NMR"):
    """
    See relationship between feature properties and pseudotime.
    """
    output_dir = (results_dir / "feature_network").mkdir()
    corr_mat = pd.read_csv(
        results_dir
        / "unsupervised"
        / "unsupervised.variable_contibution_SpectralEmbedding.correlation.csv",
        index_col=0,
    )
    annot = get_feature_annotations(x, data_type)
    annot = annot.loc[annot["measurement_type"] == "absolute"]
    rows = ["lipid_size", "lipid_density"]
    cols = corr_mat.columns
    n, m = (
        len(rows),
        len(cols),
    )
    fig, axes = plt.subplots(n, m, figsize=(4 * m, 4 * n))
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            s = swarmboxenplot(
                data=annot.join(corr_mat),
                x=row,
                y=col,
                ax=axes[i, j],
                plot_kws=dict(palette="inferno"),
            )
            axes[i, j].axhline(0, linestyle="--", color="grey")
            axes[i, j].set(title=col, ylabel="Correlation with pseudotime axis")
    fig.savefig(
        output_dir
        / "pseudotime_change.dependent_on_feature_properties.swarmboxenplot.svg",
        **figkws,
    )
    plt.close(fig)


def diffusion(x, y) -> None:

    a = AnnData(x, obs=y)
    sc.pp.scale(a)
    sc.pp.neighbors(a, use_rep="X")
    sc.tl.diffmap(a)
    a.uns["iroot"] = np.flatnonzero(a.obs["WHO_score_patient"] == 0)[0]
    sc.tl.dpt(a)
    # fix for https://github.com/theislab/scanpy/issues/409:
    a.obs["dpt_order_indices"] = a.obs["dpt_pseudotime"].argsort()
    a.uns["dpt_changepoints"] = np.ones(a.obs["dpt_order_indices"].shape[0] - 1)


def get_explanatory_variables(x, y, data_type: str) -> None:
    """
    Find variables explaining the latent space discovered unsupervisedly.
    """
    output_dir = (results_dir / f"unsupervised_{data_type}").mkdir()

    res = pd.DataFrame(
        SpectralEmbedding().fit_transform(z_score(x)),
        index=x.index,
        columns=["SE1", "SE2"],
    )
    corr_mat = res.join(z_score(x)).corr().loc[x.columns, res.columns]
    corr_mat.to_csv(
        output_dir / "unsupervised.variable_contibution_SpectralEmbedding.correlation.csv"
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # # plot samples
    ax.scatter(*res.T.values)
    # # plot variables as vectors
    for i in corr_mat.index:
        ax.plot((0, corr_mat.loc[i, "SE1"] / 10), (0, corr_mat.loc[i, "SE2"] / 10))
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
        output_dir / "unsupervised.variable_contibution_SpectralEmbedding.regression.csv"
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
    from scipy.spatial.distance import euclidean, pdist, squareform
    from scipy import interpolate

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
            output_dir / f"unsupervised.{name}.patient_walk_in_space.scatter_arrow.svg",
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
    joint_metrics.to_csv(
        output_dir / "unsupervised.all_methods.patient_walk_in_space.metrics.csv"
    )

    joint_metrics = pd.read_csv(
        output_dir / "unsupervised.all_methods.patient_walk_in_space.metrics.csv",
        index_col=0,
    )
    joint_metricsz = (
        joint_metrics.groupby("method")[
            ["total_distance", "dislocation", "velo", "velo_dir"]
        ]
        .apply(z_score)
        .join(
            joint_metrics.groupby(level=0)[["n_timepoints", "time_days"]].apply(np.mean)
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
        output_dir / f"unsupervised.mean_methods.patient_walk_in_space.metrics.svg",
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
        if pg.anova(data=df, dv="velo", between=attribute)["p-unc"].squeeze() >= 0.05:
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
                output_dir / f"supervised.{attribute}.all_variables.swarmboxenplot.svg",
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
            output_dir / f"supervised.{attribute}.top_differential.swarmboxenplot.svg",
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
        "intubated",
        "patient_group",
        "alive",
    ]
    for attribute in attrs:
        stats_f = output_dir / f"supervised.{attribute}.model_fits.csv"

        if not stats_f.exists():
            # # Mixed effect
            data = z_score(x).join(y).dropna(subset=[attribute, "WHO_score_sample"])
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
                mdf = smf.glm(f"{feat} ~ {attribute} + WHO_score_sample", data).fit()
                res = (
                    mdf.params.to_frame("coefs")
                    .join(mdf.pvalues.rename("pvalues"))
                    .assign(feature=feat)
                )
                res = res.loc[res.index.str.contains(attribute)]
                _res_glm.append(res)
            res_glm = pd.concat(_res_glm)

            res = pd.concat([res_mlm.assign(model="mlm"), res_glm.assign(model="glm")])
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

    grid = clustermap(coef_mat, config="abs", cmap="RdBu_r", center=0, xticklabels=True)

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
            output_dir / f"supervised.{attribute}.Mixed_effect_models.volcano_plot.svg",
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
        grid.ax_heatmap.set(xlabel=f"Features (top 50 features for '{attribute}'")
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
        grid.ax_heatmap.set(xlabel=f"Features (top 50 features for '{attribute}'")
        grid.fig.savefig(
            output_dir
            / f"supervised.{attribute}.Mixed_effect_models.clustermap.top_50.sorted.svg",
            **figkws,
        )


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


def get_x_y_flow() -> tp.Tuple[DataFrame, DataFrame]:
    """
    Get flow cytometry dataset and its metadata.
    """
    projects_dir = Path("~/projects/archive").expanduser()
    project_dir = projects_dir / "covid-flowcyto"

    y2 = pd.read_parquet(project_dir / "metadata" / "annotation.pq")
    x2 = pd.read_parquet(project_dir / "data" / "matrix_imputed.pq")
    y2["date_sample"] = y2["datesamples"]

    return x2, y2


def get_matched_nmr_and_flow(
    x1: DataFrame, y1: DataFrame, x2: DataFrame, y2: DataFrame
) -> tp.Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Get flow cytometry dataset aligned with the NMR one.
    """
    joined = (
        x1.join(y1["accession"])
        .reset_index()
        .merge(x2.join(y2["accession"]), on="accession", how="inner")
        .set_index("Sample_id")
    )

    nx1 = joined.loc[:, x1.columns]
    nx2 = joined.loc[:, x2.columns]
    ny = y1.reindex(nx1.index)

    return nx1, nx2, ny


def cross_data_type_predictions() -> None:
    """
    Integrate the two data types on common ground
    """
    import sklearn

    output_dir = results_dir / "nmr_flow_predictions"
    output_dir.mkdir()

    x1, y1 = get_x_y_nmr()
    x2, y2 = get_x_y_flow()
    x1, x2, y = get_matched_nmr_and_flow(x1, y1, x2, y2)
    xz1 = z_score(x1)
    xz2 = z_score(x2)

    model = sklearn.linear_model.Ridge()
    model.fit(xz1, xz2)

    coefs = pd.DataFrame(model.coef_, index=xz2.columns, columns=xz1.columns)
    coefs = coefs.rename_axis(index="Immune populations", columns="Metabolites")
    coefs = coefs.loc[coefs.sum(1) > 0, :]
    coefs = coefs.loc[:, coefs.sum(0) > 0]

    grid = clustermap(
        coefs,
        center=0,
        cmap="RdBu_r",
        metric="correlation",
        cbar_kws=dict(label=r"Coefficient ($\beta$)"),
        rasterized=True,
    )
    grid.savefig(
        output_dir / "coefficients.all_variables.clustermap.svg",
        **figkws,
    )
    plt.close(grid.fig)

    x1var = ((coefs.var(0) / coefs.mean(0)) ** 2).sort_values()
    x2var = ((coefs.var(1) / coefs.mean(1)) ** 2).sort_values()

    x1var = (coefs.abs().sum(0)).sort_values()
    x2var = (coefs.abs().sum(1)).sort_values()

    n = 30
    grid = clustermap(
        coefs.loc[x2var.index[-n:], x1var.index[-n:]],
        center=0,
        cmap="RdBu_r",
        xticklabels=True,
        yticklabels=True,
        cbar_kws=dict(label=r"Coefficient ($\beta$)"),
        rasterized=True,
    )
    grid.savefig(
        output_dir / "coefficients.top_variables.clustermap.svg",
        **figkws,
    )
    plt.close(grid.fig)


def integrate_nmr_flow() -> None:
    """
    Integrate the two data types on common ground
    """
    import rcca  # type: ignore[import]

    output_dir = results_dir / "nmr_flow_integration"
    output_dir.mkdir()

    x1, y1 = get_x_y_nmr()
    x2, y2 = get_x_y_flow()

    x1, x2, y = get_matched_nmr_and_flow(x1, y1, x2, y2)

    xz1 = z_score(x1)
    xz2 = z_score(x2)

    # # Vanilla sklearn (doesn't work well)
    # from sklearn.cross_decomposition import CCA
    # n_comp = 2
    # cca = CCA(n_components=n_comp)
    # cca.fit(xz1, xz2)
    # x1_cca, x2_cca = cca.transform(xz1, xz2)
    # x1_cca = pd.DataFrame(x1_cca, index=x1.index)
    # x2_cca = pd.DataFrame(x2_cca, index=x2.index)

    # o = output_dir / f"CCA_integration.default"
    # metrics = assess_integration(
    #     a=x1_cca,
    #     b=x2_cca,
    #     a_meta=y,
    #     b_meta=y,
    #     a_name="NMR",
    #     b_name="Flow cytometry",
    #     output_prefix=o,
    #     algos=["cca", "pca", "umap"] if n_comp > 2 else ["cca"],
    #     attributes=[
    #         "dataset",
    #         "group",
    #         "hospitalized",
    #         "intubated",
    #         "WHO_score_sample",
    #         "patient_code",
    #     ],
    #     plot=True,
    #     algo_kwargs=dict(umap=dict(gamma=25)),
    #     plt_kwargs=dict(s=100),
    #     cmaps=["Set1", "Set2", "Dark2", "inferno", None],
    # )

    # CCA with CV and regularization (very good)
    model_f = output_dir / "trained_model_CV.hdf5"
    numCCs = [4, 5, 6, 7, 8]
    regs = [10, 50, 70, 80, 90, 100, 150, 200, 300, 400, 500, 1_000]

    ccaCV = rcca.CCACrossValidate(kernelcca=False, numCCs=numCCs, regs=regs)
    if not model_f.exists():
        # # N.B.: this is not deterministic and I didn't find a way to set a seed of equivalent.
        # # By saving/loading the model I ensure the results are reproducible
        # # but I don't think that says anything about how correct they are.
        ccaCV.train([xz1.values, xz2.values])
        ccaCV.save(model_f)
    ccaCV.load(model_f)

    n_comp = ccaCV.best_numCC
    reg = ccaCV.best_reg
    print(n_comp, reg)
    x1_cca = pd.DataFrame(ccaCV.comps[0], index=x1.index)
    x2_cca = pd.DataFrame(ccaCV.comps[1], index=x2.index)

    o = output_dir / f"rCCA_integration.CV.{n_comp}.{reg}"
    metrics = assess_integration(
        a=x1_cca,
        b=x2_cca,
        a_meta=y,
        b_meta=y,
        a_name="NMR",
        b_name="Flow cytometry",
        output_prefix=o,
        algos=["cca", "pca", "umap"] if n_comp > 2 else ["cca"],
        attributes=[
            "dataset",
            "group",
            "hospitalized",
            "WHO_score_sample",
            "patient_code",
        ],
        plot=True,
        algo_kwargs=dict(umap=dict(gamma=25)),
        plt_kwargs=dict(s=100),
        cmaps=["Set1", "Set2", "Dark2", "inferno", None],
    )
    pd.DataFrame(metrics)

    # Plot in light of attributes
    x_comb = pd.concat(
        [
            pd.DataFrame(x1_cca.values, index=x1_cca.index + "_NMR"),
            pd.DataFrame(x2_cca.values, index=x2_cca.index + "_Flow"),
        ]
    )
    y_comb = pd.concat([y.assign(dataset="NMR"), y.assign(dataset="flow_cytometry")])
    y_comb.index = x_comb.index

    fig = plot_projection(x_comb, y_comb, factors=attributes, algo_name="CCA", n_dims=5)
    fig.savefig(
        output_dir / "rCCA_integration.all_CCs.attributes.scatter.svg",
        **figkws,
    )
    plt.close(fig)

    # Predict other data type for each one
    w1 = pd.DataFrame(ccaCV.ws[0], index=x1.columns)
    w2 = pd.DataFrame(ccaCV.ws[1], index=x2.columns)

    n = 10
    f1 = w1[0].abs().sort_values().tail(n + 1).index[::-1]
    f2 = w2[0].abs().sort_values().tail(n + 1).index[::-1]

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 3 * 2))
    for rank in range(n):
        axes[0, rank].scatter(x1_cca[0], x1[f1[rank]], alpha=0.5, s=5)
        axes[0, rank].set(xlabel="CCA1", ylabel=f1[rank])
        axes[1, rank].scatter(x2_cca[0], x2[f2[rank]], alpha=0.5, s=5)
        axes[1, rank].set(xlabel="CCA1", ylabel=f2[rank])
    fig.savefig(
        output_dir / "rCCA_integration.CC1.top_variables.scatter.svg",
        **figkws,
    )
    plt.close(fig)

    o1 = z_score(x1_cca @ w2.T)
    o2 = z_score(x2_cca @ w1.T)

    unsupervised(
        o1,
        y,
        attributes=attributes,
        data_type="flow_cytometry",
        suffix="_predicted_from_NMR",
    )
    unsupervised(
        o2,
        y,
        attributes=attributes,
        data_type="NMR",
        suffix="_predicted_from_flow_cytometry",
    )


def assess_integration(
    a: DataFrame,
    b: DataFrame,
    a_meta: tp.Union[Series, DataFrame],
    b_meta: tp.Union[Series, DataFrame],
    output_prefix: Path = None,
    a_name: str = "IMC",
    b_name: str = "RNA",
    algos: tp.Sequence[str] = ["pca", "umap"],
    attributes: tp.Sequence[str] = ["dataset"],
    only_matched_attributes: bool = False,
    subsample: str = None,
    plot: bool = True,
    algo_kwargs: tp.Dict[str, tp.Dict[str, tp.Any]] = None,
    plt_kwargs: tp.Dict[str, tp.Any] = None,
    cmaps: tp.Sequence[str] = None,
) -> tp.Dict[str, tp.Dict[str, float]]:
    """
    Keyword arguments are passed to the scanpy plotting function.
    """
    from sklearn.metrics import silhouette_score
    from imc.graphics import rasterize_scanpy
    from seaborn_extensions.annotated_clustermap import is_numeric

    if plot:
        assert output_prefix is not None, "If `plot`, `output_prefix` must be given."
    if algo_kwargs is None:
        algo_kwargs = dict()
    for algo in algos:
        if algo not in algo_kwargs:
            algo_kwargs[algo] = dict()
    if plt_kwargs is None:
        plt_kwargs = dict()

    if isinstance(a_meta, pd.Series):
        a_meta = a_meta.to_frame()
    if isinstance(b_meta, pd.Series):
        b_meta = b_meta.to_frame()

    # TODO: silence scanpy's index str conversion
    adata = AnnData(a.append(b))
    adata.obs["dataset"] = [a_name] * a.shape[0] + [b_name] * b.shape[0]
    for attr in attributes:
        if attr == "dataset":
            continue
        adata.obs[attr] = "None"
        adata.obs.loc[adata.obs["dataset"] == a_name, attr] = a_meta[attr]
        adata.obs.loc[adata.obs["dataset"] == b_name, attr] = b_meta[attr]
    adata.obs_names_make_unique()

    if "pca" in algos:
        sc.pp.scale(adata)
        sc.pp.pca(adata, **algo_kwargs["pca"])

    if "umap" in algos:
        if subsample is not None:
            if "frac=" in subsample:
                sel_cells = adata.obs.sample(
                    frac=float(subsample.split("frac=")[1])
                ).index
            elif "n=" in subsample:
                sel_cells = adata.obs.sample(n=int(subsample.split("n=")[1])).index
            adata = adata[sel_cells]
        sc.pp.neighbors(adata)
        sc.tl.umap(adata, **algo_kwargs["umap"])
        # sc.tl.leiden(a, resolution=0.5)

    remain = [x for x in algos if x not in ["pca", "umap"]]
    if remain:
        assert len(remain) == 1
        algo = remain[0]
        adata.obsm[f"X_{algo}"] = adata.X

    # get score only for matching attributes across datasets
    if only_matched_attributes:
        attrs = [attr for attr in attributes if attr != "dataset"]
        sel = adata.obs.groupby(attrs)["dataset"].nunique() >= 2
        sel = sel.reset_index().drop("dataset", 1)
        adata = adata[
            pd.concat([adata.obs[attr].isin(sel[attr]) for attr in attrs], 1).all(1), :
        ]

    scores: tp.Dict[str, tp.Dict[str, float]] = dict()
    for i, algo in enumerate(algos):
        scores[algo] = dict()
        for j, attr in enumerate(attributes):
            scores[algo][attr] = silhouette_score(
                adata.obsm[f"X_{algo}"], adata.obs[attr]
            )

    if not plot:
        return scores

    # Plot
    adata = adata[adata.obs.sample(frac=1).index, :]
    n, m = len(algos), len(attributes)
    fig, axes = plt.subplots(n, m, figsize=(4 * m, 4 * n), squeeze=False)
    cmaps = ([None] * len(attributes)) if cmaps is None else cmaps
    for i, algo in enumerate(algos):
        for j, (attr, cmap) in enumerate(zip(attributes, cmaps)):
            try:
                if is_numeric(a_meta[attr]):
                    cmap_kws = dict(color_map=cmap)
                else:
                    cmap_kws = dict(palette=cmap)
            except KeyError:
                cmap_kws = {}
            ax = axes[i, j]
            sc.pl.embedding(
                adata,
                basis=algo,
                color=attr,
                alpha=0.5,
                show=False,
                **plt_kwargs,
                **cmap_kws,
                ax=ax,
            )
            s = scores[algo][attr]
            ax.set(
                title=f"{attr}, score: {s:.3f}",
                xlabel=algo.upper() + "1",
                ylabel=algo.upper() + "2",
            )
    rasterize_scanpy(fig)
    fig.savefig(tp.cast(Path, output_prefix) + ".joint_datasets.svg", **figkws)
    plt.close(fig)

    return scores


def predict_outcomes() -> None:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_validate

    from joblib import Parallel, delayed

    def fit(_, model, X, y, k=None):
        feature_attr = {
            RandomForestClassifier: "feature_importances_",
            LogisticRegression: "coef_",
            # ElasticNet: "coef_",
            LinearSVC: "coef_",
        }
        kws = dict(
            cv=10, scoring="roc_auc", return_train_score=True, return_estimator=True
        )

        # # Randomize order of both X and y (jointly)
        # X = X.sample(frac=1.0).copy()
        # y = y.reindex(X.index).copy()

        clf = model()

        # Build pipeline
        components = list()
        # # Z-score if needed
        if not isinstance(clf, RandomForestClassifier):
            components += [("scaler", StandardScaler())]
        # # Do feature selection if requested
        if k is not None:
            components += [("selector", SelectKBest(mutual_info_classif, k=k))]
        # # Finally add the classifier
        components += [("classifier", clf)]
        pipe = Pipeline(components)

        # Train/cross-validate with real data
        out1 = cross_validate(pipe, X, y, **kws)
        # Train/cross-validate with shuffled labels
        out2 = cross_validate(pipe, X, y.sample(frac=1.0), **kws)

        # Extract coefficients/feature importances
        feat = feature_attr[clf.__class__]
        coefs = tuple()
        for out in [out1, out2]:
            co = pd.DataFrame(
                [
                    pd.Series(
                        getattr(c["classifier"], feat),
                        index=X.columns[c["selector"].get_support()]
                        if k is not None
                        else X.columns,
                    )
                    for c in out["estimator"]
                ]
            )
            # If keeping all variables, simply report mean
            if k is None:
                coefs += (co.mean(0),)
            # otherwise, simply count how often variable was chosen at all
            else:
                coefs += ((~co.isnull()).sum(),)
        return (
            out1["train_score"].mean(),
            out1["test_score"].mean(),
            out2["train_score"].mean(),
            out2["test_score"].mean(),
        ) + coefs

    output_dir = results_dir / "predict"
    output_dir.mkdir()

    x1, y1 = get_x_y_nmr()
    x2, y2 = get_x_y_flow()
    x1, x2, y = get_matched_nmr_and_flow(x1, y1, x2, y2)
    xz1 = z_score(x1)
    xz2 = z_score(x2)

    # Use only mild-severe patients
    m = y.query("patient_group.isin(['moderate', 'severe']).values", engine="python")

    # Convert classes to binary
    target = m["patient_group"].cat.remove_unused_categories().cat.codes

    # Align
    xz1 = xz1.loc[target.index]
    xz2 = xz2.loc[target.index]

    # # For the other classifiers
    N = 100

    insts = [
        RandomForestClassifier,
        # LogisticRegression,
        # LinearSVC,
        # ElasticNet,
    ]
    for dtype, X in [("NMR", xz1), ("flow_cytometry", xz2), ("combined", xz1.join(xz2))]:
        for label, k in [("", None), (".feature_selection", 8)]:
            for model in insts:
                name = str(type(model())).split(".")[-1][:-2]
                print(name)

                # Fit
                res = Parallel(n_jobs=-1)(
                    delayed(fit)(i, model=RandomForestClassifier, X=X, y=target, k=k)
                    for i in range(N)
                )
                # Get ROC_AUC scores only
                scores = pd.DataFrame(
                    np.asarray([r[:-2] for r in res]),
                    columns=[
                        "train_score",
                        "test_score",
                        "train_score_random",
                        "test_score_random",
                    ],
                ).rename_axis(index="iteration")
                scores.to_csv(
                    output_dir
                    / f"severe-mild_prediction.{dtype}.{name}{label}.scores.csv"
                )

    label = ""
    k = None
    name = str(type(model())).split(".")[-1][:-2]
    _res = list()
    for dtype, X in [("NMR", xz1), ("flow_cytometry", xz2), ("combined", xz1.join(xz2))]:
        scores = pd.read_csv(
            output_dir / f"severe-mild_prediction.{dtype}.{name}{label}.scores.csv",
            index_col=0,
        )
        p = scores.loc[:, scores.columns.str.contains("test")].melt()
        _res.append(p.assign(dataset=dtype))
    res = pd.concat(_res)
    res["dataset"] = pd.Categorical(
        res["dataset"], categories=["NMR", "flow_cytometry", "combined"]
    )
    res["variable"] = pd.Categorical(
        res["variable"], categories=["test_score_random", "test_score"]
    )

    fig, _ = swarmboxenplot(data=res, x="variable", hue="dataset", y="value")
    fig.axes[0].axhline(0.5, linestyle="--", color="grey")
    fig.savefig(
        output_dir
        / f"severe-mild_prediction.performance_data_type_comparison.{name}{label}.scores.svg",
        **figkws,
    )

    p = res.query("variable == 'test_score'")
    fig, _ = swarmboxenplot(data=p, x="dataset", y="value")
    fig.axes[0].axhline(0.5, linestyle="--", color="grey")

    s = p.groupby("dataset")["value"].mean().squeeze()
    for i, q in enumerate(s.index):
        fig.axes[0].text(i, 0.7, s=f"{s[q]:.3f}", ha="center")
    fig.savefig(
        output_dir
        / f"severe-mild_prediction.performance_data_type_comparison.{name}{label}.scores.only_real.svg",
        **figkws,
    )


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
