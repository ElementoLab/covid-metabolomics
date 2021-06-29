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
import pingouin as pg

from imc.types import Path, Series, DataFrame
from imc.utils import z_score
from imc.graphics import close_plots
from seaborn_extensions import clustermap, swarmboxenplot, volcano_plot
from seaborn_extensions.annotated_clustermap import is_numeric

from src.config import *
from src.models import DataSet, AnnData
from src.ops import (
    unsupervised,
    get_explanatory_variables,
    overlay_individuals_over_global,
    plot_projection,
)


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

    supervised(x1, y1, attributes)
    supervised_joint(x1, y1, attributes)
    supervised_temporal(x1, y1, attributes)
    feature_enrichment()
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


def get_change_per_patient(x: DataFrame, y: DataFrame) -> None:
    output_dir = (results_dir / "supervised").mkdir()
    stats_f = output_dir / "supervised.joint_model.model_fits.csv"
    sig = (
        pd.read_csv(stats_f, index_col=0)
        .query("model == 'mlm'")
        .loc["WHO_score_sample"]
        .set_index("feature")["coefs"]
    )

    # xz = x.groupby(y['patient_code']).apply(z_score).dropna()
    # yz = y.reindex(xz.index)

    xz = z_score(x)
    score = xz @ sig
    # score = (score - score.min()) / (score.max() - score.min())

    score.groupby(y["WHO_score_sample"]).mean()
    score.groupby(y["patient_code"]).mean()

    var = get_nmr_feature_annotations()

    try:
        exp = pd.read_csv(metadata_dir / "ukbb.feature_range.csv", index_col=0)
    except FileNotFoundError:
        exp = pd.DataFrame(index=x.columns, columns=["mean", "ci_lower", "ci_upper"])

    for pat in y["patient_code"].unique():
        pdf = y.loc[y["patient_code"] == pat].sort_values("date_sample")
        if pdf.shape[0] <= 1:
            continue

        score.loc[pdf.index]
        pdf["days_since_symptoms"]
        t1 = pdf.iloc[0, :]
        t2 = pdf.iloc[1, :]
        diff = score.loc[t2.name] - score.loc[t1.name]
        days = t2["days_since_symptoms"] - t1["days_since_symptoms"]
        print(
            f"Patient {pat}; "
            f"timepoints: {pdf.shape[0]}; "
            f"first t: {t1['days_since_symptoms']}; "
            f"time: {days}; "
            f"diff: {diff:.3f}; "
            f"alive: {t1['alive']}"
        )
        if pd.isnull(days):
            continue

        # for feat in ['GlycA', 'score']:
        feat = "GlycA"
        px = pdf["days_since_symptoms"].astype(float)

        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        axes[0].scatter(px, x[feat].loc[pdf.index])
        sns.barplot(px, x[feat].loc[pdf.index], ax=axes[1])
        unit = var.loc[feat, "unit"]
        for ax in axes:
            ax.set(ylabel=f"{feat} ({unit})")

            base = exp.reindex([feat]).squeeze()
            if base.isnull().all():
                continue

            ax.axhline(base["mean"], linestyle="--", color="grey", linewidth=0.5)
            n = len(px)
            for i in range(1, 4):
                py1 = base["mean"] - base["std"] * i
                py2 = base["mean"] + base["std"] * i
                ax.fill_between(
                    px,
                    [py1] * n,
                    [py2] * n,
                    color="grey",
                    alpha=0.1,
                    zorder=-100 - i,
                    edgecolor=None,
                )
        fig.savefig(output_dir / f"patient_{pat}.timeline_{feat}.svg", **figkws)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        axes[0].scatter(px, score.loc[pdf.index])
        # color = plt.get_cmap("RdBu_r")(score.loc[pdf.index])[:, :-1]
        sns.barplot(px, score.loc[pdf.index], ax=axes[1])
        for ax in axes:
            ax.set(ylabel=f"COVID severity score")

            base = {"mean": score.mean(), "std": score.std()}
            ax.axhline(base["mean"], linestyle="--", color="grey", linewidth=0.5)
            n = len(px)
            for i in range(1, 4):
                py1 = base["mean"] - base["std"] * i
                py2 = base["mean"] + base["std"] * i
                ax.fill_between(
                    px,
                    [py1] * n,
                    [py2] * n,
                    color="grey",
                    alpha=0.1,
                    zorder=-100 - i,
                    edgecolor=None,
                )
        fig.savefig(output_dir / f"patient_{pat}.timeline_score.svg", **figkws)
        plt.close(fig)


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
    y["hospitalized"] = pd.Categorical(y["hospitalized"].replace({"no": False, "yes": True}))
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

    # ids = pd.read_csv(metadata_dir / 'accession_ip_mapping.csv', index_col=0, squeeze=True).rename("accession2")

    # Intubation
    intubation = pd.read_csv(metadata_dir / "intubation_dates.csv", index_col=0)[
        "date intubation"
    ].rename("date_intubation")
    d = (
        intubation.apply(
            lambda x: (pd.NaT, pd.NaT) if pd.isnull(x) else tuple(map(pd.to_datetime, x.split("-")))
        )
        .apply(pd.Series)
        .rename(columns={0: "date_intubation_start", 1: "date_intubation_end"})
    )
    y = y.join(d)
    y["sample_under_intubation"] = (y["date_sample"] > y["date_intubation_start"]) & (
        y["date_sample"] < y["date_intubation_end"]
    )
    y["days_since_intubation_start"] = (y["date_sample"] - y["date_intubation_start"]).apply(
        lambda x: x.days
    )

    # Tosilizumab treatment
    tosi = (
        pd.read_csv(metadata_dir / "tosilizumab_dates.csv", index_col=0)["date tosi"]
        .rename("date_tosilizumab_start")
        .apply(pd.to_datetime)
    )
    y = y.join(tosi)
    y["days_since_tosilizumab_start"] = (y["date_sample"] - y["date_tosilizumab_start"]).apply(
        lambda x: x.days
    )

    return x, y

    return DataSet(x=x, obs=y, name="", data_type="NMR", var=get_nmr_feature_annotations())


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
            url = URL("https://biobank.ndph.ox.ac.uk/showcase/showcase/docs/nmrm_app2.pdf")
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
                perf = [r.strip().split(", R : ") for r in lines[start_line + 3].split("CV: ")[1:]]
                res = pd.DataFrame(perf, index=features, columns=["CV", "R"]).assign(group=group)
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
        res = res.drop("group", axis=1).reset_index().drop_duplicates().set_index("feature")
        res.to_csv(nightingale_rep_csv_f)
    robustness = pd.read_csv(nightingale_rep_csv_f, index_col=0)
    return robustness


@close_plots
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
    fig.savefig((results_dir / "nightingale_tech").mkdir() / "assay_robustness.svg", **figkws)

    # # Relate technical and biological variability
    # x, _ = get_x_y_nmr()
    # cv2 = ((x.std() / x.mean()) ** 2).rename("CV2")

    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.scatter(annot["R"] ** 6, cv2.reindex(annot.index), alpha=0.5)


@close_plots
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


@close_plots
def compare_clinical_and_panel():
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
    ## ranges are collapsed to min-max across sex
    healthy_ranges = {
        "total_bilirubin": (1.0, 1.4),
        "AST": (10, 40),
        "ALT": (7, 56),
        "creatinine": (0.6, 1.3),
        "CRP": (0, 10),
        "hemoglobin": (12, 17.5),
        "hematocrit": (36, 51),
        "LDH": (140, 280),
        "RDWCV": (11.8, 16.1),
        "MCV": (80, 96),
    }

    var = get_nmr_feature_annotations()
    x, y = get_x_y_nmr()

    y["ALT-AST_log_ratio"] = np.log(y["ALT"] / y["AST"])
    labs += ["ALT-AST_log_ratio"]

    z = x.join(y[labs + ["WHO_score_sample"]])
    fig, stats = swarmboxenplot(
        data=z, x="WHO_score_sample", y=labs, plot_kws=dict(palette="inferno")
    )
    for ax, lab in zip(fig.axes, labs):
        if lab in healthy_ranges:
            r = healthy_ranges[lab]
            ax.fill_between(
                y["WHO_score_sample"].cat.categories.tolist(), *r, alpha=0.25, color="grey"
            )
    # # add healthy range
    n = fig.axes[0].get_gridspec().ncols
    for ax in fig.axes[-n:]:
        ax.set_xlabel("WHO_score_sample")
    fig.savefig(results_dir / "clinical_parameters.blood_liver.change_with_severity.svg", **figkws)

    # #
    labs += ["WHO_score_sample"]
    con = (
        x.join(y[labs])
        .dropna()
        .astype(float)
        .corr()
        .rename_axis(index="Metabolites")
        .rename_axis(columns="Clinical parameters")
    )
    grid = clustermap(
        con.loc[x.columns, labs],
        cbar_kws=dict(label="Pearson correlation"),
        cmap="RdBu_r",
        center=0,
        dendrogram_ratio=0.1,
        row_colors=var[["metagroup", "group"]],
        figsize=(5, 10),
    )
    grid.ax_heatmap.set_ylabel("Metabolites" + grid.ax_heatmap.get_ylabel())
    grid.fig.savefig(
        results_dir / "clinical_parameters.blood_liver.correlation_with_metabolism.svg", **figkws
    )

    # (con.loc[x.columns, 'ALT'] - con.loc[x.columns, 'WHO_score_sample'] * -1).abs().sort_values()
    # fig, stats = swarmboxenplot(data=x.join(y), x='WHO_score_sample', y='XL_HDL_P')
    # fig, ax = plt.subplots()
    # ax.scatter(data=x.join(y), x='ALT', y='XL_HDL_P')

    # Creatinine comparison
    unit = var.loc["Creatinine", "unit"]
    v = (y["creatinine"] * 100).append(x["Creatinine"]).max()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4))
    for ax in axes:
        ax.plot((0, v), (0, v), linestyle="--", color="grey")
        ax.scatter(y["creatinine"] * 100, x["Creatinine"], alpha=0.5)
        s = pg.corr((y["creatinine"] * 100), x["Creatinine"]).squeeze()
        ax.set(
            title=f"Creatinine\nr = {s['r']:.3f}; p = {s['p-val']:.3e}; ",
            xlabel="Clinical labs",
            ylabel=f"NMR panel ({unit})",
        )
    axes[-1].loglog()
    fig.savefig(results_dir / "NMR_vs_clinical.creatinine.svg", **figkws)

    import statsmodels.formula.api as smf

    data = z_score(z.dropna().astype(float))

    # # GLM
    labs.pop(labs.index("WHO_score_sample"))
    labs.pop(labs.index("ALT-AST_log_ratio"))

    _res_glm = list()
    for feat in tqdm(x.columns, desc="feature", position=1):
        mdf = smf.glm(f"{feat} ~ {' + '.join(labs)}", data).fit()
        res = mdf.params.to_frame("coefs").join(mdf.pvalues.rename("pvalues")).assign(feature=feat)
        _res_glm.append(res)
    res_glm = pd.concat(_res_glm).drop("Intercept")
    res_glm["qvalues"] = pg.multicomp(res_glm["pvalues"].values, method="fdr_bh")[1]

    coefs = res_glm.reset_index().pivot_table(index="feature", columns="index", values="coefs")
    qvals = res_glm.reset_index().pivot_table(index="feature", columns="index", values="qvalues")
    log_qvals = -np.log10(qvals)

    grid = clustermap(
        log_qvals,
        cbar_kws=dict(label="-log10(p-value)"),
        cmap="RdBu_r",
        robust=True,
        center=0,
        dendrogram_ratio=0.1,
        row_colors=var[["metagroup", "group"]],
        figsize=(5, 10),
    )
    grid.fig.savefig(
        results_dir / "clinical_parameters.blood_liver.regression_with_metabolism.svg", **figkws
    )


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
    feature_annotation = pd.DataFrame(index=x.columns, columns=var_annot.keys(), dtype=bool)
    for group in var_annot:
        feature_annotation.loc[~feature_annotation.index.isin(var_annot[group]), group] = False
    return feature_annotation


@close_plots
def plot_all_signatures():
    from src.ukbb import get_signatures as ukbbsigs
    from src.dierckx import get_signatures as dierckxsigs

    stats_f = results_dir / "supervised" / "supervised.joint_model.model_fits.csv"
    sig = (
        pd.read_csv(stats_f, index_col=0)
        .query("model == 'mlm'")
        .loc["WHO_score_sample"]
        .set_index("feature")["coefs"]
        .rename("severity")
    )
    dim_f = (
        results_dir
        / "unsupervised_NMR"
        / "unsupervised.variable_contribution_SpectralEmbedding.correlation.variable_ordering.csv"
    )
    dim = pd.read_csv(dim_f, index_col=0)

    ukbb_sigs = ukbbsigs().sort_index()
    dierckx_sigs = dierckxsigs().sort_index()

    p = (
        ukbb_sigs.join(sig)
        .join(dim)
        .join(dierckx_sigs)
        .drop("future_infectious_disease", 1)
        .dropna()
    )

    grid = clustermap(p, col_cluster=False, center=0, cmap="RdBu_r")


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
        | annot["group"].str.contains("Lipo|lipo|lipid") & (annot["group"] != "Apolipoproteins"),
        "metagroup",
    ] = "Lipid"

    annot["lipid_class"] = np.nan
    for lipid_class in lipoprots:
        annot.loc[
            annot.index.str.endswith(lipid_class) | annot.index.str.endswith(lipid_class + "_pct"),
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
    annot["lipid_size"] = pd.Categorical(annot["lipid_size"], ordered=True, categories=sizes.keys())

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


@close_plots
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
    net = pd.DataFrame(connectivities.toarray(), x.columns, x.columns).sort_index(0).sort_index(1)
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

    net = p.rename_axis(["a", "b"]).reset_index().pivot_table(index="a", columns="b", values=0)
    np.fill_diagonal(net.values, 1.0)
    return net


def get_feature_network_hierarchical(x: DataFrame) -> DataFrame:
    raise NotImplementedError


@close_plots
def get_feature_network(x: DataFrame, y: DataFrame, data_type: str = "NMR") -> DataFrame:
    # import networkx as nx
    import scanpy as sc
    from imc.graphics import rasterize_scanpy
    from src.models import PyMDE, AnnData

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
    grid.savefig(output_dir / "feature_annotation.network.scanpy.clustermap.svg", **figkws)
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
    fig.savefig(output_dir / "feature_annotation.network.scanpy.only_lipoproteins.svg", **figkws)
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
    fig.savefig(output_dir / "feature_annotation.network.scanpy.ringity_analysis.svg", **figkws)
    plt.close(fig)

    # Check the same on steady state only
    xx = x.loc[y.query("group == 'control'").index]
    xx = z_score(xx.drop(annot.query("measurement_type == 'relative'").index, axis=1))
    xx = xx.loc[:, annot.index[annot["subgroup"].str.endswith("DL")]]
    annott = annot.loc[xx.columns, (annot.nunique() > 1) & ~annot.columns.str.contains("_")]

    an = AnnData(xx.T, obs=annott.join(change))
    sc.pp.neighbors(an, n_neighbors=15, use_rep="X")
    sc.tl.umap(an)
    sc.tl.leiden(an)

    feats = annott.columns.tolist() + ["leiden", "alive"]
    fig, ax = plt.subplots(len(feats), 1, figsize=(4, len(feats) * 4), sharex=True, sharey=True)
    size_cmap = sns.color_palette("inferno", an.obs["lipid_size"].nunique())
    density_cmap = sns.color_palette("inferno", an.obs["lipid_density"].nunique())
    cmaps = ["tab20b", "Paired"] + [density_cmap, size_cmap] + ["tab10"] + ["coolwarm"]
    for ax, feat, cmap in zip(fig.axes, feats, cmaps):
        p = dict(cmap=cmap) if an.obs[feat].dtype.name.startswith("float") else dict(palette=cmap)
        sc.pl.umap(an, color=feat, **p, edges=True, ax=ax, show=False, s=50, alpha=0.5)
    for ax in fig.axes:
        ax.set(xlabel="", ylabel="")
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "feature_annotation.only_healthy.network.scanpy.svg", **figkws)
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
        output_dir / "feature_annotation.only_healthy.network.scanpy.clustermap.symmetric.svg",
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


def feature_properties_change(x: DataFrame, data_type: str = "NMR") -> None:
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


def feature_properties_pseudotime(x: DataFrame, data_type: str = "NMR") -> None:
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
        output_dir / "pseudotime_change.dependent_on_feature_properties.swarmboxenplot.svg",
        **figkws,
    )
    plt.close(fig)


# def diffusion(x, y) -> None:

#     a = AnnData(x, obs=y)
#     sc.pp.scale(a)
#     sc.pp.neighbors(a, use_rep="X")
#     sc.tl.diffmap(a)
#     a.uns["iroot"] = np.flatnonzero(a.obs["WHO_score_patient"] == 0)[0]
#     sc.tl.dpt(a)
#     # fix for https://github.com/theislab/scanpy/issues/409:
#     a.obs["dpt_order_indices"] = a.obs["dpt_pseudotime"].argsort()
#     a.uns["dpt_changepoints"] = np.ones(a.obs["dpt_order_indices"].shape[0] - 1)


def supervised(
    x: DataFrame,
    y: DataFrame,
    attributes: tp.Sequence[str],
    plot_all: bool = True,
    overwrite: bool = False,
) -> None:
    import statsmodels.formula.api as smf

    output_dir = (results_dir / "supervised").mkdir()

    # # convert ordinal categories to numeric
    for attr in ["WHO_score_sample", "WHO_score_patient"]:
        y[attr] = y[attr].cat.codes.astype(float).replace(-1, np.nan)
    for attr in attributes:
        if y[attr].dtype.name == "Int64":
            y[attr] = y[attr].astype(float)
    cats = list(filter(lambda w: ~is_numeric(y[w]), attributes))
    nums = list(filter(lambda w: is_numeric(y[w]), attributes))

    for attr in cats:
        # first just get the stats
        fig, stats = swarmboxenplot(
            data=x.join(y),
            x=attr,
            y=x.columns,
            swarm=False,
            boxen=False,
        )
        plt.close(fig)
        stats.to_csv(
            output_dir / f"supervised.{attr}.all_variables.stats.csv",
            index=False,
        )

        if plot_all:
            fig, stats = swarmboxenplot(
                data=x.join(y),
                x=attr,
                y=x.columns,
            )
            fig.savefig(
                output_dir / f"supervised.{attr}.all_variables.swarmboxenplot.svg",
                **figkws,
            )

        # now plot top variables
        fig, _ = swarmboxenplot(
            data=x.join(y),
            x=attr,
            y=stats.sort_values("p-unc")["Variable"].head(20).unique(),
            plot_kws=dict(palette=palettes.get(attr)),
        )
        fig.savefig(
            output_dir / f"supervised.{attr}.top_differential.swarmboxenplot.svg",
            **figkws,
        )

    # Use also a MLM and compare to GLM
    for attr in tqdm(attributes, desc="attribute", position=0):
        stats_f = output_dir / f"supervised.{attr}.model_fits.csv"

        if not stats_f.exists() or overwrite:
            data = z_score(x).join(y[[attr, "patient_code"]]).dropna(subset=[attr])

            # # GLM
            _res_glm = list()
            for feat in tqdm(x.columns, desc="feature", position=1):
                mdf = smf.glm(f"{feat} ~ {attr}", data).fit()
                res = (
                    mdf.params.to_frame("coefs")
                    .join(mdf.pvalues.rename("pvalues"))
                    .assign(feature=feat)
                )
                res = res.loc[res.index.str.contains(attr)]
                _res_glm.append(res)
            res_glm = pd.concat(_res_glm)
            res_glm["qvalues"] = pg.multicomp(res_glm["pvalues"].values, method="fdr_bh")[1]

            # # # Mixed effect
            _res_mlm = list()
            for feat in tqdm(x.columns, desc="feature", position=2):
                mdf = smf.mixedlm(
                    f"{feat} ~ {attr}",
                    data,
                    groups=data["patient_code"],
                ).fit()
                res = (
                    mdf.params.to_frame("coefs")
                    .join(mdf.pvalues.rename("pvalues"))
                    .assign(feature=feat)
                )
                res = res.loc[res.index.str.contains(attr)]
                _res_mlm.append(res)
            res_mlm = pd.concat(_res_mlm)
            res_mlm["qvalues"] = pg.multicomp(res_mlm["pvalues"].values, method="fdr_bh")[1]

            res = pd.concat([res_mlm.assign(model="mlm"), res_glm.assign(model="glm")])
            res = res.rename_axis(index="contrast")
            res.to_csv(stats_f)

    all_stats = pd.concat(
        [
            pd.read_csv(
                output_dir / f"supervised.{attr}.model_fits.csv",
                index_col=0,
            ).assign(attr=attr)
            for attr in nums
        ]
    )

    coef_mat = all_stats.reset_index().pivot_table(index="feature", columns="index", values="coefs")

    grid = clustermap(coef_mat, config="abs", cmap="RdBu_r", center=0, xticklabels=True)

    for attribute in nums:
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
            output_dir / f"supervised.{attribute}.Mixed_effect_models.clustermap.top_50.svg",
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
            output_dir / f"supervised.{attribute}.Mixed_effect_models.clustermap.top_50.sorted.svg",
            **figkws,
        )


def supervised_joint(
    x: DataFrame,
    y: DataFrame,
    attributes: tp.Sequence[str],
    plot_all: bool = True,
    overwrite: bool = False,
) -> None:
    """A single model of disease severity adjusted for various confounders."""
    import statsmodels.formula.api as smf

    output_dir = (results_dir / "supervised").mkdir()

    # # convert ordinal categories to numeric
    for attr in ["WHO_score_sample", "WHO_score_patient"]:
        y[attr] = y[attr].cat.codes.astype(float).replace(-1, np.nan)
    for attr in attributes:
        if y[attr].dtype.name == "Int64":
            y[attr] = y[attr].astype(float)
    cats = list(filter(lambda w: ~is_numeric(y[w]), attributes))
    nums = list(filter(lambda w: is_numeric(y[w]), attributes))

    # Use also a MLM and compare to GLM
    stats_f = output_dir / "supervised.joint_model.model_fits.csv"

    attrs = ["age", "race", "bmi", "WHO_score_sample"]
    model_str = "{} ~ " + " + ".join(attrs)

    if not stats_f.exists() or overwrite:
        data = z_score(x).join(y[attrs + ["patient_code"]]).dropna(subset=attrs)

        # # GLM
        _res_glm = list()
        for feat in tqdm(x.columns, desc="feature", position=1):
            mdf = smf.glm(model_str.format(feat), data).fit()
            res = (
                mdf.params.to_frame("coefs")
                .join(mdf.conf_int().rename(columns={0: "ci_l", 1: "ci_u"}))
                .join(mdf.pvalues.rename("pvalues"))
                .assign(feature=feat)
            )
            # res = res.loc[res.index.str.contains(attr)]
            _res_glm.append(res)
        res_glm = pd.concat(_res_glm)
        res_glm["qvalues"] = pg.multicomp(res_glm["pvalues"].values, method="fdr_bh")[1]

        # # # Mixed effect
        _res_mlm = list()
        for feat in tqdm(x.columns, desc="feature", position=2):
            mdf = smf.mixedlm(model_str.format(feat), data, groups=data["patient_code"]).fit()
            res = (
                mdf.params.to_frame("coefs")
                .join(mdf.pvalues.rename("pvalues"))
                .join(mdf.conf_int().rename(columns={0: "ci_l", 1: "ci_u"}))
                .assign(feature=feat)
            )
            # res = res.loc[res.index.str.contains(attr)]
            _res_mlm.append(res)
        res_mlm = pd.concat(_res_mlm)
        res_mlm["qvalues"] = pg.multicomp(res_mlm["pvalues"].values, method="fdr_bh")[1]

        res = pd.concat([res_mlm.assign(model="mlm"), res_glm.assign(model="glm")])
        res = res.rename_axis(index="contrast")
        res.to_csv(stats_f)

    # Plot
    attribute = "WHO_score_sample"
    res = pd.read_csv(stats_f, index_col=0).loc[attribute]
    assert np.allclose((res["ci_l"] - res["coefs"]).abs(), (res["ci_u"] - res["coefs"]).abs())
    res["ci"] = res["coefs"] - res["ci_l"]

    output_prefix = output_dir / f"supervised.joint_model.{attribute}."

    res_glm = res.query("model == 'glm'").set_index("feature")
    res_mlm = res.query("model == 'mlm'").set_index("feature")

    cglm = res_glm["coefs"].rename("GLM")
    cmlm = res_mlm["coefs"].rename("MLM")
    c = cglm.to_frame().join(cmlm)
    pglm = res_glm["pvalues"].rename("GLM")
    pmlm = res_mlm["pvalues"].rename("MLM")
    p = pglm.to_frame().join(pmlm)
    q = p.copy()
    for col in q:
        q[col] = pg.multicomp(q[col].values, method="fdr_bh")[1]

    # Compare models
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
        output_prefix + "General_vs_Mixed_effect_model_comparison.scatter.svg",
        **figkws,
    )

    # Plot all variables as rank vs change plot
    # # check error is symmetric
    var = get_nmr_feature_annotations().reindex(c.index)
    cat = var["group"].astype(pd.CategoricalDtype())
    cmap = sns.color_palette("tab20")
    score = (-np.log10(res_mlm["qvalues"])) * (res_mlm["coefs"] > 0).astype(int).replace(0, -1)
    ci = res_mlm["ci"].rename("MLM")
    qmlm = res_mlm["qvalues"]
    n_top = 10

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 3 * 2))
    for ax in fig.axes:
        ax.axhline(0, linestyle="--", color="grey")
    feats = (
        score.sort_values().head(n_top).index.tolist()
        + score.sort_values().tail(n_top).index.tolist()
    )
    for ax, crit, text in zip(fig.axes, [ci.index, feats], [False, True]):
        rank = score.loc[crit].rank()
        for i, group in enumerate(cat.unique()):
            sel = (cat == group) & cat.index.to_series().rename("group").isin(rank.index)
            # ax.errorbar(
            #     rank.loc[sel],
            #     score.loc[sel],
            #     fmt="o",
            #     yerr=ci.loc[sel],
            #     color=cmap[i],
            #     alpha=0.2,
            # )
            f = sel[sel].index
            ax.scatter(
                rank.loc[f],
                score.loc[f],
                color=cmap[i],
                s=10 + 2.5 ** -np.log10(qmlm.loc[f]),
                alpha=0.5,
                label=group,
            )
            if text:
                for idx in rank.loc[f].index:
                    ax.text(rank.loc[idx], score.loc[idx], s=idx, rotation=90, ha="center")
        v = (score.abs() + res_mlm["ci"]).max()
        v += v * 0.1
        ax.set(
            title=attribute,
            xlabel=r"Metabolites (ranked)",
            ylabel=r"Change with COVID-10 severity (signed -log10(p-value))",
            ylim=(-v, v),
        )
    ax0.legend(loc="upper left", bbox_to_anchor=(1, 1))
    from matplotlib.lines import Line2D

    p = qmlm.min()
    s0 = Line2D([0], [0], marker="o", label="1.0 (max)", markersize=np.sqrt(10 + 1))
    s1 = Line2D(
        [0],
        [0],
        marker="o",
        label=f"{p:.3e} (min)",
        markersize=np.sqrt(10 + 2.5 ** -np.log10(p)),
    )
    ax1.legend(handles=[s0, s1], title="FDR", loc="upper left", bbox_to_anchor=(1, 0))
    ax1.axvline((ax1.get_xlim()[1] - ax1.get_xlim()[0]) / 2, linestyle="--", color="grey")
    ax0.axvline(n_top, linestyle="--", color="grey")
    ax0.axvline(score.shape[0] - n_top, linestyle="--", color="grey")
    fig.savefig(
        output_prefix + "rank_vs_change.scatter.svg",
        **figkws,
    )

    # now plot top variables
    # # Add healthy (UKbb) range
    try:
        exp = pd.read_csv(metadata_dir / "ukbb.feature_range.csv", index_col=0)
    except FileNotFoundError:
        exp = pd.DataFrame(index=x.columns, columns=["mean", "ci_lower", "ci_upper"])

    alpha = 0.05
    y[attribute] = y[attribute].astype(pd.CategoricalDtype(ordered=True))
    feats = res_mlm.query(f"qvalues < {alpha}").sort_values("pvalues").index
    var = get_nmr_feature_annotations()
    if y[attribute].dtype.name in ["object", "category"]:
        fig = swarmboxenplot(
            data=x.join(y),
            x=attribute,
            y=feats,
            plot_kws=dict(palette=palettes.get(attribute)),
            test=False,
        )
        for feat, ax in zip(feats, fig.axes):
            ax.set(ylabel=var.loc[feat, "unit"])

            # Add healthy (UKbb) range
            base = exp.reindex([feat]).squeeze()
            if base.isnull().all():
                continue
            ax.axhline(base["mean"], linestyle="--", color="grey", linewidth=0.5)
            px = y[attribute].cat.categories
            px = [px.min() - 1] + y[attribute].cat.categories.tolist() + [px.max() + 1]
            n = len(px)
            for i in range(1, 4):
                py1 = base["mean"] - base["std"] * i
                py2 = base["mean"] + base["std"] * i
                ax.fill_between(
                    px,
                    [py1] * n,
                    [py2] * n,
                    color="grey",
                    alpha=0.1,
                    zorder=-100 - i,
                    edgecolor=None,
                )
        fig.savefig(
            output_prefix + "top_differential-Mixed_effect_models.swarmboxenplot.svg",
            **figkws,
        )
    else:
        from imc.graphics import get_grid_dims

        fig = get_grid_dims(feats, return_fig=True, sharex=True)
        for feat, ax in zip(feats, fig.axes):
            sns.regplot(data=x.join(y), x=attribute, y=feat, ax=ax)
            xl, xr = y[attribute].apply([min, max])
            xl = min(-1, 0)
            ax.set(
                title=feat + f"; FDR = {p.loc[feat, 'MLM']:.2e}",
                ylabel=var.loc[feat, "unit"],
                xlim=(xl - xl * 0.1, xr + xr * 0.1),
            )
        fig.savefig(
            output_prefix + "top_differential-Mixed_effect_models.regplot.svg",
            **figkws,
        )

    # Plot volcano
    data = res_mlm.rename(
        columns={
            "feature": "Variable",
            "coefs": "hedges",
            "pvalues": "p-unc",
            "qvalues": "p-cor",
        }
    ).assign(A="Healthy", B="High COVID-19 severity")
    fig = volcano_plot(
        stats=data.reset_index(drop=True), diff_threshold=0.01, invert_direction=False
    )
    fig.savefig(
        output_prefix + "Mixed_effect_models.volcano_plot.svg",
        **figkws,
    )

    # Plot heatmap of differential only
    n_top = 100
    f = c.abs().sort_values("MLM").tail(n_top).index.drop_duplicates()
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
    grid.ax_heatmap.set(xlabel=f"Features (top {n_top} features for '{attribute}'")
    grid.fig.savefig(
        output_prefix + f"Mixed_effect_models.clustermap.top_{n_top}.svg",
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
    grid.ax_heatmap.set(xlabel=f"Features (top {n_top} features for '{attribute}'")
    grid.fig.savefig(
        output_prefix + f"Mixed_effect_models.clustermap.top_{n_top}.sorted.svg",
        **figkws,
    )


def supervised_temporal(
    x: DataFrame,
    y: DataFrame,
    attributes: tp.Sequence[str],
    plot_all: bool = True,
    overwrite: bool = False,
) -> None:
    """A single model of disease severity adjusted for various confounders."""
    import statsmodels.formula.api as smf

    output_dir = (results_dir / "supervised").mkdir()

    test_vars = ["days_since_intubation_start", "days_since_tosilizumab_start"]

    # Use also a MLM and compare to GLM
    stats_f = output_dir / "supervised.temporal.model_fits.csv"
    attrs = ["age", "race", "bmi", "WHO_score_sample"]

    if not stats_f.exists() or overwrite:
        model_str = "{} ~ " + " + ".join(attrs + test_vars)

        data = z_score(x).join(y[attrs + test_vars]).dropna(subset=attrs + test_vars)

        # # convert ordinal categories to numeric
        for col in data.columns:
            if data[col].dtype.name == "category":
                data[col] = data[col].cat.codes.astype(float).replace(-1, np.nan)
            if data[col].dtype.name == "Int64":
                data[col] = data[col].astype(float)

        assert data["WHO_score_sample"].dtype.name != "category"

        # # GLM
        _res_glm = list()
        for feat in tqdm(x.columns, desc="feature", position=1):
            mdf = smf.glm(model_str.format(feat), data).fit()
            res = (
                mdf.params.to_frame("coefs")
                .join(mdf.conf_int().rename(columns={0: "ci_l", 1: "ci_u"}))
                .join(mdf.pvalues.rename("pvalues"))
                .assign(feature=feat)
            )
            _res_glm.append(res)
        res_glm = pd.concat(_res_glm)
        res_glm["qvalues"] = pg.multicomp(res_glm["pvalues"].values, method="fdr_bh")[1]

        res = res_glm.assign(model="glm")
        res = res.rename_axis(index="contrast")
        res.to_csv(stats_f)

    # Plot
    attribute = "days_since_tosilizumab_start"
    res = pd.read_csv(stats_f, index_col=0).loc[attribute]
    assert np.allclose((res["ci_l"] - res["coefs"]).abs(), (res["ci_u"] - res["coefs"]).abs())
    res["ci"] = res["coefs"] - res["ci_l"]

    output_prefix = output_dir / f"supervised.temporal.{attribute}."
    res_glm = res.query("model == 'glm'").set_index("feature")

    # Plot all variables as rank vs change plot
    # # check error is symmetric
    var = get_nmr_feature_annotations()
    cat = var["group"].astype(pd.CategoricalDtype())
    cmap = sns.color_palette("tab20")
    score = (-np.log10(res_glm["qvalues"])) * (res_glm["coefs"] > 0).astype(int).replace(0, -1)
    ci = res_glm["ci"].rename("GLM")
    cglm = res_glm["coefs"].rename("GLM")
    pglm = res_glm["pvalues"]
    qglm = res_glm["qvalues"]
    n_top = 10

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 3 * 2))
    for ax in fig.axes:
        ax.axhline(0, linestyle="--", color="grey")
    feats = (
        score.sort_values().head(n_top).index.tolist()
        + score.sort_values().tail(n_top).index.tolist()
    )
    for ax, crit, text in zip(fig.axes, [ci.index, feats], [False, True]):
        rank = score.loc[crit].rank()
        for i, group in enumerate(cat.unique()):
            sel = (cat == group) & cat.index.to_series().rename("group").isin(rank.index)
            # ax.errorbar(
            #     rank.loc[sel],
            #     score.loc[sel],
            #     fmt="o",
            #     yerr=ci.loc[sel],
            #     color=cmap[i],
            #     alpha=0.2,
            # )
            f = sel[sel].index
            ax.scatter(
                rank.loc[f],
                score.loc[f],
                color=cmap[i],
                s=10 + 2.5 ** -np.log10(qglm.loc[f]),
                alpha=0.5,
                label=group,
            )
            if text:
                for idx in rank.loc[f].index:
                    ax.text(rank.loc[idx], score.loc[idx], s=idx, rotation=90, ha="center")
        v = (score.abs() + res_glm["ci"]).max()
        v += v * 0.1
        ax.set(
            title=attribute,
            xlabel=r"Metabolites (ranked)",
            ylabel=r"Change with COVID-10 severity (signed -log10(p-value))",
            ylim=(-v, v),
        )
    ax0.legend(loc="upper left", bbox_to_anchor=(1, 1))
    from matplotlib.lines import Line2D

    p = qglm.min()
    s0 = Line2D([0], [0], marker="o", label="1.0 (max)", markersize=np.sqrt(10 + 1))
    s1 = Line2D(
        [0],
        [0],
        marker="o",
        label=f"{p:.3e} (min)",
        markersize=np.sqrt(10 + 2.5 ** -np.log10(p)),
    )
    ax1.legend(handles=[s0, s1], title="FDR", loc="upper left", bbox_to_anchor=(1, 0))
    ax1.axvline((ax1.get_xlim()[1] - ax1.get_xlim()[0]) / 2, linestyle="--", color="grey")
    ax0.axvline(n_top, linestyle="--", color="grey")
    ax0.axvline(score.shape[0] - n_top, linestyle="--", color="grey")
    fig.savefig(
        output_prefix + "rank_vs_change.scatter.svg",
        **figkws,
    )

    # Plot volcano
    data = (
        res_glm.reset_index()
        .rename(
            columns={
                "feature": "Variable",
                "coefs": "hedges",
                "pvalues": "p-unc",
                "qvalues": "p-cor",
            }
        )
        .assign(A="Healthy", B="High COVID-19 severity")
    )
    fig = volcano_plot(
        stats=data.reset_index(drop=True), diff_threshold=0.01, invert_direction=False
    )
    fig.savefig(
        output_prefix + "GLM.volcano_plot.svg",
        **figkws,
    )

    # Plot heatmap of differential only
    n_top = 100
    f = cglm.abs().sort_values().tail(n_top).index.drop_duplicates()
    f = cglm.loc[f].sort_values().index.drop_duplicates()
    stats = (
        (-np.log10(pglm.to_frame("-log10(p-value)")))
        .join(cglm.rename("Coefficient"))
        .join((qglm < 0.05).to_frame("Significant"))
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
    grid.ax_heatmap.set(xlabel=f"Features (top {n_top} features for '{attribute}'")
    grid.fig.savefig(
        output_prefix + f"GLM.clustermap.top_{n_top}.svg",
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
    grid.ax_heatmap.set(xlabel=f"Features (top {n_top} features for '{attribute}'")
    grid.fig.savefig(
        output_prefix + f"GLM.clustermap.top_{n_top}.sorted.svg",
        **figkws,
    )


def score_signature(x: DataFrame, diff: Series) -> Series:
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
        (xz[up].mean(axis=1) * (float(up.size) / n))
        - (xz[down].mean(axis=1) * (float(down.size) / n))
    ).rename("signature_score")
    return scores


def feature_enrichment() -> None:
    output_dir = (results_dir / "supervised").mkdir()

    stats_f = output_dir / "supervised.joint_model.model_fits.csv"
    stats = pd.read_csv(stats_f, index_col=0).query("model == 'mlm'")
    var = get_nmr_feature_annotations()
    groups = ["metagroup", "group", "subgroup", "lipid_class", "lipid_density", "lipid_size"]

    # Using PAGE
    _res = list()
    libraries = var_to_dict(var[groups])
    for contrast in stats.index.drop("Intercept").unique():
        s = stats.loc[contrast].set_index("feature")
        enr = page(s["coefs"], libraries)
        _res.append(enr.assign(contrast=contrast))
    res = pd.concat(_res)
    res["q"] = pg.multicomp(res["p"].values, method="fdr_bh")[1]
    res.to_csv(output_dir / "enrichment.page.csv")

    # # Plot
    res = pd.read_csv(output_dir / "enrichment.page.csv", index_col=0)
    p = res.query("contrast == 'WHO_score_sample'")
    p = p.rename(columns={"term": "Variable", "Z": "hedges", "p": "p-unc", "q": "p-cor"})
    p = p.groupby("Variable").mean().reset_index()
    fig = volcano_plot(
        p.assign(A="Healthy", B="High COVID-19 severity"),
        diff_threshold=None,
        n_top=15,
        invert_direction=False,
    )
    fig.savefig(output_dir / "enrichment.page.volcano_plot.svg", **figkws)

    # Using overrepresentation test
    _res = list()
    for contrast in stats.index.drop("Intercept").unique():
        s = stats.loc[contrast].set_index("feature")
        enr = enrich_categorical(s, var[groups], directional=True)
        _res.append(enr.assign(contrast=contrast))
    res = pd.concat(_res)
    res.to_csv(output_dir / "enrichment.chi2_independence.csv", index=False)

    # # Plot
    res = pd.read_csv(output_dir / "enrichment.chi2_independence.csv")
    p = res.query("contrast == 'WHO_score_sample' & test == 'log-likelihood'").sort_values("pval")
    p = p.rename(
        columns={"attribute_item": "Variable", "cramer": "hedges", "pval": "p-unc", "qval": "p-cor"}
    )
    p = p.groupby("Variable").mean().reset_index()
    fig = volcano_plot(
        p.assign(A="Healthy", B="High COVID-19 severity"),
        diff_threshold=None,
        n_top=15,
        invert_direction=False,
    )
    fig.savefig(output_dir / "enrichment.chi2_independence.volcano_plot.svg", **figkws)


def enrich_categorical(
    data: DataFrame, var: DataFrame, diff_threshold: float = 0.05, directional: bool = False
) -> DataFrame:
    data = data.copy()
    data["diff"] = False
    sig = data["qvalues"] < diff_threshold
    data.loc[sig, "diff"] = True
    if directional:
        direction = data["coefs"] > 0
        data.loc[sig & direction, "direction"] = "up"
        data.loc[sig & ~direction, "direction"] = "down"
    else:
        data["direction"] = "all"

    if data["diff"].sum() == 0:
        print("No significant features.")
        return pd.DataFrame()

    _res = list()
    for col in var:
        _res2 = list()
        for item in var[col].unique():
            for direction in data["direction"].unique():
                data["in_item"] = (var[col] == item) & (data["direction"] == direction)
                _, _, stats = pg.chi2_independence(data=data, x="in_item", y="diff")
                _res2.append(stats.assign(direction=direction, attribute=col, attribute_item=item))
        res2 = pd.concat(_res2)
        res2["qval"] = pg.multicomp(res2["pval"].values, method="fdr_bh")[1]
        _res.append(res2)
    res = pd.concat(_res)
    return res


def var_to_dict(var: DataFrame) -> tp.Dict:
    gsl = dict()
    for col in var:
        gs = dict()
        for val in var[col].dropna().unique():
            gs[val] = var.loc[var[col] == val].index.tolist()
        gsl[col] = gs
    return gsl


def page(parameter_vector: Series, gene_set_libraries: tp.Dict) -> DataFrame:
    from scipy import stats

    results = dict()
    for lib in gene_set_libraries:
        for gs, genes in gene_set_libraries[lib].items():
            μ = parameter_vector.mean()
            δ = parameter_vector.std()
            Sm = parameter_vector.reindex(genes).mean()
            m = parameter_vector.shape[0]
            # Get Z-scores
            Z = (Sm - μ) * (m ** (1 / 2)) / δ
            # Get P-values
            p = stats.norm.sf(abs(Z)) * 2
            results[(lib, gs)] = [μ, δ, Sm, m, Z, p]
    return (
        pd.DataFrame(results, index=["μ", "δ", "Sm", "m", "Z", "p"])
        .T.sort_values("p")
        .rename_axis(index=["database", "term"])
    )


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


def _load_model(model, filename):
    h5 = h5py.File(filename, "r")
    for key, value in h5.attrs.items():
        setattr(model, key, value)
    for di in range(len(h5.keys())):
        ds = "dataset%d" % di
        for key, value in h5[ds].items():
            if di == 0:
                setattr(model, key, [])
            model.__getattribute__(key).append(value[()])
    h5.close()


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
    _load_model(ccaCV, model_f)

    n_comp = ccaCV.best_numCC
    reg = ccaCV.best_reg
    print(n_comp, reg)
    x1_cca = pd.DataFrame(ccaCV.comps[0], index=x1.index)
    x2_cca = pd.DataFrame(ccaCV.comps[1], index=x2.index)

    o = output_dir / f"rCCA_integration.CV.{n_comp}.{reg}"
    metrics, anndata = assess_integration(
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
            "intubated",
            "alive",
            "WHO_score_sample",
            "patient_code",
        ],
        metrics=["silhouette_score", "anova"],
        plot=True,
        return_anndata=True,
        algo_kwargs=dict(umap=dict(gamma=25)),
        plt_kwargs=dict(s=100),
        cmaps=["Set1", "Set2", "Dark2", "Dark2", "Dark2", "inferno", None],
    )
    metrics.to_csv(output_dir / "rCCA_integration.metrics.csv")

    # Checkout the clusters from clustering on CCA-space
    import scanpy as sc

    sc.tl.leiden(anndata, 0.5)
    fig = sc.pl.umap(anndata, color=anndata.obs.columns, show=False)[0].figure
    fig.savefig(output_dir / "rCCA_integration.leiden_clustering_on_CCA_space.UMAP.svg", **figkws)

    # Plot in light of attributes
    x_comb = pd.concat(
        [
            pd.DataFrame(x1_cca.values, index=x1_cca.index + "_NMR"),
            pd.DataFrame(x2_cca.values, index=x2_cca.index + "_Flow"),
        ]
    )
    y_comb = pd.concat([y.assign(dataset="NMR"), y.assign(dataset="flow_cytometry")])
    y_comb["dataset"] = y_comb["dataset"].astype(pd.CategoricalDtype())
    y_comb.index = x_comb.index

    grid = clustermap(x_comb.T.corr(), cmap="RdBu_r", center=0, rasterized=True)
    grid.fig.savefig(output_dir / "rCCA_integration.sample_correlation.clustermap.svg", **figkws)
    from seaborn_extensions.annotated_clustermap import plot_attribute_heatmap

    q = palettes.copy()
    q["dataset"] = sns.color_palette("tab10", 2)
    fig = plot_attribute_heatmap(
        y=y_comb.iloc[grid.dendrogram_row.reordered_ind],
        attributes=attributes + ["dataset"],
        palettes=q,
        cmaps=cmaps,
    )
    fig.savefig(
        output_dir / "rCCA_integration.sample_correlation.clustermap.attributes.svg", **figkws
    )

    # Let's investigate the new clusters of patients (stratification)
    from scipy.cluster.hierarchy import fcluster

    clust = fcluster(grid.dendrogram_col.linkage, t=6, criterion="maxclust")
    order = [5, 4, 3, 6, 1, 2]
    labels = ["Healthy", "Mild2", "Mild1", "Severe2", "Severe1", "Convalescent"]
    clust = pd.Series(clust, index=x_comb.index).replace(dict(zip(order, labels)))
    cat = pd.Categorical(clust, ordered=True, categories=labels)
    clust = pd.Series(cat, index=clust.index, name="cluster")

    # # check clustering is correct
    grid2 = clustermap(x_comb.T.corr(), cmap="RdBu_r", center=0, rasterized=True, row_colors=clust)
    grid2.fig.savefig(
        output_dir / "rCCA_integration.sample_correlation.clustermap.clusters_labeled.svg", **figkws
    )
    # # See how clinical parameters are different between groups
    rem = [
        "patient_code",
        "Medium_ethanol",
        "Isopropyl_alcohol",
        "Low_glucose",
        "High_lactate",
        "High_pyruvate",
        "Gluconolactone",
        "Low_protein",
        "Below_limit_of_quantification",
        "sample_under_intubation",
        "days_since_tosilizumab_start",
    ]

    # # # simple mean
    y_diff = y_comb.groupby(clust).mean().T.dropna().drop(rem).astype(float)
    y_diff = y_diff.loc[y_diff.sum(1) > 0]
    heat = clustermap(y_diff, z_score=0, col_cluster=False, center=0, cmap="RdBu_r", figsize=(8, 5))
    heat.savefig(
        output_dir / "rCCA_integration.sample_clustering.clinical_parameters.mean.svg",
        **figkws,
    )
    heat = clustermap(
        y_diff,
        z_score=0,
        row_cluster=False,
        col_cluster=False,
        center=0,
        cmap="RdBu_r",
        figsize=(8, 5),
    )
    heat.savefig(
        output_dir / "rCCA_integration.sample_clustering.clinical_parameters.mean.sorted.svg",
        **figkws,
    )

    # # # with LMs
    import statsmodels.formula.api as smf

    cats = y_comb.columns[y_comb.dtypes == "category"].tolist()
    cats = [c for c in cats if y_comb[c].cat.ordered]
    d = y_comb[y_diff.index.tolist()].astype(float).join(y_comb[cats].apply(lambda x: x.cat.codes))

    _res_glm = list()
    for feat in tqdm(d.columns, desc="feature", position=1):
        mdf = smf.glm(f"{feat} ~ cluster - 1", d.join(clust)).fit()
        res = mdf.params.to_frame("coefs").join(mdf.pvalues.rename("pvalues")).assign(feature=feat)
        _res_glm.append(res)
    res_glm = pd.concat(_res_glm).rename_axis(index="group")
    res_glm["qvalues"] = pg.multicomp(res_glm["pvalues"].values, method="fdr_bh")[1]
    res_glm["group"] = res_glm.index.str.split("[").map(lambda x: x[1].replace("]", ""))
    coefs = res_glm.reset_index(drop=True).pivot_table(
        index="feature", columns="group", values="coefs"
    )[clust.cat.categories]

    coefs = coefs.drop(["patient_group", "WHO_score_patient"])
    heat = clustermap(coefs, z_score=0, col_cluster=False, center=0, cmap="PuOr_r", figsize=(8, 5))
    heat.savefig(
        output_dir
        / "rCCA_integration.sample_clustering.clinical_parameters.regression_coefficients.svg",
        **figkws,
    )
    heat = clustermap(
        coefs,
        z_score=0,
        row_cluster=False,
        col_cluster=False,
        center=0,
        cmap="PuOr_r",
        figsize=(8, 5),
    )
    heat.savefig(
        output_dir
        / "rCCA_integration.sample_clustering.clinical_parameters.regression_coefficients.sorted.svg",
        **figkws,
    )

    # # See how immune-metabolic parameters are different between groups
    x1i = xz1.copy()
    x1i.index = x1i.index + "_NMR"

    # # # quick-and-dirty sum difference of means
    means1 = x1i.groupby(clust.reindex(x1i.index)).mean()
    rank1 = (means1 - means1.mean(0)).abs().sum().sort_values()
    x2i = xz2.copy()
    x2i.index = x2i.index + "_Flow"
    means2 = x2i.groupby(clust.reindex(x2i.index)).mean()
    rank2 = (means2 - means2.mean(0)).abs().sum().sort_values()

    n_top = 15
    v1, v2 = rank1.tail(n_top).index.tolist(), rank2.tail(n_top).index.tolist()
    ce1 = x1i[v1].groupby(clust.reindex(x1i.index)).mean()
    ce2 = x2i[v2].groupby(clust.reindex(x2i.index)).mean()
    e = ce1.T.append(ce2.T)
    heat = clustermap(
        e,
        cmap="RdBu_r",
        metric="correlation",
        col_cluster=False,
        center=0,
        figsize=(8, 5),
        z_score=1,
    )
    heat.savefig(
        output_dir
        / f"rCCA_integration.sample_clustering.top_{n_top}_varying_features.mean_difference.svg",
        **figkws,
    )

    # # # with LMs
    import statsmodels.formula.api as smf

    n_top = 3
    x1i = xz1.copy()
    x1i.index = x1i.index + "_Flow"
    _res_glm1 = list()
    for feat in tqdm(x1i.columns, desc="feature", position=1):
        mdf = smf.glm(f"{feat} ~ cluster - 1", x1i.join(clust)).fit()
        res = mdf.params.to_frame("coefs").join(mdf.pvalues.rename("pvalues")).assign(feature=feat)
        _res_glm1.append(res)
    res_glm1 = pd.concat(_res_glm1).rename_axis(index="group")
    res_glm1["qvalues"] = pg.multicomp(res_glm1["pvalues"].values, method="fdr_bh")[1]
    v1 = (
        res_glm1.reset_index()
        .set_index("feature")
        .groupby("group")["pvalues"]
        .nsmallest(n_top)
        .index.get_level_values(1)
        .unique()
    )

    x2i = xz2.copy()
    x2i.index = x2i.index + "_Flow"
    _res_glm2 = list()
    mods = {r"+": "__p__", r"-": "__m__", r"/": "__sla__", r"(": "__OP__", r")": "__CL__"}
    for k, v in mods.items():
        x2i.columns = x2i.columns.str.replace(k, v, regex=False)
    for feat in tqdm(x2i.columns, desc="feature", position=1):
        mdf = smf.glm(f"{feat} ~ cluster - 1", x2i.join(clust)).fit()
        res = mdf.params.to_frame("coefs").join(mdf.pvalues.rename("pvalues")).assign(feature=feat)
        _res_glm2.append(res)
    res_glm2 = pd.concat(_res_glm2).rename_axis(index="group")
    res_glm2["qvalues"] = pg.multicomp(res_glm2["pvalues"].values, method="fdr_bh")[1]
    v2 = (
        res_glm2.reset_index()
        .set_index("feature")
        .groupby("group")["pvalues"]
        .nsmallest(n_top)
        .index.get_level_values(1)
        .unique()
    )
    for k, v in mods.items():
        v2 = v2.str.replace(v, k, regex=False)
    x2i = xz2.copy()
    x2i.index = x2i.index + "_Flow"

    ce1 = x1i[v1].groupby(clust.reindex(x1i.index)).mean()
    ce2 = x2i[v2].groupby(clust.reindex(x2i.index)).mean()
    e = ce1.T.append(ce2.T)
    # e = e.sort_values(e.columns.tolist(), ascending=False)
    # heat = clustermap(e, cmap="RdBu_r", row_cluster=False, col_cluster=False, center=0, figsize=(8, 5), z_score=0)
    heat = clustermap(
        e,
        cmap="RdBu_r",
        metric="correlation",
        col_cluster=False,
        center=0,
        figsize=(8, 5),
        z_score=1,
    )
    heat.savefig(
        output_dir
        / f"rCCA_integration.sample_clustering.top_{n_top}_varying_features_per_cluster.regression.svg",
        **figkws,
    )

    # fig = plot_projection(
    #     x_comb, y_comb, factors=attributes, algo_name="CCA", n_dims=5, palettes=palettes
    # )
    # fig.savefig(
    #     output_dir / "rCCA_integration.all_CCs.attributes.scatter.svg",
    #     **figkws,
    # )
    # plt.close(fig)

    # # Predict other data type for each one
    # w1 = pd.DataFrame(ccaCV.ws[0], index=x1.columns)
    # w2 = pd.DataFrame(ccaCV.ws[1], index=x2.columns)

    # n = 10
    # f1 = w1[0].abs().sort_values().tail(n + 1).index[::-1]
    # f2 = w2[0].abs().sort_values().tail(n + 1).index[::-1]

    # fig, axes = plt.subplots(2, n, figsize=(3 * n, 3 * 2))
    # for rank in range(n):
    #     axes[0, rank].scatter(x1_cca[0], x1[f1[rank]], alpha=0.5, s=5)
    #     axes[0, rank].set(xlabel="CCA1", ylabel=f1[rank])
    #     axes[1, rank].scatter(x2_cca[0], x2[f2[rank]], alpha=0.5, s=5)
    #     axes[1, rank].set(xlabel="CCA1", ylabel=f2[rank])
    # fig.savefig(
    #     output_dir / "rCCA_integration.CC1.top_variables.scatter.svg",
    #     **figkws,
    # )
    # plt.close(fig)

    # o1 = z_score(x1_cca @ w2.T)
    # o2 = z_score(x2_cca @ w1.T)

    # unsupervised(
    #     o1,
    #     y,
    #     attributes=attributes,
    #     data_type="flow_cytometry",
    #     suffix="_predicted_from_NMR",
    # )
    # unsupervised(
    #     o2,
    #     y,
    #     attributes=attributes,
    #     data_type="NMR",
    #     suffix="_predicted_from_flow_cytometry",
    # )


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
    metrics: tp.Sequence[str] = ["silhouette_score"],
    subsample: str = None,
    plot: bool = True,
    return_anndata: bool = False,
    algo_kwargs: tp.Dict[str, tp.Dict[str, tp.Any]] = None,
    plt_kwargs: tp.Dict[str, tp.Any] = None,
    cmaps: tp.Sequence[str] = None,
) -> tp.Union[DataFrame, tp.Tuple[DataFrame, AnnData]]:
    """Keyword arguments are passed to the scanpy plotting function."""
    from sklearn.metrics import silhouette_score
    from imc.graphics import rasterize_scanpy
    from anndata import AnnData
    import scanpy as sc

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
                sel_cells = adata.obs.sample(frac=float(subsample.split("frac=")[1])).index
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
        adata = adata[pd.concat([adata.obs[attr].isin(sel[attr]) for attr in attrs], 1).all(1), :]

    scores: tp.Dict[str, tp.Dict[str, tp.Dict[str, float]]] = dict()
    for i, algo in enumerate(algos):
        scores[algo] = dict()
        for j, attr in enumerate(attributes):
            scores[algo][attr] = dict()
            if "silhouette_score" in metrics:
                scores[algo][attr]["silhouette_score"] = silhouette_score(
                    adata.obsm[f"X_{algo}"], adata.obs[attr]
                )
            if "anova" in metrics:
                d = pd.DataFrame(adata.obsm[f"X_{algo}"], index=adata.obs.index).join(
                    adata.obs[attr]
                )
                res = pg.anova(data=d, dv=0, between=attr).squeeze()
                scores[algo][attr]["anova"] = res["p-unc"]
    scores_df = pd.concat(
        [pd.DataFrame(v).assign(projection=k) for k, v in scores.items()]
    ).rename_axis(index="metric")

    to_return = scores_df if not return_anndata else (scores_df, adata)

    if not plot:
        return to_return

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
                components="1,3" if algo == "cca" else "1,2",
                alpha=0.5,
                show=False,
                **plt_kwargs,
                **cmap_kws,
                ax=ax,
            )

            s = scores_df.query(f"projection == '{algo}'")[attr]
            s = "; ".join([f"{k}: {v:.3e}" for k, v in s.items()])
            ax.set(
                title=f"{attr}\n{s}",
                xlabel=algo.upper() + "1",
                ylabel=algo.upper() + "2",
            )
    rasterize_scanpy(fig)
    fig.savefig(tp.cast(Path, output_prefix) + ".joint_datasets.svg", **figkws)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(attributes), figsize=(3 * len(attributes), 1))
    for j, (ax, attr) in enumerate(zip(axes, attributes)):
        cmap = sns.color_palette(cmaps[j], len(adata.obs[attr].cat.categories) + 1)
        for i, s in enumerate(adata.obs[attr].cat.categories):
            p = adata.obs.query(f"{attr} == '{s}'")
            p = adata[p.index].obsm["X_cca"][:, 0]
            sns.kdeplot(p, color=cmap[i], ax=ax, cumulative=True)
            ax.text(p.mean(), 0.5, s=s, color=cmap[i])
    fig.savefig(tp.cast(Path, output_prefix) + ".joint_datasets.kde_distribution.svg", **figkws)
    plt.close(fig)
    return to_return


def predict_outcomes() -> None:
    from joblib import Parallel, delayed
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.linear_model import LogisticRegression, ElasticNet
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC, NuSVC
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.model_selection import cross_validate

    def fit(_, model, X, y, k=None):
        from collections import defaultdict

        feature_attr = defaultdict(lambda: None)
        feature_attr.update(
            {
                RandomForestClassifier: "feature_importances_",
                LogisticRegression: "coef_",
                ElasticNet: "coef_",
                LinearSVC: "coef_",
            }
        )
        kws = dict(cv=10, scoring="roc_auc", return_train_score=True, return_estimator=True)

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
        coefs = (np.nan, np.nan)
        if feat is not None:
            coefs = tuple()
            for out in [out1, out2]:
                co = pd.DataFrame(
                    [
                        pd.Series(
                            getattr(c["classifier"], feat).squeeze(),
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

    y["first_sample"] = 0
    for pat in y["patient_code"].unique():
        s = y.loc[y["patient_code"] == pat].sort_values("date_sample").index[0]
        y.loc[s, "first_sample"] = 1

    # Use only mild-severe patients
    y.loc[y["WHO_score_patient"] <= 4, "patient_severity"] = "low"
    y.loc[y["WHO_score_patient"] > 4, "patient_severity"] = "high"
    y["patient_severity"] = pd.Categorical(
        y["patient_severity"], ordered=True, categories=["low", "high"]
    )
    # m = y.query("patient_group.isin(['moderate', 'severe']).values")
    # m = y.query("patient_group.isin(['moderate', 'severe']).values & first_sample == 1")
    m = y.query("patient_severity.isin(['low', 'high']).values & first_sample == 1")

    # Convert classes to binary
    # target = m["patient_group"].cat.remove_unused_categories().cat.codes
    target = m["patient_severity"].cat.remove_unused_categories().cat.codes

    # Align
    xz1 = xz1.reindex(target.index)
    xz2 = xz2.reindex(target.index)

    # # For the other classifiers
    N = 100

    insts = [
        RandomForestClassifier,
        LogisticRegression,
        LinearSVC,
        ElasticNet,
        NuSVC,
        QuadraticDiscriminantAnalysis,
    ]
    predict_options = [
        ("", None),
        (".feature_selection_k8", 8),
        (".feature_selection_k6", 6),
        (".feature_selection_k5", 5),
        (".feature_selection_k4", 4),
        (".feature_selection_k3", 3),
        (".feature_selection_k2", 2),
        (".feature_selection_k1", 1),
    ]

    for model in insts:
        for label, k in predict_options:
            # for dtype, X in [("combined", xz1.join(xz2))]:
            for dtype, X in [("NMR", xz1), ("flow_cytometry", xz2), ("combined", xz1.join(xz2))]:
                name = str(type(model())).split(".")[-1][:-2]
                print(name, label, dtype)

                # Fit
                res = Parallel(n_jobs=-1)(
                    delayed(fit)(i, model=model, X=X, y=target, k=k) for i in range(N)
                )
                # Get ROC_AUC scores
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
                    # / f"severe-mild_prediction.first_sample_only.{dtype}.{name}{label}.scores.csv"
                    / f"severity_scale_prediction.first_sample_only.{dtype}.{name}{label}.scores.csv"
                )
                # Get coefficients/variable ranks
                real_coefs = pd.DataFrame([r[-2] for r in res])
                random_coefs = pd.DataFrame([r[-1] for r in res])
                coefs = (
                    real_coefs.assign(type="real")
                    .append(random_coefs.assign(type="random"))
                    .fillna(0)
                    .rename_axis(index="iteration")
                )
                coefs.to_csv(
                    output_dir
                    # / f"severe-mild_prediction.first_sample_only.{dtype}.{name}{label}.coefs.csv"
                    / f"severity_scale_prediction.first_sample_only.{dtype}.{name}{label}.coefs.csv"
                )

    for model in insts:
        # Only with full model
        label = ""
        k = None
        name = str(type(model())).split(".")[-1][:-2]
        _res = list()
        for dtype, X in [("NMR", xz1), ("flow_cytometry", xz2), ("combined", xz1.join(xz2))]:
            scores = pd.read_csv(
                output_dir
                # / f"severe-mild_prediction.first_sample_only.{dtype}.{name}{label}.scores.csv",
                / f"severity_scale_prediction.first_sample_only.{dtype}.{name}{label}.scores.csv",
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
            # / f"severe-mild_prediction.first_sample_only.performance_data_type_comparison.{name}{label}.scores.svg",
            / f"severity_scale_prediction.first_sample_only.performance_data_type_comparison.{name}{label}.scores.svg",
            **figkws,
        )

        p = res.query("variable == 'test_score'")
        fig, _ = swarmboxenplot(data=p, x="dataset", y="value")
        fig.axes[0].axhline(0.5, linestyle="--", color="grey")

        s = p.groupby("dataset")["value"].mean().squeeze()
        for i, q in enumerate(s.index):
            fig.axes[0].text(i, 0.8, s=f"{s[q]:.3f}", ha="center")
        fig.savefig(
            output_dir
            # / f"severe-mild_prediction.first_sample_only.performance_data_type_comparison.{name}{label}.scores.only_real.svg",
            / f"severity_scale_prediction.first_sample_only.performance_data_type_comparison.{name}{label}.scores.only_real.svg",
            **figkws,
        )

        # Look into performance (de)crease with less predictors
        _perf = list()
        for label, k in predict_options:
            p = pd.read_csv(
                output_dir
                / f"severity_scale_prediction.first_sample_only.{dtype}.{name}{label}.scores.csv",
                index_col=0,
            )
            p = p.loc[:, p.columns.str.contains("test")].melt()
            _perf.append(p.assign(n_features=k))
        perf = pd.concat(_perf).fillna(x1.shape[1])
        perf["n_features"] = pd.Categorical(perf["n_features"], ordered=True)

        fig, stats = swarmboxenplot(data=perf, x="variable", y="value", hue="n_features")
        fig.savefig(
            output_dir
            / f"severity_scale_prediction.first_sample_only.performance_data_type_comparison.{name}{label}.scores.decreasing_n_variables.svg",
            **figkws,
        )

        fig, stats = swarmboxenplot(
            data=perf.query("~variable.str.contains('random').values"), x="n_features", y="value"
        )
        fig.savefig(
            output_dir
            / f"severity_scale_prediction.first_sample_only.performance_data_type_comparison.{name}{label}.scores.decreasing_n_variables.only_real.svg",
            **figkws,
        )

        # Look at the top variables
        _coefs = list()
        for label, k in predict_options:
            c = pd.read_csv(
                output_dir
                / f"severity_scale_prediction.first_sample_only.{dtype}.{name}{label}.coefs.csv",
                index_col=0,
            )
            c2 = (c.T.drop("type") / c.sum(1)).T
            c2["type"] = c["type"]
            _coefs.append(c2.assign(n_features=k))

        coefs = pd.concat(_coefs).fillna(0)
        r = (
            coefs.groupby(["n_features", "type"])
            .mean()
            .groupby(level=1)
            .mean()
            .T.sort_values("real")
            * 100
        )
        r["log_ratio"] = np.log(r["real"] / r["random"])

        fig, axes = plt.subplots(1, 2, figsize=(2 * 4.2, 4))
        for ax in axes:
            ax.scatter(r["log_ratio"], r["real"], s=2, alpha=0.5)
            ax.set(xlabel="Log ratio (real/randomized)", ylabel="Importance")
        axes[1].set(yscale="log")
        fig.savefig(
            output_dir
            / f"severity_scale_prediction.first_sample_only.performance_data_type_comparison.{name}{label}.variable_importance.rank_vs_score.svg",
            **figkws,
        )

        rr = r.tail(20)
        fig, ax = plt.subplots(1, 1, figsize=(2 * 4.2, 4))
        rank = rr["real"].rank(ascending=False)
        ax.scatter(rank, rr["real"], s=5)
        for idx in rr.index:
            ax.text(rank[idx], rr.loc[idx, "real"], s=idx, rotation=90, ha="right", va="top")
        ax.set(xlabel="Importance (rank)", ylabel="Importance", yscale="log")
        fig.savefig(
            output_dir
            / f"severity_scale_prediction.first_sample_only.performance_data_type_comparison.{name}{label}.variable_importance.rank_vs_score.top_20.svg",
            **figkws,
        )

        # Use this score to relate to clinical parameters


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
