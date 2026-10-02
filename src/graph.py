
                                                                             
                                                          
                                                                             
 
                                                                            
                                                                         
                                    
 
           
from pathlib import Path
                    
                                 
                                                                      
                                             
                                                                             

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

                                                                             
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

                                                                                
COLOURS = {
    "precision": "#2166ac",        
    "recall": "#d6604d",         
    "f1": "#4dac26",         
}

                                                                      
                                                                 
MODEL_RESULTS_HUMAN = []
MODEL_RESULTS_CANDIDATE = []
HUMAN_COMPARISON_RESULTS = []
FIGURE_DIR = Path("figure")


def set_figure_dir(path):
    global FIGURE_DIR
    FIGURE_DIR = Path(path)


def reset_results():
    MODEL_RESULTS_HUMAN.clear()
    MODEL_RESULTS_CANDIDATE.clear()
    HUMAN_COMPARISON_RESULTS.clear()


def figure_path(filename):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR / filename


                                                                             
              
                                                                             


def add_model_result_human(
    model_name,
    precision,
    recall,
    f1,
    report_dict,
    runtime=None,
    y_true=None,
    y_pred=None,
):
    MODEL_RESULTS_HUMAN.append(
        {
            "model": model_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "runtime": runtime,
            "report": report_dict,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def add_model_result_candidate(
    model_name,
    precision,
    recall,
    f1,
    report_dict,
    runtime=None,
    y_true=None,
    y_pred=None,
):
    MODEL_RESULTS_CANDIDATE.append(
        {
            "model": model_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "runtime": runtime,
            "report": report_dict,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def add_human_comparison_result(
    model_name,
    precision,
    recall,
    f1,
    report_dict,
    runtime=None,
    y_true=None,
    y_pred=None,
):
    HUMAN_COMPARISON_RESULTS.append(
        {
            "model": model_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "runtime": runtime,
            "report": report_dict,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def ensure_figure_dir():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


                                                                             
                                     
                                                                             


def plot_models_vs_human_gold():
    base_models = ["scispaCy", "BioBERT", "PubMedBERT + OpenMed", "ClinicalBERT", "D4Data NER (DistilBERT)"]
    results = [r for r in MODEL_RESULTS_HUMAN if r["model"] in base_models]

    if not results:
        print("No base model results for Figure 1.")
        return

    results = sorted(results, key=lambda r: base_models.index(r["model"]))

    models = [r["model"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1_scores = [r["f1"] for r in results]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_p = ax.bar(
        x - width,
        precisions,
        width,
        label="Precision",
        color=COLOURS["precision"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_r = ax.bar(
        x,
        recalls,
        width,
        label="Recall",
        color=COLOURS["recall"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_f = ax.bar(
        x + width,
        f1_scores,
        width,
        label="F1-score",
        color=COLOURS["f1"],
        edgecolor="white",
        linewidth=0.5,
    )

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Exact-Span Model Performance Against Human Gold (BC5CDR)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = figure_path("fig1_models_vs_human_gold.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


                                                                             
                                                                    
                                                                             


def plot_candidate_gold_comparison():
    if not HUMAN_COMPARISON_RESULTS:
        print("No human comparison results for Figure 2.")
        return

    results = HUMAN_COMPARISON_RESULTS
    labels = [r["model"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1_scores = [r["f1"] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_p = ax.bar(
        x - width,
        precisions,
        width,
        label="Precision",
        color=COLOURS["precision"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_r = ax.bar(
        x,
        recalls,
        width,
        label="Recall",
        color=COLOURS["recall"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_f = ax.bar(
        x + width,
        f1_scores,
        width,
        label="F1-score",
        color=COLOURS["f1"],
        edgecolor="white",
        linewidth=0.5,
    )

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Majority Candidate Gold", "Weighted Candidate Gold"], rotation=0
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Exact-Span Pseudo-Gold Quality Against Human Gold (BC5CDR)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = figure_path("fig2_candidate_gold_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


                                                                             
                                                                  
                                                                             


def plot_per_label_performance():
    base_models = ["scispaCy", "BioBERT", "PubMedBERT + OpenMed", "ClinicalBERT", "D4Data NER (DistilBERT)"]
    results = [r for r in MODEL_RESULTS_HUMAN if r["model"] in base_models]

    if not results:
        print("No results for Figure 3.")
        return

    results = sorted(results, key=lambda r: base_models.index(r["model"]))
    models = [r["model"] for r in results]
    disease_f1 = [r["report"].get("DISEASE", {}).get("f1-score", 0) for r in results]
    chemical_f1 = [r["report"].get("CHEMICAL", {}).get("f1-score", 0) for r in results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_d = ax.bar(
        x - width / 2,
        disease_f1,
        width,
        label="DISEASE",
        color="#b2182b",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_c = ax.bar(
        x + width / 2,
        chemical_f1,
        width,
        label="CHEMICAL",
        color="#2166ac",
        edgecolor="white",
        linewidth=0.5,
    )

    for bars in [bars_d, bars_c]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Per-Label F1-score Against Human Gold (BC5CDR)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = figure_path("fig3_per_label_f1.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


                                                                             
                                                   
                                                                             


def plot_scispacy_confusion_matrix():
    results = [r for r in MODEL_RESULTS_HUMAN if r["model"] == "scispaCy"]

    if not results or results[0]["y_true"] is None:
        print("No scispaCy data for Figure 4.")
        return

    r = results[0]
    labels = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]
    cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)

    ax.set_title("scispaCy — Confusion Matrix vs Human Gold (BC5CDR)")
    plt.tight_layout()
    out = figure_path("fig4_scispacy_confusion_matrix.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


                                                                             
                                                                  
                                                                             


def plot_weighted_gold_confusion_matrix():
    results = [r for r in HUMAN_COMPARISON_RESULTS if "Weighted" in r["model"]]

    if not results or results[0]["y_true"] is None:
        print("No weighted candidate gold data for Figure 5.")
        return

    r = results[0]
    labels = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]
    cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)

    ax.set_title("Weighted Candidate Gold — Confusion Matrix vs Human Gold (BC5CDR)")
    plt.tight_layout()
    out = figure_path("fig5_weighted_gold_confusion_matrix.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


                                                                             
                                                              
                                                                             


def plot_all_model_confusion_matrices():
    base_models = ["scispaCy", "BioBERT", "PubMedBERT + OpenMed", "ClinicalBERT", "D4Data NER (DistilBERT)"]
    results = [r for r in MODEL_RESULTS_HUMAN if r["model"] in base_models]

    if not results:
        print("No base model results for Figure 6 (confusion grid).")
        return

    results = sorted(results, key=lambda r: base_models.index(r["model"]))
    labels = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]
    short_labels = ["O", "B-DIS", "I-DIS", "B-CHEM", "I-CHEM"]

    n = len(results)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
    axes = axes.flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        if r["y_true"] is None or r["y_pred"] is None:
            ax.set_visible(False)
            continue

        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_labels)
        disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)

        ax.set_title(r["model"], fontsize=12, pad=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

                                                                      
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Confusion Matrices vs Human Gold (BC5CDR) — All Models",
        fontsize=14,
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = figure_path("fig6_all_model_confusion_matrices.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


                                                                             
                                                                    
                                                                             


def plot_separate_model_confusion_matrices():
    base_models = ["scispaCy", "BioBERT", "PubMedBERT + OpenMed", "ClinicalBERT", "D4Data NER (DistilBERT)"]
    results = [r for r in MODEL_RESULTS_HUMAN if r["model"] in base_models]

    if not results:
        print("No base model results for separate confusion matrices.")
        return

    labels = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]
    confusion_dir = figure_path("confusion")
    confusion_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        if r["y_true"] is None or r["y_pred"] is None:
            print(f"No y_true / y_pred for {r['model']} — skipping.")
            continue

        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)

        ax.set_title(f"{r['model']} — Confusion Matrix vs Human Gold (BC5CDR)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        safe_name = r["model"].lower().replace(" ", "_")
        out = confusion_dir / f"cm_{safe_name}.png"
        plt.savefig(out)
        plt.close()
        print(f"Saved: {out}")


                                                                             
             
                                                                             


def plot_all():
    ensure_figure_dir()

    plot_models_vs_human_gold()
    plot_candidate_gold_comparison()
    plot_per_label_performance()
    plot_scispacy_confusion_matrix()
    plot_weighted_gold_confusion_matrix()
    plot_all_model_confusion_matrices()
    plot_separate_model_confusion_matrices()
