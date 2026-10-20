"""Charts for the incident report (matplotlib, Agg backend, figures always closed)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def probability_chart(packet, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    names = list(packet["probabilities_pct"].keys())
    vals = [packet["probabilities_pct"][n] for n in names]
    fig, ax = plt.subplots(figsize=(5, 3))
    try:
        ax.bar(names, vals, color=["#e45756" if n == packet["predicted_class"] else "#4c78a8" for n in names])
        ax.set_ylabel("Probability (%)"); ax.set_ylim(0, 100); ax.set_title("Class probabilities")
        fig.tight_layout(); fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)
    return str(path)


def feature_chart(packet, path, top_n=10):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    feats = packet["top_features"][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(6, 0.45 * max(len(feats), 3) + 1))
    try:
        ax.barh([f["feature"] for f in feats], [f["z"] for f in feats],
                color=["#e45756" if f["z"] > 0 else "#4c78a8" for f in feats])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("z-score vs Normal baseline"); ax.set_title("Top deviating features")
        fig.tight_layout(); fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)
    return str(path)
