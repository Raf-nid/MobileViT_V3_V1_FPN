# -*- coding: utf-8 -*-
"""
MoE expert usage analysis

This script:
1) Loads ``ExpertUsage_Summary.mat`` produced by the MoE evaluation pipeline
2) Scans the same directory for ``*_Amp.mat`` / ``*_Rec.mat`` pairs
3) Computes per-sample FMC metrics (NCC, MAE, MSE)
4) Merges routing statistics with FMC metrics
5) Generates plots (usage by condition, heatmaps/boxplots, optional PCA/t-SNE, performance)

Usage:
    1. Set ``MAT_PATH`` below to your ``ExpertUsage_Summary.mat`` file
    2. Run: ``python analyze_experts.py``
"""

from __future__ import print_function

import datetime
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required. Install with: pip install pandas")

# Optional sklearn for PCA / t-SNE
_SKLEARN_OK = True
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
except ImportError:
    _SKLEARN_OK = False

# =============================================================================
# =============================================================================
# >>>  MODIFIER UNIQUEMENT CETTE LIGNE AVEC LE CHEMIN VERS VOTRE .mat  <<<
# =============================================================================
# =============================================================================
MAT_PATH = r"Evaluation_MbViT_XXS4_FF8_MF_Normalisation_NW_W_20251217_125228_125_epochs/Matlab_MbViT_XXS4_FF8_MF_Normalisation_NW_W/ExpertUsage_Summary.mat"
MAT_PATH = "Evaluation_MbViT_XXS4_FF8_MF_Valid_Normalisation_NW_W_20260107_130757_400_epochs/Matlab_MbViT_XXS4_FF8_MF_Valid_Normalisation_NW_W/ExpertUsage_Summary.mat"
# =============================================================================
# =============================================================================

# -------------------------
# Condition parsing
# -------------------------
_FREQ_PAT = re.compile(r"(5MHz|75MHz|225MHz)", re.IGNORECASE)
# wedge rules:
# - if name contains "NW" -> no wedge
# - else if contains "wedge" or "_W_" or "W_" or token "W" -> wedge
# - else -> no wedge
_WEDGE_WORD = re.compile(r"wedge", re.IGNORECASE)
_W_TOKEN = re.compile(r"(^|[^A-Za-z0-9])W([^A-Za-z0-9]|$)", re.IGNORECASE)
_W_UNDERSCORE = re.compile(r"_W_", re.IGNORECASE)
_W_PREFIX = re.compile(r"W_", re.IGNORECASE)
_NW = re.compile(r"NW", re.IGNORECASE)


def parse_conditions(name):
    m = _FREQ_PAT.search(name)
    if m:
        freq = m.group(1)
        freq = freq.replace("mhz", "MHz").replace("MHZ", "MHz")
    else:
        freq = "UNK"

    if _NW.search(name):
        wedge = 0
    else:
        wedge = 1 if (_WEDGE_WORD.search(name) or _W_UNDERSCORE.search(name) or _W_PREFIX.search(name) or _W_TOKEN.search(name)) else 0

    return freq, int(wedge)


# -------------------------
# .mat utilities
# -------------------------
def _to_str_list(mat_cell):
    if mat_cell is None:
        return []
    arr = np.array(mat_cell).squeeze()
    out = []
    for x in arr:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="ignore"))
        elif isinstance(x, np.ndarray):
            if x.dtype.kind in ("U", "S"):
                try:
                    out.append(("".join(x.tolist())).strip())
                except Exception:
                    out.append(str(x))
            else:
                try:
                    flat = x.flatten().tolist()
                    out.append(("".join([chr(int(v)) for v in flat])).strip())
                except Exception:
                    out.append(str(x))
        else:
            out.append(str(x))
    return out


def _scalar_int(x, default_value):
    if x is None:
        return int(default_value)
    try:
        return int(np.array(x).reshape(-1)[0])
    except Exception:
        return int(default_value)


def _first_numeric_array(mat_dict):
    best = None
    best_size = -1
    for k, v in mat_dict.items():
        if str(k).startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in ("f", "i", "u"):
            if v.size > best_size:
                best = v
                best_size = v.size
    return best


def load_expert_summary(mat_path):
    data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    for k in list(data.keys()):
        if str(k).startswith("__"):
            del data[k]

    sample_names = _to_str_list(data.get("sample_names"))
    expert_usage = np.array(data.get("expert_usage"))
    expert_fraction = np.array(data.get("expert_fraction"))

    expert_entropy = np.array(data.get("expert_entropy"))
    expert_entropy = expert_entropy.reshape(-1) if expert_entropy.size else expert_entropy

    expert_variance = np.array(data.get("expert_variance"))
    expert_variance = expert_variance.reshape(-1) if expert_variance.size else expert_variance

    default_experts = expert_fraction.shape[1] if expert_fraction.ndim == 2 else 0
    num_experts = _scalar_int(data.get("num_experts"), default_experts)
    num_layers = _scalar_int(data.get("num_layers"), 0)
    num_samples = _scalar_int(data.get("num_samples"), len(sample_names))

    return {
        "mat_path": str(mat_path),
        "sample_names": sample_names,
        "expert_usage": expert_usage,
        "expert_fraction": expert_fraction,
        "expert_entropy": expert_entropy,
        "expert_variance": expert_variance,
        "num_samples": num_samples,
        "num_experts": num_experts,
        "num_layers": num_layers,
        "gate_probs_per_layer": data.get("gate_probs_per_layer", None),
        "topk_values_per_layer": data.get("topk_values_per_layer", None),
        "expert_sequences": data.get("expert_sequences", None),
        "raw": data,
    }


# -------------------------
# FMC metrics
# -------------------------
def compute_metrics(A, B):
    a = np.asarray(A).astype(np.float64).ravel()
    b = np.asarray(B).astype(np.float64).ravel()

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    a = a[mask]
    b = b[mask]

    diff = b - a
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))

    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = (np.sqrt(np.sum(a0 * a0)) * np.sqrt(np.sum(b0 * b0)) + 1e-12)
    ncc = float(np.sum(a0 * b0) / denom)

    return ncc, mae, mse


def scan_amp_rec_pairs(root_dir):
    root_dir = Path(root_dir)
    amp_files = list(root_dir.rglob("*_Amp.mat"))
    rec_files = list(root_dir.rglob("*_Rec.mat"))

    amp_map = {}
    for p in amp_files:
        base = p.name[:-len("_Amp.mat")]
        amp_map[base] = p

    rec_map = {}
    for p in rec_files:
        base = p.name[:-len("_Rec.mat")]
        rec_map[base] = p

    pairs = {}
    for base, ap in amp_map.items():
        rp = rec_map.get(base, None)
        if rp is not None:
            pairs[base] = (ap, rp)
    return pairs


def load_fmc_array(mat_path):
    d = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    arr = _first_numeric_array(d)
    return arr


# -------------------------
# Plot helpers
# -------------------------
def _savefig(path, dpi=220):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close()


def _heatmap(mat, xlabels, ylabels, out_path, title, cbar_label="Value"):
    w = max(7, int(0.9 * len(xlabels) + 2))
    h = max(4, int(0.45 * len(ylabels) + 2))
    plt.figure(figsize=(w, h))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, label=cbar_label)
    plt.xticks(range(len(xlabels)), xlabels, rotation=30, ha="right")
    plt.yticks(range(len(ylabels)), ylabels)
    plt.title(title)
    _savefig(out_path)


def _bar_mean_stacked(df, value_cols, group_cols, out_path, title):
    g = df.groupby(group_cols, dropna=False)[value_cols].mean().reset_index()
    xlabels = []
    for row in g[group_cols].itertuples(index=False, name=None):
        xlabels.append(" | ".join([str(v) for v in row]))

    plt.figure(figsize=(max(8, int(len(xlabels) * 1.2)), 4.8))
    bottom = np.zeros(len(g), dtype=float)
    for c in value_cols:
        plt.bar(xlabels, g[c].values, bottom=bottom, label=c)
        bottom = bottom + g[c].values
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean fraction")
    plt.title(title)
    plt.legend(ncols=min(6, len(value_cols)), fontsize=9)
    _savefig(out_path)


def _boxplot_with_points(df, ycol, group_col, out_path, title):
    order = sorted(df[group_col].dropna().unique().tolist())
    if len(order) == 0:
        return

    groups = []
    labels = []
    for k in order:
        vals = df[df[group_col] == k][ycol].values
        # Filtrer les NaN
        vals = vals[np.isfinite(vals)]
        groups.append(vals)
        labels.append(str(k))

    # Skip if all values are empty
    all_vals = np.concatenate([g for g in groups if len(g) > 0]) if any(len(g) > 0 for g in groups) else np.array([])
    if len(all_vals) == 0:
        return

    # ===== Figure 1: full boxplot over all values =====
    plt.figure(figsize=(max(8, int(len(labels) * 1.2)), 4.8))
    plt.boxplot(groups, labels=labels, showfliers=True)

    for i, vals in enumerate(groups):
        if len(vals) > 0:
            x = (i + 1) + 0.08 * (np.random.rand(len(vals)) - 0.5)
            plt.scatter(x, vals, s=10, alpha=0.55)

    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ycol)
    plt.title(title)
    _savefig(out_path)

    # ===== Figure 2: Boxplot zoome (sans outliers extremes) =====
    # Calculer les limites basees sur IQR global
    Q1 = np.percentile(all_vals, 25)
    Q3 = np.percentile(all_vals, 75)
    IQR = Q3 - Q1

    # Limites: Q1 - 1.5*IQR et Q3 + 1.5*IQR (definition standard des outliers)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Check for significant outliers
    n_outliers = np.sum((all_vals < lower_bound) | (all_vals > upper_bound))

    # Ne creer le zoom que s'il y a des outliers (>1% des donnees)
    if n_outliers > 0 and n_outliers / len(all_vals) > 0.005:
        # Add 10% margin for visualization
        margin = 0.1 * IQR if IQR > 0 else 0.1 * (upper_bound - lower_bound + 1e-6)
        ylim_low = lower_bound - margin
        ylim_high = upper_bound + margin

        plt.figure(figsize=(max(8, int(len(labels) * 1.2)), 4.8))
        plt.boxplot(groups, labels=labels, showfliers=False)  # zoomed view: hide fliers

        for i, vals in enumerate(groups):
            if len(vals) > 0:
                # only overlay points inside the zoomed y-range
                mask_in_range = (vals >= ylim_low) & (vals <= ylim_high)
                vals_in = vals[mask_in_range]
                if len(vals_in) > 0:
                    x = (i + 1) + 0.08 * (np.random.rand(len(vals_in)) - 0.5)
                    plt.scatter(x, vals_in, s=10, alpha=0.55)

        plt.ylim(ylim_low, ylim_high)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(ycol)
        plt.title(title + " (zoom, {} outliers exclus)".format(n_outliers))

        # Save with _zoomed suffix
        out_path = Path(out_path)
        zoomed_path = out_path.parent / (out_path.stem + "_zoomed" + out_path.suffix)
        _savefig(zoomed_path)


def _scatter(x, y, out_path, title, xlabel, ylabel):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.6, 5.0))
    plt.scatter(x, y, s=14, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    _savefig(out_path)


# -------------------------
# Build merged dataset
# -------------------------
def build_routing_df(summary):
    names = summary["sample_names"]
    frac = summary["expert_fraction"]
    usage = summary["expert_usage"]
    ent = summary["expert_entropy"]
    var = summary["expert_variance"]

    if frac.ndim != 2:
        raise ValueError("expert_fraction must be 2D [N,E].")

    n, e = frac.shape
    if len(names) != n:
        fixed = names[:n]
        while len(fixed) < n:
            fixed.append("sample_%d" % len(fixed))
        names = fixed

    rows = []
    for i in range(n):
        freq, wedge = parse_conditions(names[i])
        r = {
            "sample": names[i],
            "base": names[i],
            "freq": freq,
            "wedge": int(wedge),
            "cond": "%s_W%d" % (freq, int(wedge)),
            "entropy": float(ent[i]) if i < len(ent) else np.nan,
            "variance": float(var[i]) if i < len(var) else np.nan,
        }
        for ex in range(e):
            r["frac_e%d" % (ex + 1)] = float(frac[i, ex])
            if usage is not None and np.ndim(usage) == 2 and usage.shape == (n, e):
                r["usage_e%d" % (ex + 1)] = int(usage[i, ex])
        rows.append(r)

    df = pd.DataFrame(rows)
    frac_cols = [c for c in df.columns if c.startswith("frac_e")]
    df["dominant_expert"] = df[frac_cols].values.argmax(axis=1) + 1
    return df


def merge_fmc_metrics(df_routing, pairs):
    metrics_rows = []
    for base, (ap, rp) in pairs.items():
        try:
            A = load_fmc_array(ap)
            B = load_fmc_array(rp)
            ncc, mae, mse = compute_metrics(A, B)
        except Exception:
            ncc, mae, mse = np.nan, np.nan, np.nan

        freq, wedge = parse_conditions(base)
        metrics_rows.append({
            "base": base,
            "amp_path": str(ap),
            "rec_path": str(rp),
            "freq_from_fmc": freq,
            "wedge_from_fmc": int(wedge),
            "ncc": ncc,
            "mae": mae,
            "mse": mse,
        })

    df_m = pd.DataFrame(metrics_rows)

    df = df_routing.merge(df_m, how="left", left_on="sample", right_on="base", suffixes=("", "_m"))

    if "ncc" in df.columns:
        missing = df["ncc"].isna()
    else:
        missing = np.array([False] * len(df))

    if missing.any():
        bases = df_m["base"].tolist()
        for idx in df.index[missing].tolist():
            s = str(df.loc[idx, "sample"])
            found = None
            for b in bases:
                if (b in s) or (s in b):
                    found = b
                    break
            if found is not None:
                rowm = df_m[df_m["base"] == found].iloc[0]
                for k in ["base", "amp_path", "rec_path", "freq_from_fmc", "wedge_from_fmc", "ncc", "mae", "mse"]:
                    df.loc[idx, k] = rowm[k]

    if "freq_from_fmc" in df.columns:
        df["freq"] = df["freq"].where(df["freq"] != "UNK", df["freq_from_fmc"])
    if "wedge_from_fmc" in df.columns:
        df["wedge"] = df["wedge"].where(df["wedge"].isin([0, 1]), df["wedge_from_fmc"])
    df["cond"] = df.apply(lambda r: "%s_W%d" % (r["freq"], int(r["wedge"])), axis=1)

    return df


# -------------------------
# Per-layer routing extraction (best effort)
# -------------------------
def _safe_to_numeric_array(x):
    """Tente de convertir x en array numpy numerique. Retourne None si echec."""
    if x is None:
        return None
    try:
        arr = np.array(x)
        # Si c'est un array d'objets (nested), on ne peut pas le convertir directement
        if arr.dtype == object:
            return None
        return arr.astype(np.float64)
    except (ValueError, TypeError):
        return None


def extract_per_layer_fractions(summary, df_routing):
    """
    Extrait les fractions d'utilisation des experts par couche (best effort).
    Retourne un dict {layer_idx: array [N, E]} ou {} si non disponible.
    """
    N = len(df_routing)
    E = int(summary.get("num_experts", 0))
    out = {}

    if E <= 0 or N <= 0:
        return out

    # Methode 1: gate_probs_per_layer
    g = summary.get("gate_probs_per_layer", None)
    if g is not None:
        try:
            # g est typiquement un cell array MATLAB [num_samples, 1] ou [1, num_layers]
            g_arr = np.array(g, dtype=object).squeeze()
            if g_arr.ndim == 0:
                g_arr = np.array([g_arr.item()], dtype=object)

            for li in range(len(g_arr)):
                try:
                    layer_data = g_arr[li]
                    A = _safe_to_numeric_array(layer_data)
                    if A is not None:
                        if A.ndim == 2 and A.shape[0] == N and A.shape[1] == E:
                            out[li] = A
                        elif A.ndim == 3 and A.shape[0] == N:
                            # Moyenne sur la dimension des tokens
                            out[li] = np.mean(A, axis=1)
                except Exception:
                    continue

            if len(out) > 0:
                return out
        except Exception:
            pass

    # Methode 2: topk_values_per_layer (indices des experts selectionnes)
    topk = summary.get("topk_values_per_layer", None)
    if topk is not None:
        try:
            topk_arr = np.array(topk, dtype=object).squeeze()
            if topk_arr.ndim == 0:
                topk_arr = np.array([topk_arr.item()], dtype=object)

            for li in range(len(topk_arr)):
                try:
                    layer_data = topk_arr[li]
                    # Tenter de convertir en array d'entiers
                    idxs = _safe_to_numeric_array(layer_data)
                    if idxs is None:
                        continue
                    idxs = idxs.astype(int)

                    M = np.zeros((N, E), dtype=float)

                    if idxs.ndim == 1 and idxs.shape[0] == N:
                        # one expert per sample
                        for i in range(N):
                            e = int(idxs[i])
                            if 1 <= e <= E:
                                M[i, e - 1] = 1.0
                        out[li] = M
                    elif idxs.ndim == 2 and idxs.shape[0] == N:
                        # multiple experts per sample (top-k)
                        T = idxs.shape[1]
                        for i in range(N):
                            for t in range(T):
                                e = int(idxs[i, t])
                                if 1 <= e <= E:
                                    M[i, e - 1] += 1.0
                            if T > 0:
                                M[i, :] /= float(T)
                        out[li] = M
                except Exception:
                    continue

            if len(out) > 0:
                return out
        except Exception:
            pass

    # Method 3: expert_sequences (per-sample expert sequence)
    seqs = summary.get("expert_sequences", None)
    if seqs is not None:
        try:
            seq_arr = np.array(seqs, dtype=object).squeeze()
            if seq_arr.ndim == 0:
                seq_arr = np.array([seq_arr.item()], dtype=object)

            sequences = []
            for x in seq_arr:
                try:
                    s = np.array(x).astype(int).reshape(-1)
                    sequences.append(s)
                except Exception:
                    sequences.append(np.array([], dtype=int))

            if len(sequences) == 0:
                return out

            L = max([len(s) for s in sequences])
            if L > 0:
                for li in range(L):
                    M = np.zeros((N, E), dtype=float)
                    for i in range(min(N, len(sequences))):
                        s = sequences[i]
                        if li < len(s):
                            e = int(s[li])
                            if 1 <= e <= E:
                                M[i, e - 1] = 1.0
                    out[li] = M
            return out
        except Exception:
            pass

    return out


# -------------------------
# Analysis plots
# -------------------------
def make_routing_plots(df, outdir, tag):
    outdir = Path(outdir) / tag / "routing"
    outdir.mkdir(parents=True, exist_ok=True)

    frac_cols = [c for c in df.columns if c.startswith("frac_e")]
    _bar_mean_stacked(df, frac_cols, ["cond"], outdir / "mean_fraction_by_condition.png", "Mean expert fractions by condition")
    _bar_mean_stacked(df, frac_cols, ["freq"], outdir / "mean_fraction_by_freq.png", "Mean expert fractions by frequency")
    _bar_mean_stacked(df, frac_cols, ["wedge"], outdir / "mean_fraction_by_wedge.png", "Mean expert fractions wedge vs non-wedge")

    conds = sorted(df["cond"].unique().tolist())
    mats = []
    for cnd in conds:
        mats.append(df[df["cond"] == cnd][frac_cols].mean().values)
    mat = np.stack(mats, axis=0) if len(mats) else np.zeros((0, len(frac_cols)))
    _heatmap(mat, frac_cols, conds, outdir / "heatmap_cond_x_expert_mean_fraction.png", "Mean fractions: condition x expert")

    corr = np.corrcoef(df[frac_cols].values.T)
    _heatmap(corr, frac_cols, frac_cols, outdir / "corr_expert_fractions.png", "Correlation of expert fractions", cbar_label="Corr")

    _boxplot_with_points(df, "entropy", "cond", outdir / "entropy_by_condition.png", "Gate entropy by condition")
    _boxplot_with_points(df, "variance", "cond", outdir / "variance_by_condition.png", "Gate variance by condition")

    tab = pd.crosstab(df["cond"], df["dominant_expert"], normalize="index")
    tab.to_csv(outdir / "dominant_expert_distribution_by_condition.csv")
    _heatmap(tab.values,
             ["E%d" % int(c) for c in tab.columns.tolist()],
             [str(i) for i in tab.index.tolist()],
             outdir / "dominant_expert_distribution_heatmap.png",
             "Dominant expert distribution (normalized)")

    if _SKLEARN_OK and len(df) >= 3:
        embdir = outdir / "embedding"
        embdir.mkdir(parents=True, exist_ok=True)
        X = df[frac_cols].values.astype(np.float32)
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(5, Xs.shape[1]))
        Z = pca.fit_transform(Xs)

        plt.figure(figsize=(6.5, 4.2))
        xs = np.arange(1, len(pca.explained_variance_ratio_) + 1)
        plt.plot(xs, pca.explained_variance_ratio_, marker="o")
        plt.xlabel("Component")
        plt.ylabel("Explained variance ratio")
        plt.title("PCA explained variance")
        _savefig(embdir / "pca_variance_explained.png")

        uniq = sorted(df["cond"].unique().tolist())
        plt.figure(figsize=(7.2, 5.8))
        for u in uniq:
            m = (df["cond"].values == u)
            plt.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.85, label=u)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA PC1 vs PC2 (by condition)")
        plt.legend(fontsize=8, ncols=2)
        _savefig(embdir / "pca_pc1_pc2_by_condition.png")

        loadings = pca.components_[:2, :]
        _heatmap(loadings, frac_cols, ["PC1", "PC2"], embdir / "pca_loadings_pc1_pc2.png", "PCA loadings (PC1/PC2)")

        if len(df) <= 2000:
            perplex = min(30, max(5, int(len(df) / 20)))
            tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplex)
            T = tsne.fit_transform(Xs)
            plt.figure(figsize=(7.2, 5.8))
            for u in uniq:
                m = (df["cond"].values == u)
                plt.scatter(T[m, 0], T[m, 1], s=18, alpha=0.85, label=u)
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.title("t-SNE (fractions) by condition")
            plt.legend(fontsize=8, ncols=2)
            _savefig(embdir / "tsne_by_condition.png")


def make_performance_plots(df, outdir, tag):
    outdir = Path(outdir) / tag / "performance"
    outdir.mkdir(parents=True, exist_ok=True)

    for metric in ["ncc", "mae", "mse"]:
        if metric not in df.columns:
            continue
        _boxplot_with_points(df, metric, "freq", outdir / ("%s_by_freq.png" % metric), "%s by frequency" % metric.upper())
        _boxplot_with_points(df, metric, "wedge", outdir / ("%s_by_wedge.png" % metric), "%s by wedge (0 no, 1 yes)" % metric.upper())
        _boxplot_with_points(df, metric, "cond", outdir / ("%s_by_condition.png" % metric), "%s by condition" % metric.upper())
        _boxplot_with_points(df, metric, "dominant_expert", outdir / ("%s_by_dominant_expert.png" % metric), "%s by dominant expert" % metric.upper())

    frac_cols = [c for c in df.columns if c.startswith("frac_e")]
    for metric in ["ncc", "mae", "mse"]:
        if metric not in df.columns:
            continue
        for c in frac_cols:
            _scatter(df[c].values, df[metric].values,
                     outdir / "corr" / ("%s_vs_%s.png" % (c, metric)),
                     "%s vs %s" % (metric.upper(), c),
                     c, metric.upper())

    if "dominant_expert" in df.columns:
        for metric in ["ncc", "mae", "mse"]:
            if metric not in df.columns:
                continue
            tab = df.pivot_table(index="cond", columns="dominant_expert", values=metric, aggfunc="mean")
            tab.to_csv(outdir / ("mean_%s_by_cond_x_domexpert.csv" % metric))
            _heatmap(tab.values,
                     ["E%d" % int(c) for c in tab.columns.tolist()],
                     [str(i) for i in tab.index.tolist()],
                     outdir / ("heatmap_mean_%s_cond_x_domexpert.png" % metric),
                     "Mean %s: condition x dominant expert" % metric.upper(),
                     cbar_label=metric.upper())


def make_per_layer_plots(df, per_layer, outdir, tag):
    if not per_layer:
        return
    outdir = Path(outdir) / tag / "per_layer"
    outdir.mkdir(parents=True, exist_ok=True)

    conds = sorted(df["cond"].unique().tolist())
    layer_ids = sorted(per_layer.keys())
    E = per_layer[layer_ids[0]].shape[1]

    mat_overall = np.zeros((len(layer_ids), E), dtype=float)
    for i, li in enumerate(layer_ids):
        mat_overall[i, :] = np.mean(per_layer[li], axis=0)
    _heatmap(mat_overall,
             ["E%d" % (j + 1) for j in range(E)],
             ["L%d" % (li + 1) for li in layer_ids],
             outdir / "heatmap_layer_x_expert_overall.png",
             "Overall per-layer expert usage (mean fraction)")

    for cnd in conds:
        idx = df.index[df["cond"] == cnd].to_numpy()
        if len(idx) == 0:
            continue
        mat = np.zeros((len(layer_ids), E), dtype=float)
        for i, li in enumerate(layer_ids):
            mat[i, :] = np.mean(per_layer[li][idx, :], axis=0)
        _heatmap(mat,
                 ["E%d" % (j + 1) for j in range(E)],
                 ["L%d" % (li + 1) for li in layer_ids],
                 outdir / "by_condition" / ("heatmap_layer_x_expert_%s.png" % cnd),
                 "Per-layer expert usage (mean fraction) - %s" % cnd)

    for li in layer_ids:
        dom = per_layer[li].argmax(axis=1) + 1
        col = "dom_layer_%d" % (li + 1)
        df[col] = dom
        tab = pd.crosstab(df["cond"], df[col], normalize="index")
        tab.to_csv(outdir / ("dominant_distribution_%s.csv" % col))
        _heatmap(tab.values,
                 ["E%d" % int(c) for c in tab.columns.tolist()],
                 [str(i) for i in tab.index.tolist()],
                 outdir / ("dominant_distribution_%s.png" % col),
                 "Dominant expert distribution - %s" % col)


def main():
    # =========================================================================
    # Utilise MAT_PATH defini en haut du fichier
    # =========================================================================
    mat_path = Path(MAT_PATH)

    # Verification que le fichier existe
    if not mat_path.exists():
        print("=" * 60)
        print("ERROR: .mat file not found:")
        print(f"  {mat_path.resolve()}")
        print("")
        print("Update the MAT_PATH variable at the top of this script")
        print("to point to your ExpertUsage_Summary.mat file.")
        print("=" * 60)
        raise SystemExit(1)

    print("=" * 60)
    print("Loading file:", mat_path.resolve())
    print("=" * 60)

    # Output directory next to the .mat file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = mat_path.parent / f"ExpertAnalysis_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    # Load expert summary struct
    summ = load_expert_summary(mat_path)
    df_r = build_routing_df(summ)

    # Scan for Amp/Rec pairs in the same folder
    fmc_root = str(mat_path.parent)
    pairs = scan_amp_rec_pairs(fmc_root)
    print(f"Amp/Rec pairs found: {len(pairs)}")

    # Merge FMC metrics into routing dataframe
    df = merge_fmc_metrics(df_r, pairs)

    # Save merged CSV
    df.to_csv(outdir / "merged_routing_fmc.csv", index=False)
    print(f"CSV saved: {outdir / 'merged_routing_fmc.csv'}")

    # Routing plots
    print("Generating routing plots...")
    make_routing_plots(df, outdir, tag="analysis")

    # Performance plots
    print("Generating performance plots...")
    make_performance_plots(df, outdir, tag="analysis")

    # Per-layer fractions if available
    per_layer = extract_per_layer_fractions(summ, df_r)
    if per_layer:
        print("Generating per-layer plots...")
        make_per_layer_plots(df, per_layer, outdir, tag="analysis")

    # Save run metadata
    meta = {
        "mat_path": str(mat_path.resolve()),
        "num_samples": int(summ.get("num_samples", 0)),
        "num_experts": int(summ.get("num_experts", 0)),
        "num_layers": int(summ.get("num_layers", 0)),
        "fmc_root": fmc_root,
        "num_pairs_amp_rec": len(pairs),
        "num_rows_with_ncc": int(df["ncc"].notna().sum()) if "ncc" in df.columns else 0,
    }
    pd.DataFrame([meta]).to_csv(outdir / "run_meta.csv", index=False)

    # Resume final
    frac_cols = [c for c in df.columns if c.startswith("frac_e")]
    conds = sorted(df["cond"].unique().tolist())

    print("")
    print("=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Number of samples: {len(df)}")
    print(f"Conditions found: {conds}")
    print(f"Number of experts: {summ.get('num_experts', '?')}")
    if "ncc" in df.columns:
        print(f"Samples with FMC NCC metric: {df['ncc'].notna().sum()}")
    if len(frac_cols) > 0:
        print("")
        print("Mean expert usage by condition:")
        means = df.groupby(["freq", "wedge"])[frac_cols].mean()
        print(means.to_string())
    print("")
    print("=" * 60)
    print(f"DONE. Results in: {outdir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
