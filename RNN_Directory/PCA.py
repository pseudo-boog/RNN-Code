import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SKPCA
import matplotlib.pyplot as plt
from collections import defaultdict
from RNN_Module import get_data_filtered

# --- helpers ---
def _resample_to_length(arr_2col, L=100):
    T = arr_2col.shape[0]
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, L)
    A_new = np.interp(x_new, x_old, arr_2col[:, 0])
    P_new = np.interp(x_new, x_old, arr_2col[:, 1])
    return np.stack([A_new, P_new], axis=1)  # [L, 2]


def _flatten_AP(series_resampled):
    return series_resampled.reshape(-1)


# --- main PCA runner ---
def run_pca_on_timeseries(
        mech_list=None,
        start_iteration=0,
        end_iteration=5000,
        L=100,
        n_components=2,
        standardize=True,
        plot=True,
        save_scores_csv=None
    ):
    if mech_list is None:
        # (label, prefix)
        mech_list = [("1000", 'M1_cd'), ("0100", 'M1_pd'), ("0010", 'M1_sd'), ("0001", 'M1_n')]

    data_list, labels_list = get_data_filtered(mech_list, start_iteration, end_iteration)

    # Resample and flatten each series
    vectors = []
    for t in data_list:
        arr = t.cpu().numpy()  # [T,2]
        arrL = _resample_to_length(arr, L=L)        # [L,2]
        vec  = _flatten_AP(arrL)                    # [2L]
        vectors.append(vec)

    X = np.stack(vectors, axis=0)  # [N, 2L]

    # Standardize across the whole dataset (zero mean, unit variance) — recommended before PCA
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # PCA
    pca = SKPCA(n_components=n_components, svd_solver="full", random_state=0)
    scores = pca.fit_transform(X)                # [N, n_components]
    loadings = pca.components_.T                 # [2L, n_components]
    evr = pca.explained_variance_ratio_          # [n_components]

    print(f"PCA explained variance ratio: {np.round(evr, 4)} (sum={evr.sum():.4f})")

    # Optional: save scores with labels
    if save_scores_csv is not None:
        out_df = pd.DataFrame(scores, columns=[f"PC{k+1}" for k in range(n_components)])
        out_df["label"] = labels_list
        out_df.to_csv(save_scores_csv, index=False)
        print(f"Saved PCA scores to: {save_scores_csv}")

    # Plot PC1 vs PC2 (if available)
    if plot and n_components >= 2:
        # deterministic color per label
        unique_labels = sorted(set(labels_list))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        colors = [label_to_idx[lab] for lab in labels_list]

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(scores[:, 0], scores[:, 1], c=colors, alpha=0.8)
        # make a legend mapping color->label
        handles = []
        for lab, idx in label_to_idx.items():
            handles.append(plt.Line2D([0], [0], marker='o', linestyle='', label=lab))
        plt.legend(handles=handles, title="Label", loc="best", frameon=True)
        plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
        plt.title("PCA of (A,P) time series (resampled & flattened)")
        plt.tight_layout()
        plt.show()

    return {
        "X_mat": X,
        "labels": labels_list,
        "scaler": scaler,
        "pca_model": pca,
        "scores": scores,
        "loadings": loadings,
        "explained_variance_ratio": evr,
        "resample_length": L
    }

# --- convenience wrapper matching your starting idea ---
def pca_(n_components=2, L=100):
    mech_list = [("10000", 'M1_cd'), ("01000", 'M1_pd'), ("00100", 'M1_sd'), ("00010", 'M1_n'),  ('00001', 'M1_si')]
    return run_pca_on_timeseries(mech_list=mech_list, start_iteration=0, end_iteration=5000,
                                 L=L, n_components=n_components, standardize=True, plot=True)

pca_(2, 100)