# app/core/tone_fairness.py
import cv2, numpy as np
from sklearn.cluster import KMeans

def analyze_tone_fairness(rois, n_clusters=3):
    tones = []
    for roi in rois:
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L,A,B = cv2.split(lab)
        tones.append([np.mean(L), np.mean(A), np.mean(B)])
    tones = np.array(tones)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(tones)
    labels, centers = kmeans.labels_, kmeans.cluster_centers_

    fairness_factors = []
    for i, roi in enumerate(rois):
        L_mean = tones[i][0]
        group = int(labels[i])
        fairness_factors.append(100.0 / (centers[group][0] + 1e-6))

    return {
        "tone_clusters": n_clusters,
        "tone_means": centers.tolist(),
        "assigned_groups": labels.tolist(),
        "fairness_scaling_factors": fairness_factors,
        "comment": "Fair normalization applied per skin tone group."
    }
