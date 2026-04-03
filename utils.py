import numpy as np

def aggregate_scores(window_preds):

    mus = np.array([v[0] for v in window_preds.values()])
    p_up = np.array([v[1] for v in window_preds.values()])

    mu_mean = mus.mean()
    confidence = p_up.mean()

    return mu_mean * confidence
