import pandas as pd
import numpy as np

def handle_missing_and_weights(features, labels=None):
    # fill null trong feat = 0
    features_filled = features.fillna(0)

    sample_weight = None

    if labels is not None:
        value_counts = labels.value_counts(normalize=True)
        n_classes = value_counts.shape[0]
        if n_classes == 2:
            # scale_pos_weight = số mẫu âm / số mẫu dương
            neg = (labels == value_counts.index[0]).sum()
            pos = (labels == value_counts.index[1]).sum()
            scale_pos_weight = neg / (pos + 1e-8)
            return features_filled, scale_pos_weight
    return features_filled, None

