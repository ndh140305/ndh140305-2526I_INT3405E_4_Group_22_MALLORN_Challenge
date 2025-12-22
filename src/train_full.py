
import pandas as pd
import numpy as np
from xgb_baseline import get_xgb_classifier
from data_processing import handle_missing_and_weights
try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    RandomOverSampler = None

features_path = 'features/full_train_features_labeled.csv'  
def train_full_data(features_path, oversample=True):
    df = pd.read_csv(features_path)
    X = df.drop(['object_id', 'target'], axis=1)
    y = df['target']
    # Thống kê nhãn
    print('Label distribution:')
    print(y.value_counts())
    print('\nFeature statistics:')
    print(X.describe())
    # Xử lý missing value và sample_weight nếu cần
    X_filled, sample_weight = handle_missing_and_weights(X, y)
    num_class = y.nunique()
    sw_train = None
    if sample_weight is not None:
        if np.isscalar(sample_weight):
            model = get_xgb_classifier(num_class=num_class)
            model.set_params(scale_pos_weight=sample_weight)
            sw_train = None
        else:
            model = get_xgb_classifier(num_class=num_class)
            sw_train = sample_weight.values
    else:
        model = get_xgb_classifier(num_class=num_class)
    # Oversample nếu cần
    if oversample and RandomOverSampler is not None:
        ros = RandomOverSampler(random_state=42)
        X_filled, y = ros.fit_resample(X_filled, y)
        if sw_train is not None:
            sw_train = sw_train[ros.sample_indices_]
        print(f"[INFO] Oversampled: {np.bincount(y) if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.integer) else pd.Series(y).value_counts().to_dict()}")
    elif oversample:
        print("[WARNING] imblearn chưa được cài đặt, không thể oversample. Cài bằng: pip install imbalanced-learn")
    # Fit model
    if sw_train is not None:
        model.fit(X_filled, y, sample_weight=sw_train, verbose=True)
    else:
        model.fit(X_filled, y, verbose=True)
    print('Train full data done!')
    # Lưu model
    import os
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model_full_xgb.pkl')
    print('Đã lưu model vào models/model_full_xgb.pkl')
    return model

if __name__ == '__main__':
    train_full_data(features_path, oversample=True)
