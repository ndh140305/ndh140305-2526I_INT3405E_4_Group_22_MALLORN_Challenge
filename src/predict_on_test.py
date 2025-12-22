import pandas as pd
import joblib
from data_processing import handle_missing_and_weights
import os

def predict_on_test(model_path, test_features_path, output_path):
    # Nạp model
    model = joblib.load(model_path)
    # Nạp feature test
    df = pd.read_csv(test_features_path)
    object_ids = df['object_id']
    X = df.drop(['object_id'], axis=1)
    # Xử lý missing value 
    X_filled, _ = handle_missing_and_weights(X)
    if hasattr(model, 'feature_names_in_'):
        X_filled = X_filled[model.feature_names_in_]
    else:
        # fallback: lấy từ booster
        booster = getattr(model, 'get_booster', lambda: None)()
        if booster is not None:
            feat_names = booster.feature_names
            X_filled = X_filled[feat_names]
    
    THRESHOLD = 0.45

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_filled)[:, 1]
        y_pred = (probs > THRESHOLD).astype(int)
    else:
        y_pred = model.predict(X_filled)
    # Lưu kết quả
    submission = pd.DataFrame({'object_id': object_ids, 'target': y_pred})
    submission.to_csv(output_path, index=False)
    print(f"Đã lưu file dự đoán: {output_path}")

if __name__ == '__main__':
    model_path = 'models/model_xgb_tuned.pkl'
    
    test_features_path = 'features/full_test_features.csv'

    output_path = 'data/submission.csv'
    predict_on_test(model_path, test_features_path, output_path)
