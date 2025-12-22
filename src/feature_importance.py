import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(model_path, features_path, topk=100, output_path=None):
    model = joblib.load(model_path)
    df = pd.read_csv(features_path)
    X = df.drop(['object_id', 'target'], axis=1)
    # Lấy importance
    importances = model.feature_importances_
    feat_names = X.columns
    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False)
    print(imp_df.head(topk))
    if output_path:
        imp_df.to_csv(output_path, index=False)
    # Vẽ biểu đồ
    plt.figure(figsize=(10, min(topk, 100)))
    plt.barh(imp_df['feature'][:topk][::-1], imp_df['importance'][:topk][::-1])
    plt.xlabel('Importance')
    plt.title(f'Top {topk} Feature Importance')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model_path = 'models/model_full_xgb.pkl'
    features_path = 'features/full_train_features_labeled.csv'
    output_path = 'features/feature_importance.csv'
    plot_feature_importance(model_path, features_path, topk=30, output_path=output_path)
