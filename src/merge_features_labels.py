import pandas as pd

def merge_features_with_labels(features_path, labels_path, output_path):
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    # Chỉ giữ lại object_id có nhãn (inner join)
    merged = features.merge(labels[['object_id', 'target']], on='object_id', how='inner')
    # Loại bỏ các dòng có target là NaN (nếu còn)
    merged = merged.dropna(subset=['target'])
    merged.to_csv(output_path, index=False)
    print(f'Saved merged features with labels to {output_path}')
    print(f'Tổng số dòng: {len(merged)}, số object_id: {merged["object_id"].nunique()}')
    print('Phân bố nhãn:')
    print(merged['target'].value_counts())
