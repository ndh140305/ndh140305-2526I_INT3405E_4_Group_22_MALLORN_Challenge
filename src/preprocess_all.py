import os
import pandas as pd
from feature_engineering import extract_features

def process_all_splits(data_dir, output_dir, splits=20):
    os.makedirs(output_dir, exist_ok=True)
    train_log_path = os.path.join(data_dir, 'train_log.csv')
    train_log = pd.read_csv(train_log_path)
    for i in range(1, splits+1):
        split_name = f'split_{i:02d}'
        train_path = os.path.join(data_dir, split_name, 'train_full_lightcurves.csv')
        test_path = os.path.join(data_dir, split_name, 'test_full_lightcurves.csv')
        print(f'Processing {split_name}...')
        # Train features
        train_df = pd.read_csv(train_path)
        train_features = extract_features(train_df)
        # Merge với nhãn, loại bỏ object_id không có nhãn
        merged = train_features.merge(train_log[['object_id', 'target']], on='object_id', how='inner')
       
        merged.to_csv(os.path.join(output_dir, f'{split_name}_train_features_labeled.csv'), index=False)
        # Test features
        test_df = pd.read_csv(test_path)
        test_features = extract_features(test_df)
        test_features.to_csv(os.path.join(output_dir, f'{split_name}_test_features.csv'), index=False)
        print(f'Done {split_name}')

if __name__ == '__main__':
    process_all_splits(data_dir='data', output_dir='features', splits=20)
