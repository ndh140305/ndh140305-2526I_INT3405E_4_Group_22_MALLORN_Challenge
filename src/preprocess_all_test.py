import os
import pandas as pd
from feature_engineering import extract_features

def process_all_test_splits(data_dir, output_dir, splits=20):
    os.makedirs(output_dir, exist_ok=True)
    dfs = []
    for i in range(1, splits+1):
        split_name = f'split_{i:02d}'
        test_path = os.path.join(data_dir, split_name, 'test_full_lightcurves.csv')
        print(f'Processing {split_name}...')
        if not os.path.exists(test_path):
            print(f"Bỏ qua {test_path} (không tồn tại)")
            continue
        test_df = pd.read_csv(test_path)
        test_features = extract_features(test_df)
        test_features.to_csv(os.path.join(output_dir, f'{split_name}_test_features.csv'), index=False)
        dfs.append(test_features)
        print(f'Done {split_name}')
    if not dfs:
        print("Không có dữ liệu test nào để hợp nhất!")
        return
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Tổng số object_id test: {merged['object_id'].nunique()} | Tổng dòng: {len(merged)}")
    merged.to_csv(os.path.join(output_dir, 'full_test_features.csv'), index=False)
    print(f"Đã lưu file feature test hợp nhất: {os.path.join(output_dir, 'full_test_features.csv')}")

if __name__ == '__main__':
    process_all_test_splits(data_dir='data', output_dir='features', splits=20)
