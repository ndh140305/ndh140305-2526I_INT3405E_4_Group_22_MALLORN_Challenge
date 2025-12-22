import os
import pandas as pd

def merge_all_train_features(features_dir, output_path):
    files = [f for f in os.listdir(features_dir) if f.endswith('_train_features_labeled.csv')]
    files = sorted(files)  # đảm bảo đúng thứ tự split
    dfs = []
    for f in files:
        path = os.path.join(features_dir, f)
        print(f"Đang nạp {f} ...")
        df = pd.read_csv(path)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Tổng số object_id: {merged['object_id'].nunique()} | Tổng dòng: {len(merged)}")
    merged.to_csv(output_path, index=False)
    print(f"Đã lưu file hợp nhất: {output_path}")

if __name__ == '__main__':
    features_dir = 'features'
    output_path = os.path.join(features_dir, 'full_train_features_labeled.csv')
    merge_all_train_features(features_dir, output_path)
