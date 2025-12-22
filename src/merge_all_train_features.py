import os
import pandas as pd


def merge_all_features(features_dir, output_train_path, output_test_path):
    train_files = [f for f in os.listdir(features_dir) if f.endswith('_train_features_labeled.csv')]
    test_files = [f for f in os.listdir(features_dir) if f.endswith('_test_features.csv')]
    train_files = sorted(train_files)
    test_files = sorted(test_files)
    train_dfs = []
    test_dfs = []
    for f in train_files:
        path = os.path.join(features_dir, f)
        print(f"Đang nạp train: {f} ...")
        df = pd.read_csv(path)
        train_dfs.append(df)
    for f in test_files:
        path = os.path.join(features_dir, f)
        print(f"Đang nạp test: {f} ...")
        df = pd.read_csv(path)
        test_dfs.append(df)
    merged_train = pd.concat(train_dfs, ignore_index=True)
    merged_test = pd.concat(test_dfs, ignore_index=True)
    # Loại bỏ object_id trùng lặp ở train và test, chỉ giữ dòng đầu tiên cho mỗi object_id
    merged_train = merged_train.drop_duplicates(subset='object_id', keep='first').reset_index(drop=True)
    merged_test = merged_test.drop_duplicates(subset='object_id', keep='first').reset_index(drop=True)
    print(f"Tổng số object_id train: {merged_train['object_id'].nunique()} | Tổng dòng: {len(merged_train)}")
    print(f"Tổng số object_id test: {merged_test['object_id'].nunique()} | Tổng dòng: {len(merged_test)}")
    merged_train.to_csv(output_train_path, index=False)
    merged_test.to_csv(output_test_path, index=False)
    print(f"Đã lưu file hợp nhất train: {output_train_path}")
    print(f"Đã lưu file hợp nhất test: {output_test_path}")

if __name__ == '__main__':
    features_dir = 'features'
    output_train_path = os.path.join(features_dir, 'full_train_features_labeled.csv')
    output_test_path = os.path.join(features_dir, 'full_test_features.csv')
    merge_all_features(features_dir, output_train_path, output_test_path)
