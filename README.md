## Tổng quan bài toán

- Bài toán: Phân loại sự kiện Tidal Disruption Events (TDE) từ dữ liệu lightcurve mô phỏng LSST
- Mục tiêu: Xác định chính xác các object là TDE (class hiếm) dựa trên các đặc trưng thống kê, vật lý và domain-specific được trích xuất từ chuỗi thời gian đa dải (u, g, r, i, z, y)
- Đánh giá:
  - Tập dữ liệu nhỏ: 3k cho tập train và 7k cho tập test
  - Dữ liệu mất cân bằng nặng: TDE rất hiếm, trong tập train số nhãn 0 gấp 19 lần nhãn 1
  - Dữ liệu nhiều nhiễu, thiếu, số lượng điểm quan sát không đồng đều
  - Số lượng feature lớn, nguy cơ overfitting cao

## Tiền xử lý

- Làm sạch dữ liệu:
  - Loại bỏ hoặc điền giá trị thiếu
  - Chuẩn hóa, kiểm tra outlier cho từng band
- Feature Engineering:
  - Sinh lượng lớn đặc trưng: thống kê, rolling, peak/trough, power spectrum, color, ratio,... cho từng band và toàn object
  - Thực tế huấn luyện cho thấy:
    - Hầu hết các đặc trưng đều không có đóng góp tốt
    - Các đặc trưng đứng top lại là các đặc trưng thống kê cơ bản
    - Các đặc trưng vật lý, thống kê phức tạp lại thể hiện không tốt
    - Nguyên nhân: XGBoost đánh giá importance dựa trên số lần đặc trưng được chọn để split tree, các đặc trưng phân tách mạnh sẽ được ưu tiên. Ngoài ra, có thể các đặc trưng được chọn chưa thực sự tốt
    - Kết quả: huấn luyện với tất cả đặc trưng và chỉ chọn đặc trưng xếp hạng cao có cải thiện nhưng rất ít
- Chia dữ liệu:
  - Sử dụng StratifiedKFold để chia train/validation đảm bảo phân phối nhãn

## Chi tiết Feature Engineering theo nhóm Feature

Các nhóm đặc trưng chính được xây dựng cho mỗi object (và cho từng band: u, g, r, i, z, y):

- Nhóm đặc trưng thống kê cơ bản: mean, std, max, min, median, count, outlier_count, n_peaks, n_troughs, fwhm, asymmetry,...

- Nhóm đặc trưng peak/trough: peak_time, peak_separation, double_peak, peak/trough count, peak/trough value, rise_time, decay_rate, decay_powerlaw_alpha,...

- Nhóm đặc trưng màu sắc và tỷ lệ: color_g_r_at_peak, color_r_i_at_peak, color_g_i_at_peak, color_gr_slope, color_gr_stability, color_ri_stability, color_gi_stability, ratio_r_g_flux_max, ratio_i_r_flux_mean, ratio_r_i_flux_median, các ratio/diff/sum/prod giữa các band (vd: band_r_vs_g_flux_sum_mean,...)

Nhóm các đặc trưng dưới sau khi thử nghiệm không có đóng góp nhiều cho mô hình

- Nhóm đặc trưng rolling/window: rolling mean, rolling std, rolling min/max, rolling median, rolling peak/trough, rolling zscore, rolling iqr, rolling mad, rolling range,... với nhiều window size khác nhau (5, 7, 9, 11)

- Nhóm đặc trưng phổ/tần số: fft_power_max, fft_power_sum, fft_power_mean, dominant_freq, log_variability, variability, amplitude, slope

- Nhóm đặc trưng vật lý/fit: max*luminosity_band*_, mean*luminosity_band*_, min*abs_mag_band*_, mean*abs_mag_band*_, fit*error_powerlaw_band*\*, dist_to_galaxy_center

- Nhóm đặc trưng entropy, robust, clipped: entropy, mad, iqr, trimmed_mean, robust_std, clipped_mean, clipped_std, clipped_min/max/skew/kurtosis,...

## Lựa chọn mô hình

- XGBoost (XGBClassifier):
  - Mạnh với dữ liệu tabular, tự động chọn feature, regularization tốt, hỗ trợ class imbalance
- Imbalanced-learn (RandomOverSampler, SMOTE):
  - Tăng số lượng TDE trong train, giúp mô hình học tốt hơn class hiếm
- Tiêu chí chọn:
  - F1-score trên validation/test, khả năng mở rộng, tốc độ huấn luyện

## Tối ưu mô hình

- Feature Selection:
  - Chọn top feature theo importance hoặc loại bỏ feature ít giá trị
- Hyperparameter Tuning:
  - Sử dụng GridSearchCV với F1-score để tìm tham số tối ưu cho mô hình (XGBoost/LightGBM)
  - Các tham số đã thử nghiệm:
    - XGBoost: n_estimators, learning_rate, max_depth, subsample, colsample_bytree, scale_pos_weight
  - Quy trình tuning:
    - Chia train/validation bằng StratifiedKFold để đảm bảo phân phối nhãn đồng đều
    - Đánh giá mô hình trên validation bằng F1-score, chọn tham số cho kết quả tốt nhất
    - Tuning threshold để tối ưu F1 do dữ liệu imbalance nặng
- Imbalance Handling:
  - Sử dụng scale_pos_weight của XGBoost hoặc oversampling

## Hướng dẫn chạy

- Tiền xử lý và sinh đặc trưng:
  ```bash
  python src/preprocess_all.py
  python src/feature_engineering.py
  ```

- Huấn luyện mô hình (chưa tune tham số, dùng để xếp hạng feature):
  ```bash
  python src/train_full.py
  ```

- Tối ưu tham số(huấn luyện trên top-k feature, tham số sau khi tối ưu đã được hard code):
  ```bash
  python src/tune_xgb.py
  ```

- Dự đoán trên tập test và tạo file submission:
  ```bash
  python src/predict_on_test.py
  ```


