
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def extract_features(df):
    # Chuẩn hóa tên cột: chữ thường, thay khoảng trắng bằng gạch dưới
    df = df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))


    #  Hiệu chỉnh bụi (de-extinction) cho flux
    #    Sử dụng EBV và hệ số hiệu chỉnh phổ (R_V = 3.1, A_lambda/A_V ~ 3.1 cho optical)
    if 'ebv' in df.columns and 'flux' in df.columns:
        R_V = 3.1
        # Hệ số hiệu chỉnh cho từng band (giá trị mẫu, có thể thay đổi tuỳ filter)
        band_ext = {
            'u': 5.155,
            'g': 3.793,
            'r': 2.751,
            'i': 2.086,
            'z': 1.479,
            'y': 1.263
        }

        def correct_extinction(row):
            band = str(row['filter']).lower()
            a_lambda = band_ext.get(band, 3.1) * row['ebv']
            return row['flux'] * 10 ** (0.4 * a_lambda)

        df['flux'] = df.apply(correct_extinction, axis=1)


    #  Hiệu chỉnh thời gian (time dilation): t_rest = t_obs / (1+Z)
    if 'z' in df.columns and 'time_(mjd)' in df.columns:
        df['time_rest'] = df['time_(mjd)'] / (1.0 + df['z'].fillna(0))
    else:
        df['time_rest'] = df['time_(mjd)'] if 'time_(mjd)' in df.columns else 0

    #  Độ sáng tuyệt đối (absolute magnitude) nếu có mag và Z
    #    M = m - 5*log10(D_L/10pc), D_L: khoảng cách chuẩn dựa trên Z (đơn giản hoá)
    if 'mag' in df.columns and 'z' in df.columns:
        # Hằng số Hubble và tốc độ ánh sáng để tính D_L (đơn giản hoá)
        c = 3e5  # km/s
        h0 = 70  # km/s/Mpc

        def lum_dist(z):
            return (c / h0) * z  # Mpc, gần đúng cho z nhỏ

        df['abs_mag'] = df.apply(
            lambda row: row['mag'] - 5 * np.log10(max(lum_dist(row['z']) * 1e6, 1) / 10)
            if pd.notnull(row['mag']) and pd.notnull(row['z']) else np.nan,
            axis=1
        )

    features = []
    object_ids = df['object_id'].unique()
    for object_id in object_ids:
        obj = df[df['object_id'] == object_id]
        feature = {'object_id': object_id}
        # Nếu có cột vị trí, tính distance to galaxy center
        if (
            'ra' in obj.columns and 'dec' in obj.columns and
            'galaxy_ra' in obj.columns and 'galaxy_dec' in obj.columns
        ):
            # Sử dụng vị trí đầu tiên (giả sử không đổi)
            ra, dec = obj.iloc[0]['ra'], obj.iloc[0]['dec']
            gra, gdec = obj.iloc[0]['galaxy_ra'], obj.iloc[0]['galaxy_dec']
            # Khoảng cách góc đơn giản (không hiệu chỉnh cầu)
            feature['dist_to_galaxy_center'] = np.sqrt((ra - gra) ** 2 + (dec - gdec) ** 2)
        else:
            feature['dist_to_galaxy_center'] = 0.0

        # Per band features
        band_peak_times = {}
        for band in obj['filter'].unique():
            band_df = obj[obj['filter'] == band]
            prefix = f'band_{band}_'
            vals = band_df['flux'].dropna().reset_index(drop=True)
            times = band_df['time_(mjd)'].values if 'time_(mjd)' in band_df else None

            # Helper to safely get stat, fallback 0 if nan/inf
            def safe_stat(func, arr, default=0.0):
                try:
                    val = func(arr)
                    if np.isnan(val) or np.isinf(val):
                        return default
                    return val
                except Exception:
                    return default

            feature[prefix + 'flux_mean'] = safe_stat(np.mean, vals) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_std'] = safe_stat(np.std, vals) if len(vals) > 1 else 0.0
            feature[prefix + 'flux_max'] = safe_stat(np.max, vals) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_min'] = safe_stat(np.min, vals) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_median'] = safe_stat(np.median, vals) if len(vals) > 0 else 0.0
            vals_err = band_df['flux_err'].dropna()
            feature[prefix + 'flux_err_mean'] = safe_stat(np.mean, vals_err) if len(vals_err) > 0 else 0.0
            feature[prefix + 'flux_err_std'] = safe_stat(np.std, vals_err) if len(vals_err) > 1 else 0.0
            feature[prefix + 'count'] = len(vals)

            # Thêm shape features cho từng band
            feature[prefix + 'flux_skew'] = safe_stat(pd.Series(vals).skew, vals) if len(vals) > 2 else 0.0
            feature[prefix + 'flux_kurtosis'] = safe_stat(pd.Series(vals).kurtosis, vals) if len(vals) > 3 else 0.0
            feature[prefix + 'flux_peak2peak'] = (
                safe_stat(np.max, vals) - safe_stat(np.min, vals)
            ) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_autocorr'] = safe_stat(pd.Series(vals).autocorr, vals) if len(vals) > 1 else 0.0

            # Rolling features (window=3)
            if len(vals) >= 3:
                roll_mean = pd.Series(vals).rolling(3).mean().mean()
                roll_std = pd.Series(vals).rolling(3).std().mean()
                feature[prefix + 'flux_roll_mean'] = 0.0 if np.isnan(roll_mean) or np.isinf(roll_mean) else roll_mean
                feature[prefix + 'flux_roll_std'] = 0.0 if np.isnan(roll_std) or np.isinf(roll_std) else roll_std
            else:
                feature[prefix + 'flux_roll_mean'] = 0.0
                feature[prefix + 'flux_roll_std'] = 0.0

            # Zero-crossings (số lần đổi dấu)
            if len(vals) > 1:
                zero_crossings = np.where(np.diff(np.sign(vals)))[0]
                feature[prefix + 'zero_crossings'] = len(zero_crossings)
            else:
                feature[prefix + 'zero_crossings'] = 0

            # Số peak (local maxima) và trough (local minima)
            if len(vals) > 2:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(vals)
                troughs, _ = find_peaks(-vals)
                feature[prefix + 'n_peaks'] = len(peaks)
                feature[prefix + 'n_troughs'] = len(troughs)
                # Peak time (thời gian đến peak lớn nhất)
                if times is not None and len(peaks) > 0:
                    peak_fluxes = vals[peaks]
                    max_peak_idx = peaks[np.argmax(peak_fluxes)]
                    feature[prefix + 'peak_time'] = times[max_peak_idx] if max_peak_idx < len(times) else 0.0
                    band_peak_times[band] = times[max_peak_idx] if max_peak_idx < len(times) else None
                else:
                    feature[prefix + 'peak_time'] = 0.0
                    band_peak_times[band] = None

                # --- Đặc trưng vật lý TDE ---
                # Rise time: thời gian từ điểm đầu đến peak đầu tiên
                feature[prefix + 'rise_time'] = (
                    times[peaks[0]] - times[0]
                ) if times is not None and len(peaks) > 0 else 0.0

                # Decay rate: độ dốc giảm sau peak đầu tiên (fit tuyến tính đoạn sau peak)
                if times is not None and len(peaks) > 0 and peaks[0] < len(times) - 2:
                    decay_times = times[peaks[0]:]
                    decay_flux = vals[peaks[0]:]
                    if len(decay_times) > 1:
                        try:
                            # Linear decay rate
                            lr_decay = LinearRegression()
                            lr_decay.fit(decay_times.reshape(-1, 1), decay_flux)
                            feature[prefix + 'decay_rate'] = lr_decay.coef_[0]
                            # Power-law fit: fit decay_flux = A*(t-t0)^alpha
                            t0 = decay_times[0]
                            t_rel = decay_times - t0 + 1e-6
                            # Lọc giá trị dương
                            mask = (decay_flux > 0) & (t_rel > 0)
                            # Chỉ fit nếu đủ điểm và giá trị hợp lệ
                            if (
                                np.sum(mask) > 4 and
                                np.all(np.isfinite(decay_flux[mask])) and
                                np.all(np.isfinite(t_rel[mask]))
                            ):
                                from scipy.optimize import curve_fit

                                def powerlaw(t, a, alpha):
                                    return a * (t) ** alpha

                                try:
                                    popt, _ = curve_fit(
                                        powerlaw, t_rel[mask], decay_flux[mask], maxfev=5000
                                    )
                                    feature[prefix + 'decay_powerlaw_alpha'] = popt[1]
                                except Exception:
                                    feature[prefix + 'decay_powerlaw_alpha'] = 0.0
                            else:
                                feature[prefix + 'decay_powerlaw_alpha'] = 0.0
                        except Exception:
                            feature[prefix + 'decay_rate'] = 0.0
                            feature[prefix + 'decay_powerlaw_alpha'] = 0.0
                    else:
                        feature[prefix + 'decay_rate'] = 0.0
                        feature[prefix + 'decay_powerlaw_alpha'] = 0.0
                else:
                    feature[prefix + 'decay_rate'] = 0.0
                    feature[prefix + 'decay_powerlaw_alpha'] = 0.0

                # Double peak: có 2 peak lớn rõ rệt không
                feature[prefix + 'double_peak'] = (
                    1 if len(peaks) >= 2 and (vals[peaks[1]] > 0.7 * vals[peaks[0]]) else 0
                )

                # Peak separation: khoảng cách thời gian giữa 2 peak lớn nhất
                if len(peaks) >= 2:
                    feature[prefix + 'peak_separation'] = times[peaks[1]] - times[peaks[0]]
                else:
                    feature[prefix + 'peak_separation'] = 0.0

                # Asymmetry: so sánh rise time và decay time quanh peak đầu
                if times is not None and len(peaks) > 0:
                    rise = times[peaks[0]] - times[0]
                    decay = times[-1] - times[peaks[0]]
                    feature[prefix + 'asymmetry'] = (
                        (decay - rise) / (decay + rise + 1e-8)
                    )
                else:
                    feature[prefix + 'asymmetry'] = 0.0

                # FWHM: độ rộng tại nửa chiều cao peak đầu
                if len(peaks) > 0:
                    half_max = vals[peaks[0]] / 2.0
                    left = np.where(vals[:peaks[0]] <= half_max)[0]
                    right = np.where(vals[peaks[0]:] <= half_max)[0]
                    if len(left) > 0 and len(right) > 0:
                        fwhm = times[peaks[0] + right[0]] - times[left[-1]]
                        feature[prefix + 'fwhm'] = fwhm
                    else:
                        feature[prefix + 'fwhm'] = 0.0
                else:
                    feature[prefix + 'fwhm'] = 0.0
            else:
                feature[prefix + 'n_peaks'] = 0
                feature[prefix + 'n_troughs'] = 0
                feature[prefix + 'peak_time'] = 0.0
                band_peak_times[band] = None
                feature[prefix + 'rise_time'] = 0.0
                feature[prefix + 'decay_rate'] = 0.0
                feature[prefix + 'double_peak'] = 0
                feature[prefix + 'peak_separation'] = 0.0
                feature[prefix + 'asymmetry'] = 0.0
                feature[prefix + 'fwhm'] = 0.0

            # Color at peak (giữa các band phổ biến)
            # Sẽ tính sau khi đã duyệt hết các band (ở ngoài vòng for band)

            # Outlier count (số điểm vượt 3σ)
            if len(vals) > 1:
                mean = np.mean(vals)
                std = np.std(vals)
                feature[prefix + 'outlier_count'] = np.sum(np.abs(vals - mean) > 3 * std)
            else:
                feature[prefix + 'outlier_count'] = 0

            # Percentile features
            for q in [10, 25, 75, 90]:
                feature[f'{prefix}flux_p{q}'] = np.percentile(vals, q) if len(vals) > 0 else 0.0

            # Color at peak và color evolution (g-r, r-i, g-i)
            # Chỉ tính nếu có đủ các band phổ biến
            bands = ['g', 'r', 'i']
            band_peaks = {}
            band_peak_times = {}
            band_colors = {'g-r': [], 'r-i': [], 'g-i': []}
            band_times = []
            for band in bands:
                band_df = obj[obj['filter'] == band]
                vals = band_df['flux'].dropna().reset_index(drop=True)
                times = band_df['time_(mjd)'].values if 'time_(mjd)' in band_df else None
                if len(vals) > 2:
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(vals)
                    if len(peaks) > 0:
                        band_peaks[band] = vals[peaks[0]]
                        band_peak_times[band] = times[peaks[0]] if times is not None else None
                    else:
                        band_peaks[band] = 0.0
                        band_peak_times[band] = None
                    # Lưu color theo thời gian (nếu có đủ bands tại cùng thời điểm)
                    if times is not None:
                        for t, v in zip(times, vals):
                            band_times.append((band, t, v))

            # Color at peak
            feature['color_g_r_at_peak'] = (
                band_peaks['g'] - band_peaks['r']
                if band_peaks.get('g') is not None and band_peaks.get('r') is not None else 0.0
            )
            feature['color_r_i_at_peak'] = (
                band_peaks['r'] - band_peaks['i']
                if band_peaks.get('r') is not None and band_peaks.get('i') is not None else 0.0
            )
            feature['color_g_i_at_peak'] = (
                band_peaks['g'] - band_peaks['i']
                if band_peaks.get('g') is not None and band_peaks.get('i') is not None else 0.0
            )

            # Color evolution (slope của g-r, r-i theo thời gian)
            if band_peak_times.get('g') is not None and band_peak_times.get('r') is not None:
                try:
                    lr_color = LinearRegression()
                    lr_color.fit(
                        np.array([band_peak_times['g'], band_peak_times['r']]).reshape(-1, 1),
                        np.array([band_peaks['g'], band_peaks['r']])
                    )
                    feature['color_gr_slope'] = lr_color.coef_[0]
                except Exception:
                    feature['color_gr_slope'] = 0.0
            else:
                feature['color_gr_slope'] = 0.0

            # Color stability: std của chỉ số màu theo thời gian (nếu có đủ bands tại cùng thời điểm)
            # Giả sử times của từng band gần khớp nhau, dùng mean gần nhất
            if len(band_times) > 0:
                import collections
                time_dict = collections.defaultdict(dict)
                for band, t, v in band_times:
                    time_dict[t][band] = v
                color_gr, color_ri, color_gi = [], [], []
                for t, band_vals in time_dict.items():
                    if all(b in band_vals for b in ['g', 'r', 'i']):
                        color_gr.append(band_vals['g'] - band_vals['r'])
                        color_ri.append(band_vals['r'] - band_vals['i'])
                        color_gi.append(band_vals['g'] - band_vals['i'])
                feature['color_gr_stability'] = np.std(color_gr) if len(color_gr) > 1 else 0.0
                feature['color_ri_stability'] = np.std(color_ri) if len(color_ri) > 1 else 0.0
                feature['color_gi_stability'] = np.std(color_gi) if len(color_gi) > 1 else 0.0
            else:
                feature['color_gr_stability'] = 0.0
                feature['color_ri_stability'] = 0.0
                feature['color_gi_stability'] = 0.0

        # Global features
        vals_all = obj['flux'].dropna()
        feature['amplitude'] = (
            safe_stat(np.max, vals_all) - safe_stat(np.min, vals_all)
        ) if len(vals_all) > 0 else 0.0
        feature['variability'] = (
            safe_stat(np.std, vals_all) / (safe_stat(np.mean, vals_all) + 1e-8)
        ) if len(vals_all) > 1 else 0.0
        feature['duration'] = (
            obj['time_(mjd)'].max() - obj['time_(mjd)'].min()
        ) if len(obj['time_(mjd)'].dropna()) > 0 else 0.0

        # Slope (linear fit flux ~ time) - loại bỏ NaN trước khi fit
        valid = obj[['time_(mjd)', 'flux']].dropna()
        if len(valid) > 1:
            try:
                lr = LinearRegression()
                lr.fit(valid[['time_(mjd)']], valid['flux'])
                slope = lr.coef_[0]
                feature['slope'] = 0.0 if np.isnan(slope) or np.isinf(slope) else slope
            except Exception:
                feature['slope'] = 0.0
        else:
            feature['slope'] = 0.0

        # Thêm global shape features
        feature['flux_skew'] = safe_stat(pd.Series(vals_all).skew, vals_all) if len(vals_all) > 2 else 0.0
        feature['flux_kurtosis'] = safe_stat(pd.Series(vals_all).kurtosis, vals_all) if len(vals_all) > 3 else 0.0
        feature['flux_peak2peak'] = (
            safe_stat(np.max, vals_all) - safe_stat(np.min, vals_all)
        ) if len(vals_all) > 0 else 0.0
        feature['flux_autocorr'] = safe_stat(pd.Series(vals_all).autocorr, vals_all) if len(vals_all) > 1 else 0.0
        if len(vals_all) >= 3:
            roll_mean = pd.Series(vals_all).rolling(3).mean().mean()
            roll_std = pd.Series(vals_all).rolling(3).std().mean()
            feature['flux_roll_mean'] = 0.0 if np.isnan(roll_mean) or np.isinf(roll_mean) else roll_mean
            feature['flux_roll_std'] = 0.0 if np.isnan(roll_std) or np.isinf(roll_std) else roll_std
        else:
            feature['flux_roll_mean'] = 0.0
            feature['flux_roll_std'] = 0.0

        # Đặc trưng tần suất: dominant freq (FFT)
        flux = vals_all.values
        if len(flux) > 5:
            try:
                fft = np.fft.fft(flux - np.mean(flux))
                freqs = np.fft.fftfreq(len(flux))
                fft_power = np.abs(fft)
                dom_freq = freqs[np.argmax(fft_power[1:]) + 1]
                dom_freq = 0.0 if np.isnan(dom_freq) or np.isinf(dom_freq) else dom_freq
                fft_power_max = np.max(fft_power)
                fft_power_max = 0.0 if np.isnan(fft_power_max) or np.isinf(fft_power_max) else fft_power_max
                feature['dominant_freq'] = dom_freq
                feature['fft_power_max'] = fft_power_max
            except Exception:
                feature['dominant_freq'] = 0.0
                feature['fft_power_max'] = 0.0
        else:
            feature['dominant_freq'] = 0.0
            feature['fft_power_max'] = 0.0

        # Feature interaction: log/ratio cho top feature
        # log_variability
        feature['log_variability'] = (
            np.log1p(feature['variability']) if feature['variability'] > 0 else 0.0
        )
        # ratio band_r_flux_max / band_g_flux_max
        if feature.get('band_g_flux_max', 0) != 0:
            feature['ratio_r_g_flux_max'] = feature.get('band_r_flux_max', 0) / (
                feature.get('band_g_flux_max', 1e-6)
            )
        else:
            feature['ratio_r_g_flux_max'] = 0.0
        # ratio band_i_flux_mean / band_r_flux_mean
        if feature.get('band_r_flux_mean', 0) != 0:
            feature['ratio_i_r_flux_mean'] = feature.get('band_i_flux_mean', 0) / (
                feature.get('band_r_flux_mean', 1e-6)
            )
        else:
            feature['ratio_i_r_flux_mean'] = 0.0
        # ratio band_r_flux_median / band_i_flux_median
        if feature.get('band_i_flux_median', 0) != 0:
            feature['ratio_r_i_flux_median'] = feature.get('band_r_flux_median', 0) / (
                feature.get('band_i_flux_median', 1e-6)
            )
        else:
            feature['ratio_r_i_flux_median'] = 0.0

        features.append(feature)

    return pd.DataFrame(features)
    features = []
    object_ids = df['object_id'].unique()
    for object_id in object_ids:
        obj = df[df['object_id'] == object_id]
        feature = {'object_id': object_id}
        # Nếu có cột vị trí, tính distance to galaxy center
        if 'ra' in obj.columns and 'dec' in obj.columns and 'galaxy_ra' in obj.columns and 'galaxy_dec' in obj.columns:
            # Sử dụng vị trí đầu tiên (giả sử không đổi)
            ra, dec = obj.iloc[0]['ra'], obj.iloc[0]['dec']
            gra, gdec = obj.iloc[0]['galaxy_ra'], obj.iloc[0]['galaxy_dec']
            # Khoảng cách góc đơn giản (không hiệu chỉnh cầu)
            feature['dist_to_galaxy_center'] = np.sqrt((ra-gra)**2 + (dec-gdec)**2)
        else:
            feature['dist_to_galaxy_center'] = 0.0
        # Per band features
        band_peak_times = {}
        for band in obj['filter'].unique():
            band_df = obj[obj['filter'] == band]
            prefix = f'band_{band}_'
            vals = band_df['flux'].dropna().reset_index(drop=True)
            times = band_df['time_(mjd)'].values if 'time_(mjd)' in band_df else None
            # Helper to safely get stat, fallback 0 if nan/inf
            def safe_stat(func, arr, default=0.0):
                try:
                    val = func(arr)
                    if np.isnan(val) or np.isinf(val):
                        return default
                    return val
                except Exception:
                    return default

            feature[prefix+'flux_mean'] = safe_stat(np.mean, vals) if len(vals) > 0 else 0.0
            feature[prefix+'flux_std'] = safe_stat(np.std, vals) if len(vals) > 1 else 0.0
            feature[prefix+'flux_max'] = safe_stat(np.max, vals) if len(vals) > 0 else 0.0
            feature[prefix+'flux_min'] = safe_stat(np.min, vals) if len(vals) > 0 else 0.0
            feature[prefix+'flux_median'] = safe_stat(np.median, vals) if len(vals) > 0 else 0.0
            vals_err = band_df['flux_err'].dropna()
            feature[prefix+'flux_err_mean'] = safe_stat(np.mean, vals_err) if len(vals_err) > 0 else 0.0
            feature[prefix+'flux_err_std'] = safe_stat(np.std, vals_err) if len(vals_err) > 1 else 0.0
            feature[prefix+'count'] = len(vals)
            # Thêm shape features cho từng band
            feature[prefix+'flux_skew'] = safe_stat(pd.Series(vals).skew, vals) if len(vals) > 2 else 0.0
            feature[prefix+'flux_kurtosis'] = safe_stat(pd.Series(vals).kurtosis, vals) if len(vals) > 3 else 0.0
            feature[prefix+'flux_peak2peak'] = (safe_stat(np.max, vals) - safe_stat(np.min, vals)) if len(vals) > 0 else 0.0
            feature[prefix+'flux_autocorr'] = safe_stat(pd.Series(vals).autocorr, vals) if len(vals) > 1 else 0.0
            # Rolling features (window=3)
            if len(vals) >= 3:
                roll_mean = pd.Series(vals).rolling(3).mean().mean()
                roll_std = pd.Series(vals).rolling(3).std().mean()
                feature[prefix+'flux_roll_mean'] = 0.0 if np.isnan(roll_mean) or np.isinf(roll_mean) else roll_mean
                feature[prefix+'flux_roll_std'] = 0.0 if np.isnan(roll_std) or np.isinf(roll_std) else roll_std
            else:
                feature[prefix+'flux_roll_mean'] = 0.0
                feature[prefix+'flux_roll_std'] = 0.0

            # Zero-crossings (số lần đổi dấu)
            if len(vals) > 1:
                zero_crossings = np.where(np.diff(np.sign(vals)))[0]
                feature[prefix+'zero_crossings'] = len(zero_crossings)
            else:
                feature[prefix+'zero_crossings'] = 0

            # Số peak (local maxima) và trough (local minima)
            if len(vals) > 2:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(vals)
                troughs, _ = find_peaks(-vals)
                feature[prefix+'n_peaks'] = len(peaks)
                feature[prefix+'n_troughs'] = len(troughs)
                # Peak time (thời gian đến peak lớn nhất)
                if times is not None and len(peaks) > 0:
                    peak_fluxes = vals[peaks]
                    max_peak_idx = peaks[np.argmax(peak_fluxes)]
                    feature[prefix+'peak_time'] = times[max_peak_idx] if max_peak_idx < len(times) else 0.0
                    band_peak_times[band] = times[max_peak_idx] if max_peak_idx < len(times) else None
                else:
                    feature[prefix+'peak_time'] = 0.0
                    band_peak_times[band] = None

                # --- Đặc trưng vật lý TDE ---
                # Rise time: thời gian từ điểm đầu đến peak đầu tiên
                feature[prefix+'rise_time'] = (times[peaks[0]] - times[0]) if times is not None and len(peaks) > 0 else 0.0
                # Decay rate: độ dốc giảm sau peak đầu tiên (fit tuyến tính đoạn sau peak)
                if times is not None and len(peaks) > 0 and peaks[0] < len(times)-2:
                    decay_times = times[peaks[0]:]
                    decay_flux = vals[peaks[0]:]
                    if len(decay_times) > 1:
                        try:
                            # Linear decay rate
                            lr_decay = LinearRegression()
                            lr_decay.fit(decay_times.reshape(-1,1), decay_flux)
                            feature[prefix+'decay_rate'] = lr_decay.coef_[0]
                            # Power-law fit: fit decay_flux = A*(t-t0)^alpha
                            t0 = decay_times[0]
                            t_rel = decay_times - t0 + 1e-6
                            # Lọc giá trị dương
                            mask = (decay_flux > 0) & (t_rel > 0)
                            # Chỉ fit nếu đủ điểm và giá trị hợp lệ
                            if np.sum(mask) > 4 and np.all(np.isfinite(decay_flux[mask])) and np.all(np.isfinite(t_rel[mask])):
                                from scipy.optimize import curve_fit
                                def powerlaw(t, A, alpha):
                                    return A * (t)**alpha
                                try:
                                    popt, _ = curve_fit(powerlaw, t_rel[mask], decay_flux[mask], maxfev=5000)
                                    feature[prefix+'decay_powerlaw_alpha'] = popt[1]
                                except Exception:
                                    feature[prefix+'decay_powerlaw_alpha'] = 0.0
                            else:
                                feature[prefix+'decay_powerlaw_alpha'] = 0.0
                        except Exception:
                            feature[prefix+'decay_rate'] = 0.0
                            feature[prefix+'decay_powerlaw_alpha'] = 0.0
                    else:
                        feature[prefix+'decay_rate'] = 0.0
                        feature[prefix+'decay_powerlaw_alpha'] = 0.0
                else:
                    feature[prefix+'decay_rate'] = 0.0
                    feature[prefix+'decay_powerlaw_alpha'] = 0.0
                # Double peak: có 2 peak lớn rõ rệt không
                feature[prefix+'double_peak'] = 1 if len(peaks) >= 2 and (vals[peaks[1]] > 0.7*vals[peaks[0]]) else 0
                # Peak separation: khoảng cách thời gian giữa 2 peak lớn nhất
                if len(peaks) >= 2:
                    feature[prefix+'peak_separation'] = times[peaks[1]] - times[peaks[0]]
                else:
                    feature[prefix+'peak_separation'] = 0.0
                # Asymmetry: so sánh rise time và decay time quanh peak đầu
                if times is not None and len(peaks) > 0:
                    rise = times[peaks[0]] - times[0]
                    decay = times[-1] - times[peaks[0]]
                    feature[prefix+'asymmetry'] = (decay - rise) / (decay + rise + 1e-8)
                else:
                    feature[prefix+'asymmetry'] = 0.0
                # FWHM: độ rộng tại nửa chiều cao peak đầu
                if len(peaks) > 0:
                    half_max = vals[peaks[0]] / 2.0
                    left = np.where(vals[:peaks[0]] <= half_max)[0]
                    right = np.where(vals[peaks[0]:] <= half_max)[0]
                    if len(left) > 0 and len(right) > 0:
                        fwhm = times[peaks[0]+right[0]] - times[left[-1]]
                        feature[prefix+'fwhm'] = fwhm
                    else:
                        feature[prefix+'fwhm'] = 0.0
                else:
                    feature[prefix+'fwhm'] = 0.0
            else:
                feature[prefix+'n_peaks'] = 0
                feature[prefix+'n_troughs'] = 0
                feature[prefix+'peak_time'] = 0.0
                band_peak_times[band] = None
                feature[prefix+'rise_time'] = 0.0
                feature[prefix+'decay_rate'] = 0.0
                feature[prefix+'double_peak'] = 0
                feature[prefix+'peak_separation'] = 0.0
                feature[prefix+'asymmetry'] = 0.0
                feature[prefix+'fwhm'] = 0.0

            # Color at peak (giữa các band phổ biến)
            # Sẽ tính sau khi đã duyệt hết các band (ở ngoài vòng for band)

            # Outlier count (số điểm vượt 3σ)
            if len(vals) > 1:
                mean = np.mean(vals)
                std = np.std(vals)
                feature[prefix+'outlier_count'] = np.sum(np.abs(vals - mean) > 3*std)
            else:
                feature[prefix+'outlier_count'] = 0

            # Percentile features
            for q in [10, 25, 75, 90]:
                feature[f'{prefix}flux_p{q}'] = np.percentile(vals, q) if len(vals) > 0 else 0.0
            # Color at peak và color evolution (g-r, r-i, g-i)
            # Chỉ tính nếu có đủ các band phổ biến
            bands = ['g', 'r', 'i']
            band_peaks = {}
            band_peak_times = {}
            band_colors = {'g-r': [], 'r-i': [], 'g-i': []}
            band_times = []
            for band in bands:
                band_df = obj[obj['filter'] == band]
                vals = band_df['flux'].dropna().reset_index(drop=True)
                times = band_df['time_(mjd)'].values if 'time_(mjd)' in band_df else None
                if len(vals) > 2:
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(vals)
                    if len(peaks) > 0:
                        band_peaks[band] = vals[peaks[0]]
                        band_peak_times[band] = times[peaks[0]] if times is not None else None
                    else:
                        band_peaks[band] = 0.0
                        band_peak_times[band] = None
                    # Lưu color theo thời gian (nếu có đủ bands tại cùng thời điểm)
                    if times is not None:
                        for t, v in zip(times, vals):
                            band_times.append((band, t, v))
            # Color at peak
            feature['color_g_r_at_peak'] = band_peaks['g'] - band_peaks['r'] if band_peaks.get('g') is not None and band_peaks.get('r') is not None else 0.0
            feature['color_r_i_at_peak'] = band_peaks['r'] - band_peaks['i'] if band_peaks.get('r') is not None and band_peaks.get('i') is not None else 0.0
            feature['color_g_i_at_peak'] = band_peaks['g'] - band_peaks['i'] if band_peaks.get('g') is not None and band_peaks.get('i') is not None else 0.0
            # Color evolution (slope của g-r, r-i theo thời gian)
            if band_peak_times.get('g') is not None and band_peak_times.get('r') is not None:
                try:
                    lr_color = LinearRegression()
                    lr_color.fit(np.array([band_peak_times['g'], band_peak_times['r']]).reshape(-1,1), np.array([band_peaks['g'], band_peaks['r']]))
                    feature['color_gr_slope'] = lr_color.coef_[0]
                except Exception:
                    feature['color_gr_slope'] = 0.0
            else:
                feature['color_gr_slope'] = 0.0
            # Color stability: std của chỉ số màu theo thời gian (nếu có đủ bands tại cùng thời điểm)
            # Giả sử times của từng band gần khớp nhau, dùng mean gần nhất
            if len(band_times) > 0:
                # Gom theo thời gian, lấy mean từng band tại mỗi thời điểm
                import collections
                time_dict = collections.defaultdict(dict)
                for band, t, v in band_times:
                    time_dict[t][band] = v
                color_gr, color_ri, color_gi = [], [], []
                for t, band_vals in time_dict.items():
                    if all(b in band_vals for b in ['g','r','i']):
                        color_gr.append(band_vals['g'] - band_vals['r'])
                        color_ri.append(band_vals['r'] - band_vals['i'])
                        color_gi.append(band_vals['g'] - band_vals['i'])
                feature['color_gr_stability'] = np.std(color_gr) if len(color_gr) > 1 else 0.0
                feature['color_ri_stability'] = np.std(color_ri) if len(color_ri) > 1 else 0.0
                feature['color_gi_stability'] = np.std(color_gi) if len(color_gi) > 1 else 0.0
            else:
                feature['color_gr_stability'] = 0.0
                feature['color_ri_stability'] = 0.0
                feature['color_gi_stability'] = 0.0
        # Global features
        vals_all = obj['flux'].dropna()
        feature['amplitude'] = (safe_stat(np.max, vals_all) - safe_stat(np.min, vals_all)) if len(vals_all) > 0 else 0.0
        feature['variability'] = (safe_stat(np.std, vals_all) / (safe_stat(np.mean, vals_all) + 1e-8)) if len(vals_all) > 1 else 0.0
        feature['duration'] = obj['time_(mjd)'].max() - obj['time_(mjd)'].min() if len(obj['time_(mjd)'].dropna()) > 0 else 0.0
        # Slope (linear fit flux ~ time) - loại bỏ NaN trước khi fit
        valid = obj[['time_(mjd)', 'flux']].dropna()
        if len(valid) > 1:
            try:
                lr = LinearRegression()
                lr.fit(valid[['time_(mjd)']], valid['flux'])
                slope = lr.coef_[0]
                feature['slope'] = 0.0 if np.isnan(slope) or np.isinf(slope) else slope
            except Exception:
                feature['slope'] = 0.0
        else:
            feature['slope'] = 0.0
        # Thêm global shape features
        feature['flux_skew'] = safe_stat(pd.Series(vals_all).skew, vals_all) if len(vals_all) > 2 else 0.0
        feature['flux_kurtosis'] = safe_stat(pd.Series(vals_all).kurtosis, vals_all) if len(vals_all) > 3 else 0.0
        feature['flux_peak2peak'] = (safe_stat(np.max, vals_all) - safe_stat(np.min, vals_all)) if len(vals_all) > 0 else 0.0
        feature['flux_autocorr'] = safe_stat(pd.Series(vals_all).autocorr, vals_all) if len(vals_all) > 1 else 0.0
        if len(vals_all) >= 3:
            roll_mean = pd.Series(vals_all).rolling(3).mean().mean()
            roll_std = pd.Series(vals_all).rolling(3).std().mean()
            feature['flux_roll_mean'] = 0.0 if np.isnan(roll_mean) or np.isinf(roll_mean) else roll_mean
            feature['flux_roll_std'] = 0.0 if np.isnan(roll_std) or np.isinf(roll_std) else roll_std
        else:
            feature['flux_roll_mean'] = 0.0
            feature['flux_roll_std'] = 0.0
        # Đặc trưng tần suất: dominant freq (FFT)
    
        flux = vals_all.values
        if len(flux) > 5:
            try:
                fft = np.fft.fft(flux - np.mean(flux))
                freqs = np.fft.fftfreq(len(flux))
                fft_power = np.abs(fft)
                dom_freq = freqs[np.argmax(fft_power[1:])+1]
                dom_freq = 0.0 if np.isnan(dom_freq) or np.isinf(dom_freq) else dom_freq
                fft_power_max = np.max(fft_power)
                fft_power_max = 0.0 if np.isnan(fft_power_max) or np.isinf(fft_power_max) else fft_power_max
                feature['dominant_freq'] = dom_freq
                feature['fft_power_max'] = fft_power_max
            except Exception:
                feature['dominant_freq'] = 0.0
                feature['fft_power_max'] = 0.0
        else:
            feature['dominant_freq'] = 0.0
            feature['fft_power_max'] = 0.0
        # Feature interaction: log/ratio cho top feature
        # log_variability
        feature['log_variability'] = np.log1p(feature['variability']) if feature['variability'] > 0 else 0.0
        # ratio band_r_flux_max / band_g_flux_max
        if feature.get('band_g_flux_max', 0) != 0:
            feature['ratio_r_g_flux_max'] = feature.get('band_r_flux_max', 0) / (feature.get('band_g_flux_max', 1e-6))
        else:
            feature['ratio_r_g_flux_max'] = 0.0
        # ratio band_i_flux_mean / band_r_flux_mean
        if feature.get('band_r_flux_mean', 0) != 0:
            feature['ratio_i_r_flux_mean'] = feature.get('band_i_flux_mean', 0) / (feature.get('band_r_flux_mean', 1e-6))
        else:
            feature['ratio_i_r_flux_mean'] = 0.0
        # ratio band_r_flux_median / band_i_flux_median
        if feature.get('band_i_flux_median', 0) != 0:
            feature['ratio_r_i_flux_median'] = feature.get('band_r_flux_median', 0) / (feature.get('band_i_flux_median', 1e-6))
        else:
            feature['ratio_r_i_flux_median'] = 0.0
        features.append(feature)
    return pd.DataFrame(features)

