import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def extract_features(df):
    # Hàm chuyển đổi flux sang apparent magnitude
    def flux_to_mag(flux):
        return -2.5 * np.log10(np.maximum(flux, 1e-8))

    def lum_dist(z):
        c = 3e5  # km/s
        h0 = 70  # km/s/Mpc
        return (c / h0) * z

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

    def safe(func, arr, default=0.0, *args, **kwargs):
        try:
            if arr is None or len(arr) == 0 or np.all(pd.isnull(arr)):
                return default
            val = func(arr, *args, **kwargs)
            if isinstance(val, (float, int, np.floating, np.integer)):
                if np.isnan(val) or np.isinf(val):
                    return default
            return val
        except Exception:
            return default

    def safe_log(arr, default=0.0):
        try:
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            arr = arr[arr > 0]
            if len(arr) == 0:
                return default
            val = np.log(arr)
            val = val[np.isfinite(val)]
            if len(val) == 0:
                return default
            return np.mean(val)
        except Exception:
            return default

    def safe_percentile(arr, q, default=0.0):
        try:
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                return default
            return np.percentile(arr, q)
        except Exception:
            return default

    def safe_rolling(series, window, func, default=0.0):
        try:
            if len(series) < window:
                return default
            roll = pd.Series(series).rolling(window)
            vals = func(roll)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                return default
            return np.mean(vals)
        except Exception:
            return default

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

        # --- PHYSICS FEATURES ---
        luminosity_stats = {}
        fit_error_powerlaw = {}
        for band in obj['filter'].unique():
            band_df = obj[obj['filter'] == band]
            # Đảm bảo times và vals cùng chiều dài, loại bỏ NaN đồng thời
            band_df = band_df[['time_rest', 'flux']].dropna()
            vals = band_df['flux'].values
            times = band_df['time_rest'].values if 'time_rest' in band_df else None
            z_val = obj['z'].iloc[0] if 'z' in obj.columns else 0.0
            d_l = lum_dist(z_val) * 1e6 if z_val > 0 else 1.0  # pc
            mags = flux_to_mag(vals)
            abs_mags = mags - 5 * np.log10(np.maximum(d_l, 1)) + 5
            if len(vals) > 0:
                luminosity_stats[band] = {
                    'max_lum': np.max(vals) * d_l**2,
                    'mean_lum': np.mean(vals) * d_l**2,
                    'min_abs_mag': np.min(abs_mags),
                    'mean_abs_mag': np.mean(abs_mags)
                }
            # Fit powerlaw t^-5/3 cho từng band
            if len(vals) > 4 and times is not None:
                from scipy.signal import find_peaks
                from scipy.optimize import curve_fit
                peaks, _ = find_peaks(vals)
                if len(peaks) > 0:
                    t0 = times[peaks[0]]
                    t_rel = times - t0 + 1e-6
                    # mask sẽ luôn cùng chiều dài với vals và t_rel
                    mask = (vals > 0) & (t_rel > 0)
                    if np.sum(mask) > 4:
                        def powerlaw(t, a):
                            return a * (t) ** (-5/3)
                        try:
                            popt, _ = curve_fit(powerlaw, t_rel[mask], vals[mask], maxfev=5000)
                            pred = powerlaw(t_rel[mask], popt[0])
                            mse = np.mean((vals[mask] - pred) ** 2)
                            fit_error_powerlaw[band] = mse
                        except Exception:
                            fit_error_powerlaw[band] = np.nan
                    else:
                        fit_error_powerlaw[band] = np.nan
                else:
                    fit_error_powerlaw[band] = np.nan
            else:
                fit_error_powerlaw[band] = np.nan

        # Gán các feature absolute magnitude/luminosity cho từng band phổ biến
        for band in ['u', 'g', 'r', 'i', 'z']:
            stats = luminosity_stats.get(band, None)
            if stats:
                feature[f'max_luminosity_band_{band}'] = stats['max_lum']
                feature[f'mean_luminosity_band_{band}'] = stats['mean_lum']
                feature[f'min_abs_mag_band_{band}'] = stats['min_abs_mag']
                feature[f'mean_abs_mag_band_{band}'] = stats['mean_abs_mag']
            else:
                feature[f'max_luminosity_band_{band}'] = 0.0
                feature[f'mean_luminosity_band_{band}'] = 0.0
                feature[f'min_abs_mag_band_{band}'] = 0.0
                feature[f'mean_abs_mag_band_{band}'] = 0.0
            # Fit error powerlaw
            feature[f'fit_error_powerlaw_band_{band}'] = fit_error_powerlaw.get(band, np.nan)

        # Color evolution: color_u_minus_g_mean, color_change_rate_u_g, blue_fraction
        u_df = obj[obj['filter'] == 'u']
        g_df = obj[obj['filter'] == 'g']
        # Tính color_u_minus_g_mean
        if len(u_df) > 0 and len(g_df) > 0:
            merged = pd.merge(u_df[['time_rest', 'flux']], g_df[['time_rest', 'flux']], on='time_rest', suffixes=('_u', '_g'))
            if len(merged) > 0:
                feature['color_u_minus_g_mean'] = np.mean(merged['flux_u'] - merged['flux_g'])
                # Fit độ dốc thay đổi màu sắc theo thời gian
                try:
                    lr = LinearRegression()
                    lr.fit(merged[['time_rest']], merged['flux_u'] - merged['flux_g'])
                    feature['color_change_rate_u_g'] = lr.coef_[0]
                except Exception:
                    feature['color_change_rate_u_g'] = 0.0
            else:
                feature['color_u_minus_g_mean'] = 0.0
                feature['color_change_rate_u_g'] = 0.0
        else:
            feature['color_u_minus_g_mean'] = 0.0
            feature['color_change_rate_u_g'] = 0.0
        # Blue fraction: tổng năng lượng band u+g chia tổng năng lượng các band
        total_flux = 0.0
        blue_flux = 0.0
        for band in obj['filter'].unique():
            vals = obj[obj['filter'] == band]['flux'].dropna().values
            total_flux += np.sum(vals)
            if band in ['u', 'g']:
                blue_flux += np.sum(vals)
        feature['blue_fraction'] = blue_flux / (total_flux + 1e-8)

        # Per band features
        band_peak_times = {}
        for band in obj['filter'].unique():
            band_df = obj[obj['filter'] == band]
            prefix = f'band_{band}_'
            vals = band_df['flux'].dropna().reset_index(drop=True)
            times = band_df['time_(mjd)'].values if 'time_(mjd)' in band_df else None
            # --- SPAM FEATURE BLOCK 2 ---
            # Thống kê cao hơn
            feature[prefix + 'flux_weighted_mean'] = np.average(vals, weights=np.arange(1, len(vals)+1)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_harmonic_mean'] = len(vals)/(np.sum(1.0/(vals+1e-8))) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_geometric_mean'] = np.exp(np.mean(np.log(np.abs(vals)+1e-8))) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_trimmed_std'] = np.std(np.sort(vals)[int(0.1*len(vals)):int(0.9*len(vals))]) if len(vals) > 10 else 0.0
            feature[prefix + 'flux_q5'] = np.percentile(vals, 5) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_q95'] = np.percentile(vals, 95) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_range'] = np.ptp(vals) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_peak_to_median'] = (np.max(vals)-np.median(vals)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_median_to_min'] = (np.median(vals)-np.min(vals)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_median_to_max'] = (np.max(vals)-np.median(vals)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_p1'] = np.percentile(vals, 1) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_p99'] = np.percentile(vals, 99) if len(vals) > 0 else 0.0
            # Clipped statistics
            if len(vals) > 10:
                clipped = vals[(vals > np.percentile(vals, 5)) & (vals < np.percentile(vals, 95))]
                feature[prefix + 'flux_clipped_mean'] = np.mean(clipped) if len(clipped) > 0 else 0.0
                feature[prefix + 'flux_clipped_std'] = np.std(clipped) if len(clipped) > 0 else 0.0
                feature[prefix + 'flux_clipped_min'] = np.min(clipped) if len(clipped) > 0 else 0.0
                feature[prefix + 'flux_clipped_max'] = np.max(clipped) if len(clipped) > 0 else 0.0
                feature[prefix + 'flux_clipped_skew'] = pd.Series(clipped).skew() if len(clipped) > 2 else 0.0
                feature[prefix + 'flux_clipped_kurtosis'] = pd.Series(clipped).kurtosis() if len(clipped) > 3 else 0.0
            else:
                feature[prefix + 'flux_clipped_mean'] = 0.0
                feature[prefix + 'flux_clipped_std'] = 0.0
                feature[prefix + 'flux_clipped_min'] = 0.0
                feature[prefix + 'flux_clipped_max'] = 0.0
                feature[prefix + 'flux_clipped_skew'] = 0.0
                feature[prefix + 'flux_clipped_kurtosis'] = 0.0

            # Rolling/windowed statistics (window=7,9,11)
            for w in [7,9,11]:
                if len(vals) >= w:
                    roll = pd.Series(vals).rolling(w)
                    feature[prefix + f'roll{w}_min'] = roll.min().mean()
                    feature[prefix + f'roll{w}_max'] = roll.max().mean()
                    feature[prefix + f'roll{w}_median'] = roll.median().mean()
                    feature[prefix + f'roll{w}_std'] = roll.std().mean()
                    # Robust lambdas for rolling.apply
                    feature[prefix + f'roll{w}_mad'] = roll.apply(lambda x: np.median(np.abs(x-np.median(x))) if len(x) > 0 and np.all(np.isfinite(x)) else 0.0).mean()
                    feature[prefix + f'roll{w}_iqr'] = roll.apply(lambda x: np.percentile(x,75)-np.percentile(x,25) if len(x) > 0 and np.all(np.isfinite(x)) else 0.0).mean()
                    feature[prefix + f'roll{w}_range'] = roll.apply(lambda x: np.ptp(x) if len(x) > 0 and np.all(np.isfinite(x)) else 0.0).mean()
                    try:
                        zscore = (pd.Series(vals)-pd.Series(vals).mean())/(pd.Series(vals).std()+1e-8)
                        feature[prefix + f'roll{w}_zscore_max'] = np.max(zscore) if np.all(np.isfinite(zscore)) else 0.0
                    except Exception:
                        feature[prefix + f'roll{w}_zscore_max'] = 0.0
                else:
                    feature[prefix + f'roll{w}_min'] = 0.0
                    feature[prefix + f'roll{w}_max'] = 0.0
                    feature[prefix + f'roll{w}_median'] = 0.0
                    feature[prefix + f'roll{w}_std'] = 0.0
                    feature[prefix + f'roll{w}_mad'] = 0.0
                    feature[prefix + f'roll{w}_iqr'] = 0.0
                    feature[prefix + f'roll{w}_range'] = 0.0
                    feature[prefix + f'roll{w}_zscore_max'] = 0.0

            # Biến đổi toán học mở rộng
            feature[prefix + 'flux_log1p_mean'] = np.mean(np.log1p(np.abs(vals))) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_expm1_mean'] = np.mean(np.expm1(vals)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_sinh_mean'] = np.mean(np.sinh(vals)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_cosh_mean'] = np.mean(np.cosh(vals)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_tanh_mean'] = np.mean(np.tanh(vals)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_arctan_mean'] = np.mean(np.arctan(vals)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_arcsin_mean'] = np.mean(np.arcsin(np.clip(vals/np.max(np.abs(vals)+1e-8),-1,1))) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_arccos_mean'] = np.mean(np.arccos(np.clip(vals/np.max(np.abs(vals)+1e-8),-1,1))) if len(vals) > 0 else 0.0
            # Sinh tổng, tích, max/min, abs, sign, reciprocal, square, cube
            feature[prefix + 'flux_abs_max'] = np.max(np.abs(vals)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_sign_mean'] = np.mean(np.sign(vals)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_reciprocal_max'] = np.max(1.0/(vals+1e-8)) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_square_max'] = np.max(vals**2) if len(vals) > 0 else 0.0
            feature[prefix + 'flux_cube_max'] = np.max(vals**3) if len(vals) > 0 else 0.0

            # Đặc trưng shape mở rộng
            for thresh, label in zip([np.mean(vals), np.median(vals)], ['mean', 'median']):
                if np.isfinite(thresh):
                    feature[prefix + f'flux_n_above_{label}'] = np.sum(vals > thresh) if len(vals) > 0 else 0.0
                    feature[prefix + f'flux_n_below_{label}'] = np.sum(vals < thresh) if len(vals) > 0 else 0.0
                else:
                    feature[prefix + f'flux_n_above_{label}'] = 0.0
                    feature[prefix + f'flux_n_below_{label}'] = 0.0
            if len(vals) > 1:
                feature[prefix + 'flux_n_increase'] = np.sum(np.diff(vals) > 0)
                feature[prefix + 'flux_n_decrease'] = np.sum(np.diff(vals) < 0)
                feature[prefix + 'flux_n_sign_change'] = np.sum(np.diff(np.sign(vals)) != 0)
            else:
                feature[prefix + 'flux_n_increase'] = 0.0
                feature[prefix + 'flux_n_decrease'] = 0.0
                feature[prefix + 'flux_n_sign_change'] = 0.0

            # Đặc trưng rolling tổ hợp giữa các band (window=5)
            for other_band in obj['filter'].unique():
                if other_band != band:
                    other_vals = obj[obj['filter'] == other_band]['flux'].dropna().values
                    oprefix = f'band_{band}_vs_{other_band}_'
                    if len(vals) >= 5 and len(other_vals) >= 5:
                        roll1 = pd.Series(vals).rolling(5).mean()
                        roll2 = pd.Series(other_vals).rolling(5).mean()
                        feature[oprefix + 'roll5_ratio_mean'] = np.nanmean(roll1/(roll2+1e-8))
                        feature[oprefix + 'roll5_diff_mean'] = np.nanmean(roll1-roll2)
                        feature[oprefix + 'roll5_sum_mean'] = np.nanmean(roll1+roll2)
                        feature[oprefix + 'roll5_prod_mean'] = np.nanmean(roll1*roll2)
                        feature[oprefix + 'roll5_norm_diff'] = np.nanmean((roll1-roll2)/(roll1+roll2+1e-8))
                    else:
                        feature[oprefix + 'roll5_ratio_mean'] = 0.0
                        feature[oprefix + 'roll5_diff_mean'] = 0.0
                        feature[oprefix + 'roll5_sum_mean'] = 0.0
                        feature[oprefix + 'roll5_prod_mean'] = 0.0
                        feature[oprefix + 'roll5_norm_diff'] = 0.0

            # --- SPAM FEATURE BLOCK ---
            # Thống kê cao hơn
            feature[prefix + 'flux_entropy'] = -np.sum((vals/np.sum(vals+1e-8)) * np.log(vals/np.sum(vals+1e-8)+1e-8)) if len(vals) > 1 else 0.0
            feature[prefix + 'flux_mad'] = np.median(np.abs(vals - np.median(vals))) if len(vals) > 1 else 0.0
            feature[prefix + 'flux_iqr'] = np.percentile(vals, 75) - np.percentile(vals, 25) if len(vals) > 1 else 0.0
            feature[prefix + 'flux_trimmed_mean'] = np.mean(np.sort(vals)[int(0.1*len(vals)):int(0.9*len(vals))]) if len(vals) > 10 else 0.0
            feature[prefix + 'flux_robust_std'] = (np.percentile(vals, 84) - np.percentile(vals, 16))/2 if len(vals) > 1 else 0.0

            # Rolling/windowed statistics (window=5)
            if len(vals) >= 5:
                roll = pd.Series(vals).rolling(5)
                feature[prefix + 'roll_min'] = roll.min().mean()
                feature[prefix + 'roll_max'] = roll.max().mean()
                feature[prefix + 'roll_median'] = roll.median().mean()
                feature[prefix + 'roll_std'] = roll.std().mean()
                feature[prefix + 'roll_skew'] = roll.apply(lambda x: pd.Series(x).skew()).mean()
                feature[prefix + 'roll_kurtosis'] = roll.apply(lambda x: pd.Series(x).kurtosis()).mean()
            else:
                feature[prefix + 'roll_min'] = 0.0
                feature[prefix + 'roll_max'] = 0.0
                feature[prefix + 'roll_median'] = 0.0
                feature[prefix + 'roll_std'] = 0.0
                feature[prefix + 'roll_skew'] = 0.0
                feature[prefix + 'roll_kurtosis'] = 0.0

            # Biến đổi toán học
            feature[prefix + 'flux_log_mean'] = safe_log(np.abs(vals)+1e-8)
            feature[prefix + 'flux_sqrt_mean'] = safe(lambda x: np.mean(np.sqrt(np.abs(x))), vals)
            feature[prefix + 'flux_abs_sum'] = safe(lambda x: np.sum(np.abs(x)), vals)
            feature[prefix + 'flux_sign_sum'] = safe(lambda x: np.sum(np.sign(x)), vals)
            feature[prefix + 'flux_square_sum'] = safe(lambda x: np.sum(np.square(x)), vals)
            feature[prefix + 'flux_cube_sum'] = safe(lambda x: np.sum(np.power(x,3)), vals)
            feature[prefix + 'flux_exp_sum'] = safe(lambda x: np.sum(np.exp(np.clip(x, None, 20))), vals)
            feature[prefix + 'flux_reciprocal_sum'] = safe(lambda x: np.sum(1.0/(x+1e-8)), vals)
            # Z-score, minmax, robust scaler
            if len(vals) > 1:
                try:
                    zscore = (vals-np.mean(vals))/(np.std(vals)+1e-8)
                    feature[prefix + 'flux_zscore_max'] = safe(np.max, zscore)
                except Exception:
                    feature[prefix + 'flux_zscore_max'] = 0.0
                try:
                    feature[prefix + 'flux_minmax'] = (safe(np.max, vals)-safe(np.min, vals))/(safe(np.max, vals)+1e-8)
                except Exception:
                    feature[prefix + 'flux_minmax'] = 0.0
                try:
                    robust_denom = safe_percentile(vals, 84) - safe_percentile(vals, 16) + 1e-8
                    feature[prefix + 'flux_robust_scaled_max'] = safe(np.max, (vals-np.median(vals))/robust_denom)
                except Exception:
                    feature[prefix + 'flux_robust_scaled_max'] = 0.0
            else:
                feature[prefix + 'flux_zscore_max'] = 0.0
                feature[prefix + 'flux_minmax'] = 0.0
                feature[prefix + 'flux_robust_scaled_max'] = 0.0

            # Đặc trưng shape/biến thiên
            feature[prefix + 'flux_plateau'] = safe(lambda x: np.mean(np.abs(np.diff(x, n=2))), vals)
            feature[prefix + 'flux_rise_decay_ratio'] = safe(lambda x: (np.max(x)-np.median(x))/(np.median(x)-np.min(x)+1e-8), vals)
            feature[prefix + 'flux_duration_above_halfmax'] = safe(lambda x: np.sum(x > (np.max(x)/2)), vals)
            feature[prefix + 'flux_time_to_halfmax'] = safe(lambda x: np.argmax(x > (np.max(x)/2)), vals)

            # Đặc trưng biến thiên
            feature[prefix + 'amplitude'] = safe(lambda x: np.max(x) - np.min(x), vals)
            feature[prefix + 'variability'] = safe(np.var, vals)
            feature[prefix + 'log_variability'] = safe_log([np.var(vals)+1e-8])

            # FFT features
            if len(vals) > 4:
                try:
                    fft = np.fft.fft(vals)
                    fft_power = np.abs(fft)**2
                    feature[prefix + 'fft_power_max'] = safe(np.max, fft_power)
                    feature[prefix + 'fft_power_sum'] = safe(np.sum, fft_power)
                    feature[prefix + 'fft_power_mean'] = safe(np.mean, fft_power)
                    feature[prefix + 'dominant_freq'] = safe(lambda x: np.argmax(x[1:]) + 1, fft_power)
                except Exception:
                    feature[prefix + 'fft_power_max'] = 0.0
                    feature[prefix + 'fft_power_sum'] = 0.0
                    feature[prefix + 'fft_power_mean'] = 0.0
                    feature[prefix + 'dominant_freq'] = 0.0
            else:
                feature[prefix + 'fft_power_max'] = 0.0
                feature[prefix + 'fft_power_sum'] = 0.0
                feature[prefix + 'fft_power_mean'] = 0.0
                feature[prefix + 'dominant_freq'] = 0.0

            # Tổ hợp giữa các band (ratio, diff, sum, product với band phổ biến)
            for other_band in obj['filter'].unique():
                if other_band != band:
                    other_vals = obj[obj['filter'] == other_band]['flux'].dropna().values
                    oprefix = f'band_{band}_vs_{other_band}_'
                    if len(vals) > 0 and len(other_vals) > 0:
                        feature[oprefix + 'flux_ratio_mean'] = np.mean(vals) / (np.mean(other_vals)+1e-8)
                        feature[oprefix + 'flux_diff_mean'] = np.mean(vals) - np.mean(other_vals)
                        feature[oprefix + 'flux_sum_mean'] = np.mean(vals) + np.mean(other_vals)
                        feature[oprefix + 'flux_prod_mean'] = np.mean(vals) * np.mean(other_vals)
                        feature[oprefix + 'flux_norm_diff'] = (np.mean(vals) - np.mean(other_vals)) / (np.mean(vals) + np.mean(other_vals) + 1e-8)
                    else:
                        feature[oprefix + 'flux_ratio_mean'] = 0.0
                        feature[oprefix + 'flux_diff_mean'] = 0.0
                        feature[oprefix + 'flux_sum_mean'] = 0.0
                        feature[oprefix + 'flux_prod_mean'] = 0.0
                        feature[oprefix + 'flux_norm_diff'] = 0.0

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
                feature[f'{prefix}flux_p{q}'] = safe_percentile(vals, q)

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

