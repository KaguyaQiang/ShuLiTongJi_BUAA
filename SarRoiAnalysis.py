import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import gamma, chi2, kstest, probplot
from matplotlib import pyplot as plt

# --- 1. 配置和路径设置 ---
IMAGE_DIR = './sar_images'
LABEL_DIR = './sar_labels'
OUTPUT_CSV = 'sar_mle_results.csv'
OUTPUT_PLOTS_DIR = './roi_plots'
OUTPUT_ANNOTATED_DIR = './sar_annotated'


def parse_yolo_annotation(label_path, img_width, img_height):
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            print(f"Warning: Label file {label_path} is empty.")
            return None


        parts = lines[0].strip().split()
        if len(parts) < 5:
            print(f"Warning: Annotation in {label_path} is incomplete.")
            return None

        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])

        w = int(width_norm * img_width)
        h = int(height_norm * img_height)
        x_c = int(x_center_norm * img_width)
        y_c = int(y_center_norm * img_height)

        x_min = max(0, x_c - w // 2)
        y_min = max(0, y_c - h // 2)
        x_max = min(img_width, x_c + (w + 1) // 2)
        y_max = min(img_height, y_c + (h + 1) // 2)

        return (x_min, y_min, x_max, y_max)

    except Exception as e:
        print(f"Error parsing {label_path}: {e}")
        return None


def visualize_roi_on_image(img_path, roi_coords, output_dir):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image for annotation: {img_path}")
            return

        x_min, y_min, x_max, y_max = roi_coords

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

        img_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f'annotated_{img_name}')
        cv2.imwrite(save_path, img)
        print(f"  Annotation saved to {save_path}")

    except Exception as e:
        print(f"Error during ROI visualization for {img_path}: {e}")


def perform_pearson_chi2_test(data, L_hat, scale_hat, alpha=0.05):
    n = data.size

    try:
        O_i, bin_edges = np.histogram(data, bins='auto')
    except Exception as e:
        print(f"Warning: Auto binning failed ({e}), using 30 bins.")
        O_i, bin_edges = np.histogram(data, bins=30)

    E_i = np.zeros_like(O_i, dtype=np.float64)
    for i in range(len(O_i)):
        # E_i = n * [CDF(bin_upper) - CDF(bin_lower)]
        cdf_upper = gamma.cdf(bin_edges[i + 1], a=L_hat, scale=scale_hat)
        cdf_lower = gamma.cdf(bin_edges[i], a=L_hat, scale=scale_hat)
        E_i[i] = n * (cdf_upper - cdf_lower)

    O_final = []
    E_final = []
    temp_O = 0
    temp_E = 0

    for i in range(len(O_i)):
        temp_O += O_i[i]
        temp_E += E_i[i]

        is_last = (i == len(O_i) - 1)

        if temp_E >= 5 or is_last:
            O_final.append(temp_O)
            E_final.append(temp_E)
            temp_O = 0
            temp_E = 0

    O_final = np.array(O_final)
    E_final = np.array(E_final)
    k = len(O_final)

    if k < 3:
        return {'Chi2_Stat': np.nan, 'P_Value': np.nan, 'DOF': np.nan, 'K_Bins': k,
                'Test_Status': 'Skipped (Too few bins)'}

    valid_indices = E_final > 1e-9
    chi2_stat = np.sum((O_final[valid_indices] - E_final[valid_indices]) ** 2 / E_final[valid_indices])

    m_estimated_params = 2
    dof = k - 1 - m_estimated_params

    if dof <= 0:
        return {'Chi2_Stat': chi2_stat, 'P_Value': np.nan, 'DOF': dof, 'K_Bins': k, 'Test_Status': 'Skipped (DOF <= 0)'}

    p_value = chi2.sf(chi2_stat, dof)

    is_fit_good = p_value >= alpha
    test_status = f"Fit Good (P > {alpha:.2f})" if is_fit_good else f"Fit Poor (P <= {alpha:.2f})"

    return {
        'Chi2_Stat': chi2_stat,
        'P_Value': p_value,
        'DOF': dof,
        'K_Bins': k,
        'Test_Status': test_status
    }


def perform_ks_test(data, L_hat, scale_hat, alpha=0.05):
    ks_stat, ks_p_value = kstest(data, 'gamma', args=(L_hat, 0, scale_hat))

    is_fit_good = ks_p_value >= alpha
    test_status = f"Fit Good (P > {alpha:.2f})" if is_fit_good else f"Fit Poor (P <= {alpha:.2f})"

    return {
        'KS_Stat': ks_stat,
        'KS_P_Value': ks_p_value,
        'KS_Status': test_status
    }


def main_analysis():
    if not os.path.exists(OUTPUT_PLOTS_DIR):
        os.makedirs(OUTPUT_PLOTS_DIR)
    if not os.path.exists(OUTPUT_ANNOTATED_DIR):
        os.makedirs(OUTPUT_ANNOTATED_DIR)

    results = []

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))]

    if not image_files:
        print(f"Error: No images found in {IMAGE_DIR}. Please check the path.")
        return

    print(f"Found {len(image_files)} image files to process.")

    for img_name in image_files:
        print(f"\n--- Processing {img_name} ---")

        img_path = os.path.join(IMAGE_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(LABEL_DIR, base_name + '.txt')

        if not os.path.exists(label_path):
            print(f"Skipping {img_name}: No corresponding label file {base_name}.txt found.")
            continue
        img_analysis = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_analysis is None:
            print(f"Error loading image {img_path}. Skipping.")
            continue

        img_h, img_w = img_analysis.shape
        roi_coords = parse_yolo_annotation(label_path, img_w, img_h)

        if not roi_coords:
            continue

        visualize_roi_on_image(img_path, roi_coords, OUTPUT_ANNOTATED_DIR)

        x_min, y_min, x_max, y_max = roi_coords

        roi_data = img_analysis[y_min:y_max, x_min:x_max].flatten()

        if roi_data.size < 100:
            print(f"Warning: ROI size is too small ({roi_data.size} pixels). Skipping analysis.")
            continue

        EPSILON = 1e-6
        roi_data = roi_data.astype(np.float64) + EPSILON

        try:
            a_hat, loc_hat, scale_hat = gamma.fit(roi_data, floc=0)

            L_hat = a_hat  # Shape Parameter (L)
            lambda_hat = 1.0 / scale_hat  # Rate Parameter (lambda)

            mean_obs = np.mean(roi_data)
            var_obs = np.var(roi_data)

            mean_theory = L_hat * scale_hat
            var_theory = L_hat * (scale_hat ** 2)

            mean_abs_error = abs(mean_theory - mean_obs)
            var_abs_error = abs(var_theory - var_obs)


            chi2_results = perform_pearson_chi2_test(roi_data, L_hat, scale_hat)
            ks_results = perform_ks_test(roi_data, L_hat, scale_hat)

            print(f"  Pixel Count: {roi_data.size}")
            print(f"  MLE Results: L={L_hat:.4f}, Lambda={lambda_hat:.4f}")
            print(f"  |Mean Error|={mean_abs_error:.4e}, |Var Error|={var_abs_error:.4e}")  # Added Error Print
            print(
                f"  Chi-Square Test: Stat={chi2_results['Chi2_Stat']:.4f}, DOF={chi2_results['DOF']}, P-Value={chi2_results['P_Value']:.4e}, Status={chi2_results['Test_Status']}")
            print(
                f"  K-S Test: Stat={ks_results['KS_Stat']:.4f}, P-Value={ks_results['KS_P_Value']:.4e}, Status={ks_results['KS_Status']}")

            results.append({
                'Image_Name': img_name,
                'Pixel_Count': roi_data.size,
                'L_hat (Shape)': L_hat,
                'Lambda_hat (Rate)': lambda_hat,

                'Mean_Observed': mean_obs,
                'Mean_Theory (L/lambda)': mean_theory,
                'Mean_Abs_Error': mean_abs_error,  # ADDED MEAN ERROR

                'Var_Observed': var_obs,
                'Var_Theory (L/lambda^2)': var_theory,
                'Var_Abs_Error': var_abs_error,  # ADDED VARIANCE ERROR

                'Chi2_Stat': chi2_results['Chi2_Stat'],
                'Chi2_P_Value': chi2_results['P_Value'],
                'KS_Stat': ks_results['KS_Stat'],
                'KS_P_Value': ks_results['KS_P_Value'],
            })


            plt.figure(figsize=(8, 5))
            plt.hist(roi_data, bins=50, density=True, alpha=0.6, color='gray',
                     label='Observed ROI Data (Normalized)')

            x_min_plot = np.max([0, np.min(roi_data)])
            x_max_plot = np.max(roi_data)
            x = np.linspace(x_min_plot, x_max_plot, 100)

            pdf_fitted = gamma.pdf(x, a=L_hat, loc=loc_hat, scale=scale_hat)

            plt.plot(x, pdf_fitted, 'r-', lw=2,
                     label=f'Fitted Gamma PDF')

            plt.title(f'PDF Fit Comparison ({img_name})')
            plt.xlabel('Pixel Intensity Value')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.grid(axis='y', alpha=0.5)
            plot_save_path_pdf = os.path.join(OUTPUT_PLOTS_DIR, f'{base_name}_gamma_fit_pdf.png')
            plt.savefig(plot_save_path_pdf)
            plt.close()

            plt.figure(figsize=(6, 6))
            probplot(roi_data, dist='gamma', sparams=(L_hat, 0, scale_hat), plot=plt)

            plt.title(f'Gamma Q-Q Plot ({img_name})')
            plt.xlabel('Theoretical Gamma Quantiles')
            plt.ylabel('Observed Data Quantiles')

            plot_save_path_qq = os.path.join(OUTPUT_PLOTS_DIR, f'{base_name}_gamma_fit_qq.png')
            plt.savefig(plot_save_path_qq)
            plt.close()

        except Exception as e:
            print(f"Error during MLE, Goodness-of-Fit Tests or plotting for {img_name}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n--- Analysis Complete ---")
        print(f"结果已保存到 {OUTPUT_CSV}")
        print(f"拟合图已保存到 {OUTPUT_PLOTS_DIR} (包含 PDF 和 Q-Q 图)")
        print(f"带标注的图像已保存到 {OUTPUT_ANNOTATED_DIR}")
    else:
        print("\n--- Analysis Complete ---")
        print("未处理任何有效的 ROI 数据。")


if __name__ == '__main__':
    main_analysis()