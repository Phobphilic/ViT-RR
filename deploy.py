import torch
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f
from model_utils import SimpViT, transform
import os
from io import BytesIO

# -----------------------------
# Streamlit Page Configurations
# -----------------------------
st.set_page_config(layout="wide", page_title="RatioGen Binary Model with Fisher-based JCI")

# -----------------------------
# Custom CSS
# -----------------------------
def add_custom_css():
    css = """
    <style>
        html, body, [class*="css"] {
            margin: 0 auto !important;
            padding: 0 !important;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            width: 100%;
        }
        .css-1d391kg {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        .stButton>button {
            width: 100%;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -----------------------------
# Load Binary Model
# -----------------------------
@st.cache_resource
def load_binary_model():
    model = SimpViT()
    binary_path = 'simpViT_binary.pth'
    full_binary_path = os.path.join(os.path.dirname(__file__), binary_path)
    try:
        model.load_state_dict(torch.load(full_binary_path, map_location=torch.device('cpu')))
        model.eval()
        st.success("Binary model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load binary model: {e}")
    return model

binary_model = load_binary_model()

# -----------------------------
# Fisher-based JCI Calculation
# -----------------------------
def fisher_jci(predictions, alpha=0.05):
    """
    Calculate Fisher-based Joint Confidence Intervals (JCI).
    Supports asymmetric intervals: pred +X / -Y.
    """
    # Number of bootstrap samples
    n_samples = predictions.shape[0]
    # Number of parameters (r1, r2)
    p = predictions.shape[1]

    # Mean prediction
    mean_pred = np.mean(predictions, axis=0)

    # Covariance matrix
    cov_matrix = np.cov(predictions, rowvar=False)

    # Inverse covariance matrix
    inv_cov = np.linalg.inv(cov_matrix)

    # F-value for confidence region
    f_val = f.ppf(1 - alpha, dfn=p, dfd=n_samples - p)

    # JCI radius factor
    jci_radius = np.sqrt(p * (n_samples - 1) / (n_samples - p) * f_val)

    # Ellipse coordinates for visualization
    theta = np.linspace(0, 2 * np.pi, 500)
    unit_circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    ellipse_points = (np.linalg.cholesky(cov_matrix) @ unit_circle * jci_radius).T + mean_pred

    # Asymmetric CI: compute max and min deviation along each axis
    lower_bounds = mean_pred - np.min(predictions, axis=0)
    upper_bounds = np.max(predictions, axis=0) - mean_pred

    return mean_pred, lower_bounds, upper_bounds, ellipse_points

# -----------------------------
# Prediction + Bootstrap
# -----------------------------
def predict_binary_with_jci(model, data, n_iter=500):
    """
    Perform prediction using binary ML model with bootstrap-based Fisher JCI.
    """
    try:
        rng = np.random.default_rng(seed=42)
        predictions = []

        for _ in range(n_iter):
            # Perturb data slightly for bootstrap sampling
            noisy_data = []
            for row in data:
                perturbed_row = [
                    x + rng.normal(0, 0.03 * max(abs(x), 1e-6)) if x is not None else 0.0
                    for x in row
                ]
                noisy_data.append(perturbed_row)

            img_tensor = transform(np.array(noisy_data), img_size=64)
            with torch.no_grad():
                pred = model(img_tensor.unsqueeze(0))
            pred_values = np.power(10, pred.squeeze(0).tolist())
            predictions.append(pred_values)

        predictions = np.array(predictions)

        # Calculate Fisher-based JCI
        mean_pred, lower_bounds, upper_bounds, ellipse_points = fisher_jci(predictions)

        return mean_pred, lower_bounds, upper_bounds, predictions, ellipse_points

    except Exception as e:
        st.error(f"Prediction failed with error: {e}")
        raise

# ================================
# Plotting function (JCI ellipse only)
# ================================
def plot_jci(mean_pred, ellipse_x, ellipse_y):
    """
    Plot JCI ellipse for binary model with center point and dashed boundary.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(ellipse_x, ellipse_y, 'b--', label='95% JCI Ellipse')  # dashed ellipse boundary
    plt.scatter(mean_pred[0], mean_pred[1], color='red', label='Central Prediction', zorder=5)
    plt.xlabel('r1', fontsize=12)
    plt.ylabel('r2', fontsize=12)
    plt.title('Joint Confidence Interval (Fisher-based)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# ================================
# Streamlit app logic
# ================================
def main():
    st.title("RatioGen: Reactivity Ratio Determination (Binary Model + Fisher-based JCI)")

    # File uploader
    file = st.file_uploader("Upload Excel file with data", type=['xlsx'])
    if file:
        data_df = pd.read_excel(file, index_col=0)
        st.write("Uploaded Data Preview:")
        st.dataframe(data_df)

        data_list = data_df.values.tolist()

        if st.button("Predict & Plot JCI"):
            # Get prediction + JCI ellipse
            mean_pred, ci_upper, ci_lower, ellipse_x, ellipse_y = predict_with_fisher_jci(
                binary_model, data_list, transform
            )

            # Show numeric results
            st.subheader("Predicted Reactivity Ratios (Binary)")
            st.write(f"r1 = {mean_pred[0]:.3f}  (+{ci_upper[0]-mean_pred[0]:.3f} / -{mean_pred[0]-ci_lower[0]:.3f})")
            st.write(f"r2 = {mean_pred[1]:.3f}  (+{ci_upper[1]-mean_pred[1]:.3f} / -{mean_pred[1]-ci_lower[1]:.3f})")

            # Show JCI plot
            plot_jci(mean_pred, ellipse_x, ellipse_y)

if __name__ == '__main__':
    main()
