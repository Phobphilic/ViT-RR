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
                    x + rng.normal(0, 0.015 * max(abs(x), 1e-6)) if x is not None else 0.0
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

# -----------------------------
# Visualization of JCI Ellipse
# -----------------------------
def plot_jci(predictions, mean_pred, ellipse_points):
    """
    Plot bootstrap points, Fisher-based JCI ellipse, and predicted point.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(predictions[:, 0], predictions[:, 1], alpha=0.3, label="Bootstrap Samples")
    plt.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r-', lw=2, label="95% JCI Ellipse")
    plt.scatter(mean_pred[0], mean_pred[1], color="black", s=60, label="Predicted Point")
    plt.xlabel("r1")
    plt.ylabel("r2")
    plt.title("Fisher-based JCI Ellipse (Binary Model)")
    plt.legend()
    plt.grid(True)

    # Save to BytesIO for Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    add_custom_css()
    st.title("RatioGen: Binary Model with Fisher-based JCI")

    # File upload
    file = st.file_uploader("Upload Excel file for binary model", type=['xlsx'])
    if file:
        data_df = pd.read_excel(file, index_col=0)
        st.write("### Uploaded Data")
        st.dataframe(data_df)
        data_list = data_df.values.tolist()

        if st.button('Run Prediction'):
            with st.spinner("Running predictions and calculating Fisher-based JCI..."):
                mean_pred, lower_bounds, upper_bounds, predictions, ellipse_points = predict_binary_with_jci(
                    binary_model, data_list, n_iter=500
                )

                # Display results
                st.subheader("Prediction Results")
                st.write(f"**r1 = {mean_pred[0]:.3f} (+{upper_bounds[0]:.3f} / -{lower_bounds[0]:.3f})**")
                st.write(f"**r2 = {mean_pred[1]:.3f} (+{upper_bounds[1]:.3f} / -{lower_bounds[1]:.3f})**")

                # Plot JCI ellipse
                buf = plot_jci(predictions, mean_pred, ellipse_points)
                st.image(buf, caption="Fisher-based JCI Ellipse", use_column_width=True)

if __name__ == '__main__':
    main()
