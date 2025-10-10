import torch
import numpy as np
import streamlit as st
import pandas as pd
from model_utils import SimpViT, SimpViT_3D, transform, transform_ternary
from scipy.stats import f
import os
import smtplib
from email.mime.text import MIMEText

IMG_SIZE = 64
ADMIN_EMAIL = "869959223@qq.com" 
EMAIL_PASSWORD = "zhangzexi1!"
REGISTRATION_FILE = "user_registrations.csv"

st.set_page_config(layout="wide", page_title="RatioGen: Reactivity Ratio Determination Model")

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

def send_notification(username, email, institution):
    try:
        msg = MIMEText(f"A new user has registered:\n\nUsername: {username}\nEmail: {email}\nInstitution: {institution}\n\nPlease review and approve in user_registrations.csv.")
        msg['Subject'] = 'New RatioGen Registration Pending Approval'
        msg['From'] = ADMIN_EMAIL
        msg['To'] = ADMIN_EMAIL

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(ADMIN_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        st.warning(f"Could not send email notification: {e}")

def register_user():
    st.sidebar.title("User Registration")
    st.sidebar.write("Please register with your institutional email to request access.")

    username = st.sidebar.text_input("Username")
    email = st.sidebar.text_input("Institutional Email")
    institution = st.sidebar.text_input("Institution Name")

    if os.path.exists(REGISTRATION_FILE):
        df = pd.read_csv(REGISTRATION_FILE)
        if email in df['Email'].values:
            approved = df[df['Email'] == email]['Approved'].values[0]
            if approved:
                st.session_state['registered'] = True
                st.sidebar.success("You are already approved. You may use the app.")
            else:
                st.sidebar.info("Registration pending approval. Please wait.")
        else:
            if st.sidebar.button("Register"):
                if not (email.endswith('.edu') or '.edu.' in email or '.ac.' in email or '.org' in email or email.endswith('.edu.cn')):
                    st.sidebar.error("Please use a valid institutional email address.")
                    return
                new_data = pd.DataFrame([[username, email, institution, False]], columns=['Username', 'Email', 'Institution', 'Approved'])
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(REGISTRATION_FILE, index=False)
                send_notification(username, email, institution)
                st.sidebar.success("Registration submitted. You will be notified once approved.")
    else:
        if st.sidebar.button("Register"):
            if not (email.endswith('.edu') or '.edu.' in email or '.ac.' in email or '.org' in email or email.endswith('.edu.cn')):
                st.sidebar.error("Please use a valid institutional email address.")
                return
            df = pd.DataFrame([[username, email, institution, False]], columns=['Username', 'Email', 'Institution', 'Approved'])
            df.to_csv(REGISTRATION_FILE, index=False)
            send_notification(username, email, institution)
            st.sidebar.success("Registration submitted. You will be notified once approved.")

def show_registrations():
    if os.path.exists(REGISTRATION_FILE):
        df = pd.read_csv(REGISTRATION_FILE)
        total = len(df)
        approved = df['Approved'].sum()
        pending = total - approved
        st.sidebar.write(f"✅ Approved users: {approved}")
        st.sidebar.write(f"⏳ Pending approval: {pending}")

def admin_approval_panel():
    st.subheader("User Approval Panel")
    password = st.text_input("Admin password", type="password")
    if password == "adminpass": # TODO: Replace or secure this
        if os.path.exists(REGISTRATION_FILE):
            df = pd.read_csv(REGISTRATION_FILE)
            for i, row in df.iterrows():
                if not row['Approved']:
                    col1, col2, col3 = st.columns([3, 3, 1])
                    col1.markdown(f"**{row['Username']}** ({row['Email']})")
                    col2.markdown(f"Institution: {row['Institution']}")
                    if col3.button("Approve", key=f"approve_{i}"):
                        df.at[i, 'Approved'] = True
                        df.to_csv(REGISTRATION_FILE, index=False)
                        st.success(f"Approved {row['Username']}")
        else:
            st.info("No registrations found.")

@st.cache_data
def load_models():
    binary_model = SimpViT()
    ternary_model = SimpViT_3D()
    binary_path = 'simpViT_binary.pth'
    ternary_path = 'simpViT_ternary.pth'
    full_binary_path = os.path.join(os.path.dirname(__file__), binary_path)
    full_ternary_path = os.path.join(os.path.dirname(__file__), ternary_path)
    try:
        binary_model.load_state_dict(torch.load(full_binary_path, map_location=torch.device('cpu')))
        ternary_model.load_state_dict(torch.load(full_ternary_path, map_location=torch.device('cpu')))
        binary_model.eval()
        ternary_model.eval()
    except Exception as e:
        st.error(f"Failed to load model with error: {e}")
    return binary_model, ternary_model

binary_model, ternary_model = load_models()

def predict_model(model, data, data_transform_function, img_size, n_iter=200, use_bootstrap=True, df=None):
    try:
        if not use_bootstrap:
            clean_data = np.nan_to_num(np.array(data, dtype=float), nan=0.0)
            img_tensor = data_transform_function(clean_data, img_size=img_size)
            with torch.no_grad():
                pred = model(img_tensor.unsqueeze(0))
            pred_values = np.power(10, pred.squeeze(0).tolist())
            return pred_values, np.zeros_like(pred_values)

        clean_data = np.nan_to_num(np.array(data, dtype=float), nan=0.0)
        seed = int(clean_data.sum() * 1e6) % (2**32 - 1)
        rng = np.random.default_rng(seed)

        predictions = []
        p = model.output_dim if hasattr(model, "output_dim") else None
        n_iter = max(n_iter, (p or 6) + 5)  

        for _ in range(n_iter):
            noisy_data = []
            for row in clean_data:
                perturbed_row = [
                    x + rng.normal(0, 0.03 * max(abs(x), 1e-6)) for x in row
                ]
                noisy_data.append(perturbed_row)

            img_tensor = data_transform_function(np.array(noisy_data), img_size=img_size)
            with torch.no_grad():
                pred = model(img_tensor.unsqueeze(0))
            pred_values = np.power(10, pred.squeeze(0).tolist())
            predictions.append(pred_values)

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        cov_matrix = np.cov(predictions, rowvar=False)

        n = predictions.shape[0] 
        p = len(mean_pred) if df is None else int(df)
        dfd = n - p

        if dfd > 0 and np.isfinite(cov_matrix).all():
            f_val = f.ppf(0.95, dfn=p, dfd=dfd)
            jci_half_width = np.sqrt(np.diag(cov_matrix) * p * f_val / dfd)
        else:
            lower = np.percentile(predictions, 2.5, axis=0)
            upper = np.percentile(predictions, 97.5, axis=0)
            jci_half_width = (upper - lower) / 2.0

        return mean_pred, jci_half_width

    except Exception as e:
        st.error(f"Prediction failed with error: {e}")
        st.write(f"Data shape: {np.array(data).shape}")
        raise

def main():
    add_custom_css()
    st.title('RatioGen: Reactivity Ratio Determination Model')

    tab1, tab2 = st.tabs(["User App", "Admin Panel"])

    with tab1:
        if 'registered' not in st.session_state:
            st.session_state['registered'] = False
        register_user()
        show_registrations()

        if st.session_state['registered']:
            col1, col2 = st.columns(2)
            if col1.button('Binary Model'):
                st.session_state.model_type = 'Binary'
                st.session_state.input_method = None
                st.session_state.trigger_prediction = False
            if col2.button('Ternary Model'):
                st.session_state.model_type = 'Ternary'
                st.session_state.input_method = None
                st.session_state.trigger_prediction = False

            if st.session_state.get('model_type'):
                st.write(f"You selected the {st.session_state.model_type} model.")
                st.header(f"Step 2: Input data for {st.session_state.model_type} model")
                handle_model_interaction()

    with tab2:
        admin_approval_panel()
        
def handle_model_interaction():
    col1, col2 = st.columns(2)
    if col1.button('Manual Data Entry'):
        st.session_state.input_method = 'Manual'
    if col2.button('Upload Excel File'):
        st.session_state.input_method = 'Excel'
        if st.session_state.model_type == 'Binary':
            st.image("excel_format_binary.png", caption="Excel format example for Binary Model")
        elif st.session_state.model_type == 'Ternary':
            st.image("excel_format_ternary.png", caption="Excel format example for Ternary Model")

    if st.session_state.input_method:
        data_list = []

        if st.session_state.input_method == 'Manual':
            num_sets = st.number_input('Number of data sets', min_value=1, value=1, step=1)
            data_list = collect_data(num_sets, st.session_state.model_type.lower())

        elif st.session_state.input_method == 'Excel':
            file = st.file_uploader("Upload Excel file", type=['xlsx'])
            if file:
                try:
                    data_df = pd.read_excel(file, index_col=0)
                    data_list = data_df.values.tolist()
                    if not data_list:
                        st.error("Excel file is empty or formatted incorrectly.")
                    else:
                        st.success("Excel file has been loaded successfully.")
                except Exception as e:
                    st.error("Failed to read Excel file. Please try the example format.")

        use_bootstrap = st.checkbox("Use Bootstrap Sampling for Error Estimation", value=True)

        if data_list:
            if st.button(f'Predict({st.session_state.model_type})'):
                st.session_state.trigger_prediction = True
                st.session_state.last_data = data_list
                st.session_state.use_bootstrap = use_bootstrap

        if st.session_state.get('trigger_prediction', False):
            model = binary_model if st.session_state.model_type == 'Binary' else ternary_model
            transform_fn = transform if st.session_state.model_type == 'Binary' else transform_ternary
            mean_pred, std_pred = predict_model(
                model,
                st.session_state.last_data,
                transform_fn,
                IMG_SIZE,
                use_bootstrap=st.session_state.use_bootstrap
            )
            display_results(mean_pred, std_pred, st.session_state.model_type.lower())
            st.session_state.trigger_prediction = False

def collect_data(num_sets, model_type):
    st.subheader("Upload data file or enter manually")

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key=f"upload_{model_type}")

    data_list = []
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            required_cols_binary = ["set num", "f1", "total conv", "conv(M1)", "conv(M2)"]
            required_cols_ternary = ["set num", "f1", "f2", "total conv", "conv(M1)", "conv(M2)", "conv(M3)"]
            required_cols = required_cols_ternary if model_type == "ternary" else required_cols_binary

            if not all(col in df.columns for col in required_cols):
                return []

            if model_type == "ternary":
                data_list = df[["f1", "f2", "total conv", "conv(M1)", "conv(M2)", "conv(M3)"]].values.tolist()
            else:
                data_list = df[["f1", "total conv", "conv(M1)", "conv(M2)"]].values.tolist()

        except Exception as e:
            return []

    else:
        for i in range(int(num_sets)):
            with st.expander(f"Data Set {i+1}"):
                f1 = st.number_input('Input f1', format="%.2f", key=f'f1_{i}_{model_type}')
                f2 = st.number_input('Input f2', format="%.2f", key=f'f2_{i}_{model_type}') if model_type == 'ternary' else None
                total_conv = st.number_input('Input total conversion', format="%.2f", key=f'total_conv_{i}_{model_type}')
                conv1 = st.number_input('Input conv(M1)', format="%.2f", key=f'conv1_{i}_{model_type}')
                conv2 = st.number_input('Input conv(M2)', format="%.2f", key=f'conv2_{i}_{model_type}')
                conv3 = st.number_input('Input conv(M3)', format="%.2f", key=f'conv3_{i}_{model_type}') if model_type == 'ternary' else None

                if model_type == 'ternary':
                    data_list.append([f1, f2, total_conv, conv1, conv2, conv3])
                else:
                    data_list.append([f1, total_conv, conv1, conv2])

    return data_list

def display_results(mean_pred, jci_half_width, model_type):
    with st.container():
        st.write(f"Results ({model_type.title()})")

        if model_type == 'binary':
            results_html = f"""
            <div>
                <p>r1 = {mean_pred[0]:.3f} ± {jci_half_width[0]:.3f} (95% JCI)</p>
                <p>r2 = {mean_pred[1]:.3f} ± {jci_half_width[1]:.3f} (95% JCI)</p>
            </div>
            """
        elif model_type == 'ternary':
            results_html = f"""
            <div>
                <p>r12 = {mean_pred[0]:.3f} ± {jci_half_width[0]:.3f} (95% JCI),
                   r21 = {mean_pred[1]:.3f} ± {jci_half_width[1]:.3f} (95% JCI)</p>
                <p>r13 = {mean_pred[2]:.3f} ± {jci_half_width[2]:.3f} (95% JCI),
                   r31 = {mean_pred[3]:.3f} ± {jci_half_width[3]:.3f} (95% JCI)</p>
                <p>r23 = {mean_pred[4]:.3f} ± {jci_half_width[4]:.3f} (95% JCI),
                   r32 = {mean_pred[5]:.3f} ± {jci_half_width[5]:.3f} (95% JCI)</p>
            </div>
            """

        st.markdown(results_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
