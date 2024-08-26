import torch
import numpy as np
import streamlit as st
import pandas as pd
from model_utils import SimpViT, SimpViT_3D, transform, transform_ternary
import os
from streamlit_authenticator import Authenticate

# Constants
IMG_SIZE = 64

# Set up page configuration
st.set_page_config(layout="wide", page_title="Reactivity Ratio Determination Model")

# Custom CSS to improve the styling
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

# Load models from files
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

# Function to make predictions
def predict_model(model, data, data_transform_function, img_size):
    try:
        img_tensor = data_transform_function(np.array(data), img_size=img_size)
        pred = model(img_tensor.unsqueeze(0))
        pred_values = np.power(10, pred.squeeze(0).tolist())  # Convert log predictions back to original scale
        return pred_values
    except Exception as e:
        st.error(f"Prediction failed with error: {e}")
        st.write(f"Data shape: {np.array(data).shape}")
        raise

# Define and hash passwords
def authenticate_users():
    users = {
        "admin": {
            "username": "admin",
            "password": "hash_of_admin_password",
            "email": "admin@example.com"
        },
        "user": {
            "username": "user",
            "password": "hash_of_user_password",
            "email": "user@example.com"
        }
    }
    hashed_passwords = {user: Authenticate.hasher(users[user]["password"]).generate() for user in users}
    authenticator = Authenticate(
        names=users.keys(),
        usernames=[users[user]["username"] for user in users],
        passwords=hashed_passwords,
        emails=[users[user]["email"] for user in users],
        secret_key="some_extremely_secret_key",
        cookie_expiry_days=30,
        cookie_secure=False,
        cookie_httponly=True
    )
    return authenticator

authenticator = authenticate_users()

# Main function defining the application logic
def main():
    add_custom_css()

    # Perform user login
    name, authentication_status, username = authenticator.login("Login", "sidebar")

    if authentication_status:
        st.sidebar.success(f"Welcome {name}!")
        run_app()  # Allow access to the main app functionality
    elif authentication_status == False:
        st.sidebar.error("Username/password is incorrect")
    elif authentication_status == None:
        st.sidebar.warning("Please enter your username and password")

def run_app():
    st.title('Reactivity Ratio Determination Model')

    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'input_method' not in st.session_state:
        st.session_state.input_method = None

    col1, col2 = st.columns(2)
    if col1.button('Binary Model'):
        st.session_state.model_type = 'Binary'
        st.session_state.input_method = None
    if col2.button('Ternary Model'):
        st.session_state.model_type = 'Ternary'
        st.session_state.input_method = None

    if st.session_state.model_type:
        st.write(f"You selected the {st.session_state.model_type} model.")
        st.header(f"Step 2: Input data for {st.session_state.model_type} model")

        col1, col2 = st.columns(2)
        if col1.button('Manual Data Entry'):
            st.session_state.input_method = 'Manual'
            st.write("Input data as decimals in the range [0,1]")
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

            if data_list and st.button(f'Predict({st.session_state.model_type})'):
                prediction = predict_model(
                    binary_model if st.session_state.model_type == 'Binary' else ternary_model,
                    data_list,
                    transform if st.session_state.model_type == 'Binary' else transform_ternary,
                    IMG_SIZE)
                if prediction is not None:
                    display_results(prediction, st.session_state.model_type.lower())


def collect_data(num_sets, model_type):
    data_list = []
    for i in range(int(num_sets)):
        with st.expander(f"Data Set {i+1}"):
            f1 = st.number_input('Input f1', format="%.2f", key=f'f1_{i}_{model_type}')
            f2 = st.number_input('Input f2', format="%.2f", key=f'f2_{i}_{model_type}') if model_type == 'ternary' else None
            total_conv = st.number_input('Input total conversion', format="%.2f", key=f'total_conv_{i}_{model_type}')
            conv1 = st.number_input('Input conv1', format="%.2f", key=f'conv1_{i}_{model_type}')
            conv2 = st.number_input('Input conv2', format="%.2f", key=f'conv2_{i}_{model_type}')
            conv3 = st.number_input('Input conv3', format="%.2f", key=f'conv3_{i}_{model_type}') if model_type == 'ternary' else None
            data_list.append([f1, f2, total_conv, conv1, conv2, conv3] if model_type == 'ternary' else [f1, total_conv, conv1, conv2])
    return data_list

def display_results(pred_values, model_type):
    with st.container():
        st.write(f"Results ({model_type.title()})")
        if model_type == 'binary':
            results_html = f"""
            <div>
                <p>r1 = {pred_values[0]:.2f}, r2 = {pred_values[1]:.2f}</p>
            </div>
            """
            st.markdown(results_html, unsafe_allow_html=True)
        elif model_type == 'ternary':
            results_html = f"""
            <div>
                <p>r12 = {pred_values[0]:<5.2f}, r21 = {pred_values[1]:<5.2f}</p>
                <p>r13 = {pred_values[2]:<5.2f}, r31 = {pred_values[3]:<5.2f}</p>
                <p>r23 = {pred_values[4]:<5.2f}, r32 = {pred_values[5]:<5.2f}</p>
            </div>
            """
            st.markdown(results_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
