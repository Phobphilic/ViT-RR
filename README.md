RatioGen: ML-Based Reactivity Ratio Prediction for Copolymerizations
RatioGen is an efficient and versatile machine learning (ML)-based platform for determining reactivity ratios in copolymerizations. The platform leverages "reactivity ratio fingerprints" (rFPs) in two- and three-dimensional matrices (2D/3D rFPs) to predict reactivity ratios in both binary and ternary copolymerizations from sparse experimental data. With the integration of deep learning models trained on millions of rFPs, RatioGen enables fast, accurate, and convenient predictions without the need for exhaustive experimental data.

This tool provides a user-friendly interface deployed through Streamlit, making it easy to explore the reactivity ratios for a wide variety of monomer combinations under varying conditions. By simply inputting experimental data, users can instantly predict the corresponding reactivity ratios.

Key Features
Instant Prediction: Predict reactivity ratios for copolymerizations from sparse experimental data (e.g., random monomer combinations, arbitrary feed ratios, conversion data) in milliseconds.

ML Model Integration: The model is trained on millions of reactivity ratio fingerprints (rFPs), ensuring highly accurate predictions.

Interactive Platform: Explore predicted reactivity ratios via interactive chord diagrams and parity plots for both binary and ternary systems.

Versatility: Analyze a broad range of monomer combinations and reaction conditions, enabling sequence regulation by adjusting factors like temperature and solvent.

Easy to Use: Streamlined interface to upload experimental data and instantly receive reactivity ratio predictions.

Live Demo
Check out the live demo of the RatioGen platform on Streamlit:
https://ratiogen.streamlit.app/

Installation and Usage
You can use the RatioGen platform directly via the Streamlit app, or deploy the model locally by following these steps:

Requirements
Python 3.8+

Streamlit

TensorFlow / PyTorch (depending on the model)

Other required libraries (see requirements.txt)

Install Dependencies
Clone this repository and install the necessary dependencies:

git clone https://github.com/your-username/RatioGen.git
cd RatioGen
pip install -r requirements.txt
Run the App Locally
To run the Streamlit app locally:

streamlit run app.py
This will open the app in your default web browser, where you can start inputting data and exploring the model's predictions.

How It Works
RatioGen uses deep learning models trained on a large dataset of "reactivity ratio fingerprints" (rFPs) for both binary and ternary copolymerizations. The model predicts reactivity ratios by mapping experimental inputs such as monomer conversions and feed ratios to corresponding rFP matrices. The predicted ratios (e.g., r₁₂, r₂₁) can be used to better understand and regulate monomer sequences in copolymerization processes.

The key stages in the development of the ML model are:

Data Collection: Experimental data and reactivity ratios are collected and stored in an rFP database.

Model Training: Deep learning models are trained to correlate the rFP matrices with corresponding reactivity ratios.

Prediction: The trained model is applied to new experimental data to predict reactivity ratios in milliseconds.

Examples
Example Input:
Monomer Combinations: Styrene (St), Methyl Methacrylate (MMA), Methyl Acrylate (MA)

Feed Ratios: Varying from 0.1 to 0.9 for different monomers

Conversions: Ranging from 1% to 99%

Example Output:
Binary Reactivity Ratios (e.g., r₁₂ = 0.40, r₂₁ = 0.35 for St-MMA)

Ternary Reactivity Ratios (e.g., r₁₂ = 0.39, r₂₁ = 2.24 for MA-MMA)

Contributions
We welcome contributions to this project! If you have suggestions for improving the platform or the model, please feel free to submit a pull request or open an issue. We are particularly interested in expanding the platform’s ability to predict reactivity ratios for more complex systems or incorporating new algorithms for better prediction accuracy.

Citing
If you use RatioGen in your research, please cite our work as follows:

mathematica
复制
[Insert Citation Here]
License
This project is licensed under the MIT License - see the LICENSE file for details.
