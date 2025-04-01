# RatioGen: ML-Based Reactivity Ratio Determination for Copolymerizations

**RatioGen** is a machine learning (ML)-based platform for determining reactivity ratios in copolymerizations. By integrating deep learning models trained on millions of "reactivity ratio fingerprints" (*r*FPs), **RatioGen** simplifies and accelerates the process of determining reactivity ratios in both binary and ternary copolymerizations from sparse experimental data.

This tool provides a user-friendly interface deployed through Streamlit, making it easy to explore the reactivity ratios for a wide variety of monomer combinations under varying conditions. By simply inputting experimental data, users can instantly predict the corresponding reactivity ratios.

---

## Key Features

- **Versatility**: Analyze a broad range of monomer combinations with arbitrary feed ratios and conversion data under different reaction conditions, enabling sequence regulation by adjusting factors like temperature and solvent.

- **Easy to Use**: The **RatioGen** web service allows users to upload experimental data and instantly receive reactivity ratio predictions.
  
- **Interactive Platform**: Explore predicted reactivity ratios via interactive chord diagrams:
    - [r₁₂ Chord Diagram](https://codepen.io/Phobphilic/full/ZYzbpJV)
    - [r₂₁ Chord Diagram](https://codepen.io/Phobphilic/full/ogvGLgG)

---

## Live Demo

Check out the live demo of the **RatioGen** platform on Streamlit:  
[**RatioGen web service**](https://ratiogen.streamlit.app/)

---

## Installation and Usage

You can use the **RatioGen** platform directly via the Streamlit app or deploy the model locally by following these steps:

### Requirements
- Python 3.8+
- Streamlit
- TensorFlow / PyTorch (depending on the model)
- Other required libraries (see `requirements.txt`)

### Install Dependencies

Clone this repository and install the necessary dependencies:

```
git clone https://github.com/Phobphilic/ViT-RR.git
cd ViT-RR
pip install -r requirements.txt
```

### Run the App Locally
To run the Streamlit app locally:
```
streamlit run deploy.py
```
This will open the app in your default web browser, where you can start inputting data and exploring the model's predictions.

---

# How It Works
**RatioGen** uses Vision Transformer-based models trained on datasets of *r*FPs for both binary and ternary copolymerizations. The models utilizes experimental inputs (arbitrary feed ratios and conversion data of a monomer combination), then automatically integrate these copolymerization results into an *r*FP matrix and instantly predict corresponding reactivity ratios. The predicted reactivity ratios can be used to better understand and regulate monomer sequences in copolymerization processes.

# Examples
- **Example Input:**
  - Binary (Styrene (St) - Methyl Methacrylate (MMA))
  - Ternary (2,2,3,4,4,4-hexafluorobutyl acrylate (HFBA)-St-MMA)

- **Feed Ratios**: Varying from 0.01 to 0.99 for different monomers

- **Conversions**: Ranging from 0% to 100%

**Example Output:**
- **Binary Reactivity Ratios** (e.g., *r*<sub>12</sub> = 0.40, *r*<sub>21</sub> = 0.35 for St-MMA)

- **Ternary Reactivity Ratios** (e.g., *r*<sub>12</sub> = 0.04, *r*<sub>21</sub> = 0.03, *r*<sub>13</sub> = 0.10, *r*<sub>31</sub> = 1.47, *r*<sub>23</sub> = 0.26, *r*<sub>32</sub> = 0.12)
