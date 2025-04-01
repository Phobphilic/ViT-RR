# RatioGen: ML-Based Reactivity Ratio Prediction for Copolymerizations
**RatioGen** is an efficient and versatile machine learning (ML)-based platform for determining reactivity ratios in copolymerizations. With the integration of deep learning models trained on millions of "reactivity ratio fingerprints" (*r*FPs), **RatioGen** enables fast, accurate, and convenient determination of reactivity ratios in both binary and ternary copolymerizations from sparse experimental data.

This tool provides a user-friendly interface deployed through Streamlit, making it easy to explore the reactivity ratios for a wide variety of monomer combinations under varying conditions. By simply inputting experimental data, users can instantly predict the corresponding reactivity ratios.

# Key Features
**Instant Prediction**: Predict reactivity ratios for copolymerizations from sparse experimental data (e.g., random monomer combinations, arbitrary feed ratios, conversion data) in milliseconds.

**Interactive Platform**: Explore predicted reactivity ratios via interactive chord diagrams (https://codepen.io/Phobphilic/pen/ZYzbpJV for *r*<sub>12</sub>; 
https://codepen.io/Phobphilic/pen/ogvGLgG for *r*<sub>21</sub>).

<p class="codepen" data-height="300" data-default-tab="html,result" data-slug-hash="ZYzbpJV" data-pen-title="Chord" data-user="Phobphilic" style="height: 300px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; border: 2px solid; margin: 1em 0; padding: 1em;">
  <span>See the Pen <a href="https://codepen.io/Phobphilic/pen/ZYzbpJV">
  Chord</a> by Phobphilic (<a href="https://codepen.io/Phobphilic">@Phobphilic</a>)
  on <a href="https://codepen.io">CodePen</a>.</span>
</p>
<script async src="https://public.codepenassets.com/embed/index.js"></script>

**Versatility**: Analyze a broad range of monomer combinations under different reaction conditions, enabling sequence regulation by adjusting factors like temperature and solvent.

**Easy to Use**: Streamlined interface to upload experimental data and instantly provide reactivity ratio predictions.

# Live Demo
Check out the live demo of the **RatioGen** platform on Streamlit:
https://ratiogen.streamlit.app/

# Installation and Usage
You can use the **RatioGen** platform directly via the Streamlit app, or deploy the model locally by following these steps:

**Requirements**
Python 3.8+

Streamlit

TensorFlow / PyTorch (depending on the model)

Other required libraries (see requirements.txt)

**Install Dependencies**
Clone this repository and install the necessary dependencies:
git clone https://github.com/Phobphilic/ViT-RR.git
cd ViT-RR
pip install -r requirements.txt

**Run the App Locally**
To run the Streamlit app locally:
streamlit run deploy.py
This will open the app in your default web browser, where you can start inputting data and exploring the model's predictions.

# How It Works
**RatioGen** uses Vision Transformer-based models trained on datasets of *r*FPs for both binary and ternary copolymerizations. The models utilizes experimental inputs (arbitrary feed ratios and conversion data of a monomer combination), then automatically integrate these copolymerization results into an rFP matrix and instantly predict corresponding reactivity ratios. The predicted reactivity ratios can be used to better understand and regulate monomer sequences in copolymerization processes.

# Examples
**Example Input:**
1. **Monomer Combinations**: Binay (Styrene (St)-Methyl Methacrylate (MMA)) ; Ternary (2,2,3,4,4,4-hexafluorobutyl acrylate (HFBA)-St-MMA)
2. **Feed Ratios**: Varying from 0.01 to 0.99 for different monomers
3. **Conversions**: Ranging from 0% to 100%

**Example Output:**
**Binary Reactivity Ratios** (e.g., *r*<sub>12</sub> = 0.40, *r*<sub>21</sub> = 0.35 for St-MMA)

**Ternary Reactivity Ratios** (e.g., *r*<sub>12</sub> = 0.04, *r*<sub>21</sub> = 0.03, *r*<sub>13</sub> = 0.10, *r*<sub>31</sub> = 1.47, *r*<sub>23</sub> = 0.26, *r*<sub>32</sub> = 0.12)
