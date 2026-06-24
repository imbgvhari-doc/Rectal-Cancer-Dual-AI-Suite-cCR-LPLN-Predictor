🎯 Rectal Cancer Dual AI Suite: cCR & LPLN Predictor

Overview

This repository contains a dual-model Machine Learning pipeline and Clinical Decision Support Application designed to predict two critical outcomes following neoadjuvant chemoradiotherapy (nCRT) in locally advanced rectal cancer:

Complete Clinical Response (cCR): Predicting primary tumor resolution to safely select candidates for the organ-preserving "Watch-and-Wait" pathway.

Lateral Pelvic Lymph Node (LPLN) Sterilization: Predicting regional nodal resolution to safely determine the necessity of a highly morbid Lateral Pelvic Node Dissection (LPND).

Developed by Dr. Harishankar BGV (Department of Surgical Oncology, Panimalar Medical College Hospital and Research Institute, Chennai).

💡 Clinical Discovery: AI Feature Pruning

During the training of the LPLN Sterilization Predictor on the institutional registry, the XGBoost algorithm performed aggressive mathematical feature pruning that revealed a profound clinical insight.

Despite conventional surgical focus on baseline nodal size and PET avidity, the AI mathematically zero-weighted these features.

PET Avidity: Nearly 100% of pathological nodes in the cohort were PET-avid, making it a "zero-variance" feature incapable of separating responders from non-responders.

Anatomical Sizing: Baseline MRI node measurements were highly variable and frequently missing, prompting the algorithm to reject them to prevent overfitting.

The Biological Conclusion: The AI successfully predicts regional nodal sterilization using only the primary tumor's baseline anatomical markers (Age, Wall Thickness, and Tumor Height). This mathematically demonstrates that the underlying biological radiosensitivity of the primary tumor dictates the radiosensitivity of its regional lymph nodes, rendering the pre-treatment size of the node largely irrelevant to its final sterilization probability.

⚠️ Data Privacy & Reproducibility

Strict Institutional Data Protection: The original training datasets contain sensitive patient health information (PHI). To comply with patient privacy and ethical guidelines, the real clinical datasets are strictly excluded from this repository.

For open-source reproducibility, a dummy_dataset.csv is provided. This file contains randomly generated data mimicking the statistical distribution of the original institutional cohort, allowing researchers to test the application framework locally.

Repository Structure

train_models_unified.py: Unified Machine Learning pipeline using XGBoost to train both the cCR and LPLN predictors.

app.py: Streamlit-based web application for real-time Clinical Decision Support with integrated SHAP explainability.

requirements.txt: Python library dependencies.

dummy_dataset.csv: Synthesized mock data for testing.

Installation & Usage

Clone the repository:

git clone [https://github.com/YourUsername/Rectal-Cancer-Dual-AI-Suite.git](https://github.com/YourUsername/Rectal-Cancer-Dual-AI-Suite.git)
cd Rectal-Cancer-Dual-AI-Suite


Install dependencies:

pip3 install -r requirements.txt


Train the models (Generates the .pkl files):

python3 train_models_unified.py


Launch the Clinical Support Web App:

streamlit run app.py


Academic Citation

If utilizing this framework or discussing the feature-pruning biological insights for clinical research, please cite:

Harishankar BGV, et al. Machine Learning-Driven Prediction of Complete Clinical Response and LPLN Sterilization in Rectal Cancer: Augmenting Organ Preservation. Presented at HAI Conclave, Bengaluru, 2026.
