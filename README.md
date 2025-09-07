# Breast cancer diagnosis predictor

## 🧠 Overview

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f31bfe15-0531-4d71-a374-a5952470e3e1" />


This application uses a pre-trained Logistic Regression model to classify tumors using features extracted from **Breast Cancer Wisconsin dataset**. The app lets users input numerical features of cell nuclei (e.g. radius mean, perimeter worst, texture se, etc.) using sliders on the sidebar. These inputs are then scaled and passed to the model, which generates a prediction.

Additionally, a **radar chart** visualizes the relative scale of the features for better interpretability .

The Breast Cancer Diagnosis app is a machine learning-powered tool designed to assist medical professionals in diagnosing breast cancer. Using a set of measurements, the app predicts whether a breast mass is benign or malignant. It provides a visual representation of the input data using a radar chart and displays the predicted diagnosis and probability of being benign or malignant. The app can be used by manually inputting the measurements or by connecting it to a cytology lab to obtain the data directly from a machine. The connection to the laboratory machine is not a part of the app itself.

The app was developed as a machine learning exercice from the public dataset [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). Note that this dataset may not be reliable as this project was developed for educational purposes in the field of machine learning only and not for professional use.

A live version of the application can be found on [Streamlit Community Cloud](https://alejandro-ao-streamlit-cancer-predict-appmain-uitjy1.streamlit.app/). 

# Breast Cancer Prediction Streamlit App

A Machine Learning web application built with **Streamlit** that predicts whether a breast tumor is **Benign** or **Malignant** based on cell nuclei measurements. The goal of this project is to demonstrate how data science and machine learning can be integrated into an interactive web app for healthcare-related predictions.

---

## 🔍 Purpose

The main objectives of this project are:

- To build a user-friendly interface with **Streamlit** for real-time prediction of breast cancer type.
- To demonstrate how **machine learning models** can be deployed using simple interfaces.
- To visualize cell feature patterns using interactive **Plotly radar charts**.
- To encourage responsible use of predictive tools in healthcare and emphasize that this application should not replace professional medical diagnosis.

---

## 🧾 Dataset Used

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository
- **Number of features:** 30 real-valued input features
- **Label:** Diagnosis (Malignant = 1, Benign = 0)

---

## ⚙️ Technologies Used

| Technology            | Description                                         |
|-----------------------|-----------------------------------------------------|
| Python 3.x            | Programming language                                |
| Streamlit             | Web application framework                           |
| Scikit-Learn          | Machine Learning model, scaling                     |
| Pandas, NumPy         | Data manipulation                                   |
| Plotly                | Radar chart visualization                           |
| Pickle                | Model and scaler serialization                      |

---

## 🧾 Features

- ✅ Sidebar sliders for user input of 30 tumor features
- ✅ Real-time prediction (Benign vs Malignant)
- ✅ Scaled input using the same scaler that trained the model
- ✅ Radar chart showing feature magnitude
- ✅ Color-coded diagnosis label (Green for Benign, Red for Malignant)
- ✅ Custom CSS styling for a better user interface

---

## 🧠 How the Prediction Works

1. User inputs feature values using sidebar sliders.
2. Inputs are converted into a NumPy array.
3. The array is scaled using the same `StandardScaler()` used during model training.
4. A trained `LogisticRegression` model predicts the class label.
5. Probabilities for both classes are displayed.
6. Result is wrapped in a styled box with CSS for better UX.

---

## 📷 Screenshots

You can include screenshots here using:

```markdown
![Homepage Screenshot](images/homepage.png)
![Prediction Screenshot](images/prediction.png)


## Installation

You can run this inside a virtual environment to make it easier to manage dependencies. I recommend using `conda` to create a new environment and install the required packages. You can create a new environment called `breast-cancer-diagnosis` by running:

```bash
conda create -n breast-cancer-diagnosis python=3.10 
```

Then, activate the environment:

```bash
conda activate breast-cancer-diagnosis
```

Then, activate the environment:

```bash
conda activate breast-cancer-diagnosis
```

To install the required packages, run:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies, including Streamlit, OpenCV, and scikit-image.

## Usage
To start the app, simply run the following command:

```bash
streamlit run app/main.py
```

This will launch the app in your default web browser. You can then upload an image of cells to analyze and adjust the various settings to customize the analysis. Once you are satisfied with the results, you can export the measurements to a CSV file for further analysis.
