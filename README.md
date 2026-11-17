# Enron-Fraud-Detection
# Enron Fraud Detection Using Machine Learning

## Project Overview

The **Enron Fraud Detection System** is a Machine Learning project that analyzes financial and email communication data to identify individuals classified as Persons of Interest (POIs).

The project uses the Enron dataset to explore patterns in financial information and email communication. It applies Machine Learning algorithms to predict whether an individual may be classified as a POI.

A Streamlit web application is also developed to allow users to enter information and view model predictions.

**Note:** A Person of Interest prediction is not proof of fraud or criminal activity. This project is developed for educational purposes.

## Objectives

- Analyze the Enron financial and email dataset.
- Clean and preprocess the collected data.
- Perform Exploratory Data Analysis (EDA).
- Create useful features for Machine Learning.
- Train and compare different classification algorithms.
- Evaluate model performance.
- Develop an interactive Streamlit application.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Machine Learning Algorithms

The project compares the following algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The best-performing model is selected using cross-validation based on F1-score.

## Project Architecture

```text
Dataset
   |
   v
Data Loading
   |
   v
Data Cleaning
   |
   v
Exploratory Data Analysis
   |
   v
Feature Engineering
   |
   v
Model Training
   |
   v
Model Evaluation
   |
   v
Trained Model
   |
   v
Streamlit Web Application
   |
   v
POI Prediction
```

## Project Structure

```text
Enron-Fraud-Detection/
│
├── app.py
│
├── data/
│   ├── final_project_dataset.pkl
│   ├── cleaned_enron_data.csv
│   └── engineered_enron_data.csv
│
├── models/
│   ├── best_model.joblib
│   └── feature_names.joblib
│
├── outputs/
│   ├── graphs/
│   └── evaluation/
│
├── src/
│   ├── load_data.py
│   ├── clean_data.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
│
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/adityat-lab/Enron-Fraud-Detection.git
```

### 2. Open the Project Folder

```bash
cd Enron-Fraud-Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### 5. Run the Project

Run the data processing and model scripts in this order:

```bash
python src/load_data.py
python src/clean_data.py
python src/eda.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py
```

### 6. Launch the Web Application

```bash
streamlit run app.py
```

Open the application in your browser:

`http://localhost:8501`

## Features

- Enron dataset analysis.
- Missing-value handling and data preprocessing.
- Exploratory data visualizations.
- Financial and email-based feature engineering.
- Multiple Machine Learning classification algorithms.
- Cross-validation-based model comparison.
- Model evaluation and prediction.
- Interactive Streamlit web application.

## Dataset

This project uses the Enron dataset containing financial and email-related information about individuals associated with Enron.

The dataset includes features such as salary, bonus, stock value, email communication, and POI labels.

Dataset and background:
- [Carnegie Mellon University Enron Email Dataset](https://www.cs.cmu.edu/~./enron/)
- [Original Fraud Detection Project Inspiration](https://github.com/geekquad/Fraud-Detection)

## Future Improvements

- Improve model performance through hyperparameter tuning.
- Add more Machine Learning algorithms.
- Improve the user interface.
- Deploy the application online.
- Add more detailed prediction explanations.

## Disclaimer

This project is developed for academic and educational purposes. Machine Learning predictions should not be treated as evidence of fraud or wrongdoing.

## Author

**Aditya Thokala**

B.Tech – Artificial Intelligence and Data Science

GitHub: [adityat-lab](https://github.com/adityat-lab)