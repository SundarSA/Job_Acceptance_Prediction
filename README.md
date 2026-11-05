# Job Acceptance Prediction System

## Project Overview

The Job Acceptance Prediction System is a Machine Learning project developed using Python and Streamlit.

It analyzes candidate academic qualifications, technical skills, work experience, and employment preferences to predict whether a candidate is likely to be placed or not placed.

The project demonstrates data preprocessing, exploratory data analysis, feature engineering, machine learning model development, model evaluation, and interactive prediction.

## Project Objectives

- Analyze candidate information and placement-related factors.
- Clean and preprocess candidate data.
- Perform exploratory data analysis.
- Build a supervised Machine Learning classification model.
- Evaluate model performance.
- Develop an interactive Streamlit dashboard for placement prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Altair
- Joblib
- Jupyter Notebook

## Machine Learning Model

**Algorithm:** Random Forest Classifier

**Model Pipeline:**
- ColumnTransformer
- OneHotEncoder for categorical features
- Numerical feature passthrough
- Random Forest classification

**Dataset:**
- Total records: 50,000
- Input features: 25
- Target classes: 2

Target values:
- `0` – Not Placed
- `1` – Placed

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 89.55% |
| Precision | 81% |
| Recall | 86% |
| F1-Score | 83% |

These results are based on the model's test dataset.

## Streamlit Dashboard Features

- Interactive candidate information input.
- Placement prediction.
- Placed and Not Placed probability display.
- Placement probability gauge.
- Placement probability level indicator.
- Model performance visualization.
- Feature importance visualization.
- Candidate prediction history.
- Candidate name and prediction date/time tracking.

## Project Structure

```text
Job_Acceptance_Prediction/
│
├── .streamlit/
│   └── config.toml
│
├── .vscode/
│   └── settings.json
│
├── Job_Acceptance_Analysis.ipynb
├── app.py
├── placement_prediction_model.pkl
├── requirements.txt
└── README.md
```

Note: The trained model (`placement_prediction_model.pkl`) and dataset are not included in this public repository.

To regenerate the trained model, open `Job_Acceptance_Analysis.ipynb`, run the data preparation and model training cells, execute GridSearchCV, and run the final model-saving cell.

## How to Run the Project

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required Python packages.
4. Obtain or regenerate the trained model file.
5. Run the Streamlit application.

Run the application using:

```bash
python -m streamlit run app.py
```

## Project Status

The current working Streamlit application and Jupyter Notebook are published on GitHub.

Further documentation and reproducibility improvements are in progress.

## Author

**SundarSA**

GitHub: https://github.com/SundarSA

## Disclaimer

This project is developed for educational and analytical purposes. Predictions are model estimates and should not be treated as guaranteed placement outcomes.