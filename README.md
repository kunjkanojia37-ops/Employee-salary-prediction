# Employee Salary Prediction Using Deep Learning

Overview

This project uses Deep Learning with TensorFlow and Keras to predict employee salaries based on:

- Age
- Experience
- Education Level

The model is trained using a Feed Forward Neural Network (FNN) and demonstrates the complete machine learning workflow including data preprocessing, feature engineering, model training, prediction, and model saving.

---

Technologies Used

- Python
- Pandas
- NumPy
- TensorFlow
- Keras
- Scikit-Learn

---

Dataset Features

Feature| Description
Age| Employee age
Experience| Years of experience
Education| Education qualification
Salary| Target variable

---

Data Preprocessing

The Education column is converted into numerical values using One-Hot Encoding.

Example:

Bachelor

becomes

High School = 0

Diploma = 0

Bachelor = 1

Master = 0

PhD = 0

---

Neural Network Architecture

Input Layer

- 7 Features

Hidden Layers

- Dense(64, ReLU)
- Dense(32, ReLU)
- Dense(16, ReLU)

Output Layer

- Dense(1, Linear)

---

Training

Optimizer:

Adam

Loss Function:

Mean Squared Error (MSE)

Metric:

Mean Absolute Error (MAE)

Epochs:

25

---

Model Performance

The project evaluates performance using:

- R² Score

Example Output:

R² Score: 0.98+

Predicted Salary: ₹52,000

(Note: Results vary depending on dataset and training.)

---

Project Structure

Employee-Salary-Prediction/

├── employee_salary_dataset.csv

├── salary_prediction.py

├── employee_salary_model.keras

├── requirements.txt

└── README.md

---

Future Improvements

- Salary Prediction Web App using Streamlit
- Hyperparameter Tuning
- Larger Real-World Dataset
- Model Deployment

---

Author

Kunj Kanojia

Self-taught Python, Machine Learning and Deep Learning Enthusiast.
