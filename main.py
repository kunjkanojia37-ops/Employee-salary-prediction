# ===========================
# IMPORT LIBRARIES
# ===========================

import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ===========================
# LOAD DATASET
# ===========================

df = pd.read_csv("/content/sample_data/employee_salary_dataset.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

# ===========================
# DATA PREPROCESSING
# ===========================

df = pd.get_dummies(
    df,
    columns=["Education"],
    dtype=int
)

# ===========================
# FEATURES AND TARGET
# ===========================

x = df.drop("Salary", axis=1)

y = df["Salary"]

# ===========================
# TRAIN TEST SPLIT
# ===========================

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

# ===========================
# BUILD NEURAL NETWORK
# ===========================

model = Sequential([

    Dense(64,activation="relu",input_shape=(x.shape[1],)), # Corrected comma here
    Dense(32,activation="relu"),
    Dense(16,activation="relu"), # Corrected typo 'ctivation' to 'activation'
    Dense(1,activation="linear")
])

# ===========================
# COMPILE MODEL
# ===========================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# ===========================
# TRAIN MODEL
# ===========================

print("Training Started.....")

history = model.fit(
    x_train,
    y_train,
    epochs=25 # Added epochs for better training
)

print("Training Completed.....")

# ===========================
# PREDICTIONS
# ===========================

prediction = model.predict(x_test)

r2 = r2_score(
    y_test,
    prediction
)

print(f"R2 Score : {r2}")

# ===========================
# PREDICT NEW EMPLOYEE
# ===========================

sample_employee = np.array([[
    30,
    5,
    0,
    0,
    1,
    0,
    0
]]) # Converted to numpy array

salary = model.predict(sample_employee)

print(
    f"Predicted Salary : ₹{salary[0][0]:,.2f}"
)

# ===========================
# SAVE MODEL
# ===========================

model.save(
    "employee_salary_model.keras"
)

print("Model Saved Successfully!")
