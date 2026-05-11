import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv("insurance_data.csv")

X = df[['age']]
y = df['bought_insurance']

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# LINEAR REGRESSION
# -----------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X)

# -----------------------------------
# LOGISTIC REGRESSION
# -----------------------------------

logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)

logistic_output = logistic_model.predict(X_test)

# -----------------------------------
# ACCURACY
# -----------------------------------

accuracy = accuracy_score(y_test, logistic_output)

# -----------------------------------
# STREAMLIT UI
# -----------------------------------

st.title("Insurance Prediction App")

st.write("Model Accuracy:",
         round(accuracy * 100,2),
         "%")

# -----------------------------------
# AGE INPUT
# -----------------------------------

age = st.slider(
    "Enter Age",
    18,
    60,
    30
)

test_age = [[age]]

prediction = logistic_model.predict(test_age)

probability = logistic_model.predict_proba(test_age)

# -----------------------------------
# PREDICTION OUTPUT
# -----------------------------------

st.write("Selected Age:", age)

if prediction[0] == 1:
    st.success("Bought Insurance")
else:
    st.error("Did Not Buy Insurance")

st.write(
    "Probability of Buying:",
    round(probability[0][1] * 100,2),
    "%"
)

# -----------------------------------
# LOGISTIC CURVE
# -----------------------------------

age_range = np.linspace(
    df.age.min(),
    df.age.max(),
    300
).reshape(-1,1)

logistic_curve = logistic_model.predict_proba(
    age_range
)[:,1]

# ===================================
# GRAPH 1
# SCATTER PLOT
# ===================================

fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.scatter(
    df.age,
    df.bought_insurance,
    color='black'
)

ax1.set_xlabel("Age")
ax1.set_ylabel("Bought Insurance")
ax1.set_title("Scatter Plot")

ax1.grid(True)

st.pyplot(fig1)

# ===================================
# GRAPH 2
# LINEAR REGRESSION
# ===================================

fig2, ax2 = plt.subplots(figsize=(8,6))

ax2.plot(
    df.age,
    linear_pred,
    color='blue',
    linewidth=3
)

ax2.set_xlabel("Age")
ax2.set_ylabel("Linear Prediction")
ax2.set_title("Linear Regression")

ax2.grid(True)

st.pyplot(fig2)

# ===================================
# GRAPH 3
# LOGISTIC REGRESSION
# ===================================

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(
    age_range,
    logistic_curve,
    color='red',
    linewidth=3
)

# Green prediction point
ax3.scatter(
    age,
    prediction[0],
    color='green',
    s=120
)

ax3.set_xlabel("Age")
ax3.set_ylabel("Probability")
ax3.set_title("Logistic Regression")

ax3.grid(True)

st.pyplot(fig3) 