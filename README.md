# Insurance Prediction using Machine Learning

Overview

This project is a Machine Learning based web application developed using Streamlit that predicts whether a person is likely to buy insurance based on their age.

The application uses:

* Linear Regression
* Logistic Regression

to analyze the dataset and visualize prediction results through interactive graphs.

Features

* User-friendly Streamlit interface
* Insurance purchase prediction
* Probability prediction using Logistic Regression
* Model accuracy calculation
* Interactive data visualizations
* Comparison between Linear and Logistic Regression

Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Scikit-learn


Dataset

Dataset used:
`insurance_data.csv`

Dataset Columns

* `age`
* `bought_insurance`

Target Column:

* `bought_insurance`

  * `1` → Bought Insurance
  * `0` → Did Not Buy Insurance


Machine Learning Algorithms

1. Linear Regression

Used to visualize prediction trends based on age.

2. Logistic Regression

Used for binary classification to predict whether a person will buy insurance or not.

---

Project Structure

bash
Insurance-Prediction/
│
├── app.py
├── insurance_data.csv
├── requirements.txt
└── README.md


Installation

1. Clone Repository

bash
git clone https://github.com/your-username/Insurance-Prediction.git

2. Navigate to Project Folder

bash
cd Insurance-Prediction

3. Install Dependencies

bash
pip install -r requirements.txt

Run the Application

bash
streamlit run app.py


After running the command, the application will open automatically in your browser.


Input

The user selects:

* Age using slider input


Output

The application predicts:

* Bought Insurance
  OR
* Did Not Buy Insurance

It also displays:

* Probability of Buying Insurance
* Logistic Regression Accuracy


Visualizations

1. Scatter Plot

Displays the relationship between:

* Age
* Insurance Purchase

2. Linear Regression Graph

Shows the prediction trend using Linear Regression.

3. Logistic Regression Curve

Displays the probability curve for insurance purchase prediction.



Example Prediction

| Age | Prediction            |
| --- | --------------------- |
| 45  | Bought Insurance      |
| 22  | Did Not Buy Insurance |


Future Improvements

* Add multiple input features
* Improve UI design
* Deploy application online
* Add advanced ML algorithms
* Create real-time prediction dashboard

Conclusion

This project demonstrates how Machine Learning algorithms can be used to predict insurance purchase behavior based on age. By combining Logistic Regression, Linear Regression, and data visualization, the application provides an interactive and practical understanding of predictive analytics in real-world scenarios.

