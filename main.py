# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from sklearn.datasets import load_breast_cancer
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import accuracy_score

# # # App Title
# # st.title("Breast Cancer Detection App")
# # st.write("A machine learning-based app to detect breast cancer using the sklearn dataset.")

# # # Load the dataset
# # @st.cache_resource
# # def load_data():
# #     data = load_breast_cancer()
# #     df = pd.DataFrame(data.data, columns=data.feature_names)
# #     df['label'] = data.target
# #     return df, data

# # df, data = load_data()

# # # Display dataset information
# # st.subheader("Dataset Overview")
# # if st.checkbox("Show raw dataset"):
# #     st.write(df)

# # st.write(f"Number of instances: {df.shape[0]}")
# # st.write(f"Number of features: {df.shape[1] - 1} (excluding target label)")

# # # Feature-target split
# # X = df.drop(columns="label", axis=1)
# # Y = df["label"]

# # # Train-test split
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# # # Train the model
# # model = LogisticRegression(max_iter=10000)
# # model.fit(X_train, Y_train)

# # # Evaluate the model
# # Y_pred = model.predict(X_test)
# # accuracy = accuracy_score(Y_test, Y_pred)

# # st.subheader("Model Performance")
# # st.write(f"Accuracy on test data: {accuracy:.2f}")

# # # User prediction
# # st.subheader("Make a Prediction")
# # user_input = {}
# # for feature in data.feature_names:
# #     user_input[feature] = st.number_input(feature, value=float(X.mean()[feature]))

# # # Convert user input to DataFrame
# # user_data = pd.DataFrame([user_input])

# # if st.button("Predict"):
# #     prediction = model.predict(user_data)[0]
# #     st.write(f"The prediction is: {'Benign' if prediction == 1 else 'Malignant'}")
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Custom CSS for Background Image
# def add_bg_image():
#     st.markdown(
#         """
#         <style>
#         .stApp {
#             background-image: url('https://www.transparenttextures.com/patterns/cubes.png'); 
#             background-size: cover;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Add the background image
# add_bg_image()

# # App Title
# st.title("Breast Cancer Detection App")
# st.write("A machine learning-based app to detect breast cancer using the sklearn dataset.")

# # Load the dataset
# @st.cache_data
# def load_data():
#     data = load_breast_cancer()
#     df = pd.DataFrame(data.data, columns=data.feature_names)
#     df['label'] = data.target
#     return df, data

# df, data = load_data()

# # Display dataset information
# st.subheader("Dataset Overview")
# if st.checkbox("Show raw dataset"):
#     st.write(df)

# st.write(f"Number of instances: {df.shape[0]}")
# st.write(f"Number of features: {df.shape[1] - 1} (excluding target label)")

# # Feature-target split
# X = df.drop(columns="label", axis=1)
# Y = df["label"]

# # Train-test split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# # Train the model
# @st.cache_resource
# def train_model():
#     model = LogisticRegression(max_iter=10000)
#     model.fit(X_train, Y_train)
#     return model

# model = train_model()

# # Evaluate the model
# Y_pred = model.predict(X_test)
# accuracy = accuracy_score(Y_test, Y_pred)

# st.subheader("Model Performance")
# st.write(f"Accuracy on test data: {accuracy:.2f}")

# # User prediction
# st.subheader("Make a Prediction")
# user_input = {}
# for feature in data.feature_names:
#     user_input[feature] = st.number_input(feature, value=float(X.mean()[feature]))

# # Convert user input to DataFrame
# user_data = pd.DataFrame([user_input])

# if st.button("Predict"):
#     prediction = model.predict(user_data)[0]
#     st.write(f"The prediction is: {'Benign' if prediction == 1 else 'Malignant'}")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# # Custom CSS for Background Image
# def add_bg_image():
#     st.markdown(
#         """
#         <style>
#         .stApp {
#             background-image: url('DALL·E 2024-11-19 01.15.23 - A visually appealing and professional background image for a breast cancer detection project. The image features a soft pink and white gradient to sym.png'); /* Replace with your image file name */
#             background-size: cover;
#             background-repeat: no-repeat;
#             background-attachment: fixed;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Add the background image
# add_bg_image()

# App Title
st.title("Breast Cancer Detection App")
st.write("A machine learning-based app to detect breast cancer using the sklearn dataset.")

# Load the dataset
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['label'] = data.target
    return df, data

df, data = load_data()

# Display dataset information
st.subheader("Dataset Overview")
if st.checkbox("Show raw dataset"):
    st.write(df)

st.write(f"Number of instances: {df.shape[0]}")
st.write(f"Number of features: {df.shape[1] - 1} (excluding target label)")

# Feature-target split
X = df.drop(columns="label", axis=1)
Y = df["label"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
@st.cache_resource
def train_model():
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, Y_train)
    return model

model = train_model()

# Evaluate the model
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

st.subheader("Model Performance")
st.write(f"Accuracy on test data: {accuracy:.2f}")

# User prediction
st.subheader("Make a Prediction")
user_input = {}
for feature in data.feature_names:
    user_input[feature] = st.number_input(feature, value=float(X.mean()[feature]))

# Convert user input to DataFrame
user_data = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = model.predict(user_data)[0]
    st.write(f"The prediction is: {'Benign' if prediction == 1 else 'Malignant'}")
