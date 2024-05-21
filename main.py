# popularity_predictor_app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('Customer-Churn-Records.csv')
#st.write("DataFrame Columns:", df.columns)

# Features and target variable
numeric_features = ['Complain','Age','Balance','EstimatedSalary','HasCrCard','Tenure','NumOfProducts','IsActiveMember','CreditScore','Satisfaction_Score']
target = 'Exited'

# Ensure only numeric features are included
df_numeric = df[numeric_features + [target]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_numeric.drop(columns=[target]), df_numeric[target], test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)


# Function to predict popularity
def predict_popularity(Complain,Age,Balance,EstimatedSalary,HasCrCard,Tenure,NumOfProducts,IsActiveMember,CreditScore,Satisfaction_Score):
    input_data = pd.DataFrame({
        'Complain':[Complain],
        'Age': [Age],
        'Balance': [Balance],
        'EstimatedSalary': [EstimatedSalary],
        'HasCrCard': [HasCrCard],
        'Tenure': [Tenure],
        'NumOfProducts': [NumOfProducts],
        'IsActiveMember': [IsActiveMember],
        #'Gender': [Gender],
        #'Geography': [Geography],
        #'CardType': [CardType],
        'CreditScore': [CreditScore],
        'Satisfaction_Score': [Satisfaction_Score]
    })

    prediction = rf_classifier.predict(input_data)
    return prediction[0]

# Streamlit app
st.title(' Bank Customer Churn Predictor')

# Image



# Input form for user to enter feature values



# Slider for Age
Age = st.slider('Age', min_value=1, max_value=100, step=1)

# Slider for Balance
Balance = st.slider('Balance', min_value=0, max_value=250899, step=1)

# Slider for EstimatedSalary
EstimatedSalary = st.slider('EstimatedSalary', min_value=11, max_value=199992, step=1)

# Slider for CreditScore
CreditScore = st.slider('CreditScore', min_value=350, max_value=850, step=1)


# Tenure dropdown
Tenure = st.selectbox('Tenure', options=list(range(11)))

# NumOfProducts dropdown
NumOfProducts = st.selectbox('NumOfProducts', options=list(range(1, 5)))

# HasCrCard dropdown
HasCrCard_options = {0: 'No', 1: 'Yes'}
HasCrCard = st.selectbox('HasCrCard', options=list(HasCrCard_options.keys()), format_func=lambda x: HasCrCard_options[x])

# IsActiveMember dropdown
IsActiveMember_options = {0: 'No', 1: 'Yes'}
IsActiveMember = st.selectbox('IsActiveMember', options=list(IsActiveMember_options.keys()), format_func=lambda x: IsActiveMember_options[x])

# Complain dropdown
Complain_options = {0: 'No', 1: 'Yes'}
Complain = st.selectbox('Complain', options=list(Complain_options.keys()), format_func=lambda x: Complain_options[x])


# Satisfaction Score dropdown
Satisfaction_Score = st.selectbox('Satisfaction_Score', options=list(range(1, 6)))

# Predict button
if st.button('Predict Churn'):
    prediction = predict_popularity(Complain,Age,Balance,EstimatedSalary,HasCrCard,Tenure,NumOfProducts,IsActiveMember,CreditScore,Satisfaction_Score)
    if prediction == 1:
        st.success('Customer exited the bank')
    else:
        st.success('Customer did not exit the bank')



