import streamlit as st
import pandas as pd
import sqlite3
import pandas as pd
import numpy as np
from src.data_cleansing import dataprep
from src.classification_models import classifications

def main():
    
    st.set_page_config(
        page_title="Predicting Survival Outcomes",
        layout="wide"
    )

    st.title("Predicting Survival Outcomes for Coronary Artery Disease Patients")
    
    connection = sqlite3.connect('data/survive.db')
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    query = connection.execute("SELECT * From survive")
    cols = [column[0] for column in query.description]
    survive = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
    
    form = st.sidebar.form("form", clear_on_submit=False)
    with form:        
        st.subheader("Select Values for Input Variables")

        gender = st.radio(
                "Gender",
            ('Male', 'Female')
        )

        age = st.number_input(
            label = "Age",
            min_value = 18,
            max_value = 99,
            value=50,
            step=1
        )

        bmi = st.number_input(
            label = "Body-Mass Index (BMI)",
            min_value = 10,
            max_value = 50,
            value=25,
            step=1
        )

        blood_pressure = st.number_input(
            label = "Blood Pressure",
            min_value = 40,
            max_value = 200,
            value=130,
            step=1
        )

        smoking = st.radio(
                "Smoker",
            ('Yes', 'No')
        )

        diabetes = st.selectbox(
            label='Diabetic',
            options=('No', 'Pre-diabetes','Diabetes')
            )

        submit = st.form_submit_button("Start Modelling")

    if submit==False:
        st.info("Select input variables on the left for the model to predict a survival outcome.")
        st.warning("Once the button is clicked, the 'survive' dataset will be pre-processed and subsequently fitted through the different classifiers.")
        
        st.image('src/image/kavanaugh-unsplash.jpg')
        st.caption("Image Credits: Bret Kavanaugh 2019 via Unsplash")
        

    if submit == True:
    
        with st.spinner('This will take a bit. Take a sip (or two) of coffee..'):
            df = dataprep(survive)

            df_output, roc_plot, classifiers, select_model = classifications(df)

            st.success('The following are evaluation metrics for the classifiers we have run:')
            st.dataframe(df_output[['f1-score','ROC AUC', 'Train Time']], height = 350)

            st.write('The best performing model from the list is ' + select_model + ', based on a product of f-1 score and ROC-AUC score, followed by ranking by model fitting time.')

            st.image([roc_plot],width=600)

        if gender == 'Male':
            gender = True
        else:
            gender = False

        if smoking == 'Yes':
            smoking = True
        else:
            smoking = False

        if diabetes == 'No':
            diabetes = 0
        elif diabetes == 'Pre-diabetes':
            diabetes = 1
        else:
            diabetes = 2

        x = [
            gender,
            smoking,
            diabetes,
            age,
            int(df["Ejection Fraction"].mean()),
            df["Sodium"].mean(),
            df["Creatinine"].mean(),
            df["Platelets"].mean(),
            df["Creatine phosphokinase"].mean(),
            blood_pressure,
            df["Hemoglobin"].mean(),
            bmi
            ]

        x = pd.DataFrame(x).T
        #x = x.reshape(1, -1)

        if select_model == 'Logistic Regression':
            predictor = classifiers[0]
        elif select_model == 'Random Forest':
            predictor = classifiers[1]
        elif select_model == 'K-Nearest Neighbours':
            predictor = classifiers[2]
        else:
            predictor = classifiers[3]

        y_pred = predictor.predict(x)
        pred = ord(y_pred.tostring())

        if pred==1:
            pred = 'Survived.'
        elif pred==0:
            pred = 'Did not Survive.'
        else:
            pred = 'Error - Contact Dev'

        st.success("Based on your selections, the " + select_model + " model predicted the patient: " + pred)

if __name__ == "__main__":
    main()