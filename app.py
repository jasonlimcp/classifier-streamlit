import streamlit as st
import pandas as pd
from src.data_preparation import dataprep
from src.classification_models import classifications

def main():
    
    st.set_page_config(
        page_title="Predicting Stroke Events",
        layout="wide"
    )

    st.title("_Predicting Patient Stroke Events_")
    
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
    
    form = st.sidebar.form("form", clear_on_submit=False)
    with form:        
        st.subheader("Select Values for Input Variables")

        gender = st.radio(
                "Gender",
            ('Male', 'Female')
        )

        age = st.number_input(label='Age',
            min_value = 1,
            max_value = 85,
            value=50,
            step=1)

        hypertension = st.radio(
                "Hypertension",
            ('Yes', 'No'),
            help='History of hypertension symptoms or high blood pressure'
        )

        heart_disease = st.radio(
                "Heart Disease",
            ('Yes', 'No'),
            help='History of heart disease'
        )

        ever_married = st.radio(
                "Marriage History",
            ('Married/Was Married', 'Never Married'),
            help = 'Has the patient ever been married?'
        )
        
        Is_Urban_Residence = st.radio(
                "Location of Residence",
            ('Urban', 'Rural')
        )

        avg_glucose_level = st.slider(
            label = "Glucose Level",
            min_value = 50.0,
            max_value = 280.0,
            value=130.0,
            step=0.1
        )

        bmi = st.number_input(
            label = "Body-Mass Index (BMI)",
            min_value = 10.0,
            max_value = 50.0,
            value=25.0,
            step=1.0
        )

        work_type = st.selectbox(
            label='Type of Work',
            options=('Self-Employed', 'Government Job','Private Industry','Not of Working Age')
            )

        smoking_status = st.selectbox(
            label='Smoking',
            options=('Never Smoked', 'Ex-Smoker','Smoker','Unknown')
            )

        submit = st.form_submit_button("Run the prediction")
        
        st.warning("_Please Read: All content on this site is solely for educational purposes on machine learning concepts. Content should not be construed or relied upon as medical information, advice or opinion in any form._")

    if submit==False:
        st.info("Select input variables on the left for the classification model to predict likelihood of a patient's stroke event.")
        
        st.image('src/image/kavanaugh-unsplash.jpg')
        st.caption("Image Credits: Bret Kavanaugh 2019 via Unsplash")
        

    if submit == True:
    
        with st.spinner('This will take a bit. Take a sip (or two) of coffee..'):
            df = dataprep(df)

            df_output, roc_plot, classifiers, select_model = classifications(df)

            st.success('The following are evaluation metrics for the classifiers we have run:')
            st.dataframe(df_output[['f1-score','ROC AUC', 'Train Time']], height = 350)

            st.write('The best performing model from the list is the **' + select_model + '** model.')

            st.image([roc_plot],width=600)

        if gender == 'Male':
            gender = 1
        else:
            gender = 0

        if hypertension == 'Yes':
            hypertension = 1
        else:
            hypertension = 0

        if heart_disease == 'Yes':
            heart_disease = 1
        else:
            heart_disease = 0

        if ever_married == 'Married/Was Married':
            ever_married = 1
        else:
            ever_married = 0

        if Is_Urban_Residence == 'Urban':
            Is_Urban_Residence = 1
        else:
            Is_Urban_Residence = 0

        if work_type == 'Self-Employed':
            work_type_Government_Job=0
            work_type_Not_of_Working_Age=0
            work_type_Private_Industry=0
            work_type_Self_employed=1
        elif work_type == 'Government Job':
            work_type_Government_Job=1
            work_type_Not_of_Working_Age=0
            work_type_Private_Industry=0
            work_type_Self_employed=0
        elif work_type == 'Private Industry':
            work_type_Government_Job=0
            work_type_Not_of_Working_Age=0
            work_type_Private_Industry=1
            work_type_Self_employed=0
        else:
            work_type_Government_Job=0
            work_type_Not_of_Working_Age=1
            work_type_Private_Industry=0
            work_type_Self_employed=0

        if smoking_status == 'Never Smoked':
            smoking_status_Former_Smoker=0
            smoking_status_Never_Smoked=1
            smoking_status_Smoker=0
            smoking_status_Unknown=0
        elif smoking_status == 'Ex-Smoker':
            smoking_status_Former_Smoker=1
            smoking_status_Never_Smoked=0
            smoking_status_Smoker=0
            smoking_status_Unknown=0
        elif smoking_status == 'Smoker':
            smoking_status_Former_Smoker=0
            smoking_status_Never_Smoked=0
            smoking_status_Smoker=1
            smoking_status_Unknown=0
        else:
            smoking_status_Former_Smoker=0
            smoking_status_Never_Smoked=0
            smoking_status_Smoker=0
            smoking_status_Unknown=1
        
        x = [gender,
        age,
        hypertension,
        heart_disease,
        ever_married,
        Is_Urban_Residence,
        avg_glucose_level,
        bmi,
        work_type_Government_Job,
        work_type_Not_of_Working_Age,
        work_type_Private_Industry,
        work_type_Self_employed,
        smoking_status_Former_Smoker,
        smoking_status_Never_Smoked,
        smoking_status_Smoker,
        smoking_status_Unknown]

        if select_model == 'Logistic Regression':
            predictor = classifiers[0]
        elif select_model == 'Random Forest':
            predictor = classifiers[1]
        elif select_model == 'K-Nearest Neighbours':
            predictor = classifiers[2]
        else:
            predictor = classifiers[3]

        x = pd.DataFrame(x).T

        y_pred = predictor.predict(x)
        pred = ord(y_pred.tostring())

        if pred==1:
            pred = 'higher'
        elif pred==0:
            pred = 'lower'
        else:
            pred = 'Error - Contact Dev'

        st.info("Based on your selections, our " + select_model + " model predicted the patient has a **" + pred + "** likelihood of encountering a stroke event.")

        with st.expander("FAQ"):
            st.write("**How is the best-performing model selected?**")
            st.write('The model selected for prediction is the highest-ranked on a product of f-1 score and ROC-AUC score, followed by ranking by model fitting time.')
            st.write("**Where can I see the source code for this site?**")
            st.write('The Github repository for this site can be found [here](https://github.com/jasonlimcp/classifier-streamlit).')
            st.write("**What is the data source used for the Stroke prediction?**")
            st.write('Data used in this classification exercise is from the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) on Kaggle.')


if __name__ == "__main__":
    main()