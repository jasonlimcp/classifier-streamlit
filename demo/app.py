import streamlit as st
import pandas as pd

def main():
    
    st.set_page_config(
        page_title="Stroke Prediction"
    )

    st.title("SVC Prediction Model for Stroke")
    
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
    
    if st.button("Start Modelling"):
        #df = dataprep(df)

        #df_output, roc_plot, classifiers, select_model = classifications(df)
        st.success('The following are evaluation metrics for the classifiers we have run:')
        #st.dataframe(df_output[['f1-score','ROC AUC', 'Train Time']], height = 350)

        #st.write('The best performing model from the list is the **' + select_model + '** model.')

        #st.image([roc_plot],width=600)


    #st.info("Based on your selections, our " + select_model + " model predicted the patient has a **" + pred + "** likelihood of encountering a stroke event.")
    else:
        st.info('Click on the run button above to begin training and fitting the model!')

if __name__ == "__main__":
    main()