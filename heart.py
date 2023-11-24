import streamlit as st
import pandas as pd
import joblib as jb

st.title('HEART RATE DECTECTOR')
loads=jb.load("model.py")
st.sidebar.header('values of the input variables')

def input_values():
    age=st.sidebar.slider('Age',18,100,1)
    sex=st.sidebar.selectbox('Sex',(0,1))
    chest_pain_type=st.sidebar.slider( 'ChestPainType',0,603,1)
    resting_bp=st.sidebar.selectbox('RestingBP',(0,1))
    cholestrol=st.sidebar.selectbox('Cholesterol',(0,1,2))
    fastingbs = st.sidebar.slider('FastingBS', 60, 202, 1)
    resting_ecg=st.sidebar.selectbox('RestingECG',(0,1))
    max_Hr=st.sidebar.slider('MaxHR',-2,6,1)
    execise_angina=st.sidebar.selectbox('ExerciseAngina',(0,1,2))
    old_peak=st.sidebar.selectbox('Oldpeak',(0,1))
    st_slope = st.sidebar.selectbox('ST_Slope', (0, 1, 2))

    data={'Age':age,
          'Sex':sex,
          'ChestPainType':chest_pain_type,
          'RestingBP':resting_bp,'Cholesterol':cholestrol,'FastingBS':fastingbs,
          'RestingECG':resting_ecg,'MaxHR':max_Hr,'ExerciseAngina':execise_angina,'Oldpeak':old_peak,
          'ST_Slope':st_slope}
    features=pd.DataFrame(data,index=[0])
    return features
input_df=input_values()
st.subheader('the data point for heart rate')
st.write(input_df)
st.subheader('input values')
predictions=loads.predict(input_df)
st.write(predictions)
prob_predictions=loads.predict_proba(input_df)
st.write(prob_predictions)
