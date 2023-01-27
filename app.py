import streamlit as st

import pickle
import numpy as np
model=pickle.load(open('model.h5','rb'))


def predict_species(sep_len,sep_width,petal_len,petal_width):
    input=np.array([[sep_len,sep_width,petal_len,petal_width]]).astype(np.float64)
    pred=model.predict(input)
    return int(pred)

def main():
    
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Flower Species Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    #sep_len = st.text_input("Sepal Length","Type Here")
    sep_len=st.slider('Enter Sepal Length',0.0,10.0)
    #sep_width = st.text_input("Sepal Width","Type Here")
    sep_width=st.slider('Enter Sepal Width',0.0,10.0)
    #petal_len = st.text_input("Petal Length","Type Here")
    petal_len=st.slider('Enter Petal Length',0.0,10.0)
    #petal_width = st.text_input("Petal Width","Type Here")
    petal_width=st.slider('Enter Petal Width',0.0,10.0)

    
    setosa_html="""  
      <div style="background-color:#33FF39;padding:10px >
       <h2 style="color:white;text-align:center;"> The flower spices is SETOSA</h2>
       </div>
    """
    versicolor_html="""  
      <div style="background-color:#DD33FF;padding:10px >
       <h2 style="color:white;text-align:center;"> The flower spices is VERSICOLOR</h2>
       </div>
    """
    virginica_html="""  
      <div style="background-color:#336BFF;padding:10px >
       <h2 style="color:white;text-align:center;"> The flower spices is VIRGINICA</h2>
       </div>
    """
    

    if st.button("Predict"):
        output=predict_species(sep_len,sep_width,petal_len,petal_width)
        #st.success('The probability of this species is {}'.format(output))

        if output == 0:
            st.markdown(setosa_html,unsafe_allow_html=True)
        elif output== 1:
            st.markdown(versicolor_html,unsafe_allow_html=True)
        else :
            st.markdown(virginica_html,unsafe_allow_html=True)
        

if __name__=='__main__':
    main()