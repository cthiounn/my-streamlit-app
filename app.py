import streamlit as st
import joblib
import time
from PIL import Image


with open("model/randomForestModel.pkl","rb") as model_file:
    model = joblib.load(model_file)
    
    
def predict_with_random_forest(data):
    return model.predict(data)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def load_icon(icon_name):
    st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)
    
    
    
def main():
    """Random Forest App
    With Streamlit

    """

    st.title("Random Forest App")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:grey;text-align:center;">Streamlit App </h2>
    </div>

    """
    st.markdown(html_temp,unsafe_allow_html=True)
    load_css('icon.css')
    load_icon('people')

    X = st.text_input("Enter data please")
        if st.button("Predict"):
            result = predict_with_random_forest([X])

    st.success('Result is {}'.format(result))