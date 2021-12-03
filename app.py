import streamlit as st 
import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
#from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from detect import *


def main():
    # Render the readme as markdown using st.markdown.
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url('https://unsplash.com/photos/ZiQkhI7417A');
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    hide_streamlit_style = """
    
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show Instructions", "Diagnose Yourself","Face Detection",'Sentiment Analysis', 'Entity Extraction', "About Us"])
    if app_mode == "Show Instructions":
        st.sidebar.success('To continue select an option.')
        st.title("instructions")
        st.subheader("Welcome to Diagnose.ai! Here is how to use our app! :nerd_face: ")
        
        st.write("Please Select Diagnose Yourself from the sidebar")
        st.markdown("""---""")

    elif app_mode == "About Us":
        st.title("About Us")
        st.header("Welcome to ML-HUB")
        st.subheader("We try to make ML accessible to everyone")
        st.write("ML-Hub.ai is a Machine Learning solution for businesses and individuals wanting to make use of the State-of-the-art Neural Networks in Computer Vision and Natural Language Processing domains. ")
        st.write("We provide simple drag-and-drop approach to problems that require tremendous coding and knowledge of data science domain.")
        st.write("Instead of spending time building neural networks and models, you can focus on what matters the most -- Results.")
        st.write("We take care of all the modelling and processing behind the scenes on our Heroku server.")
        st.write("We provide solutions for :")
        st.write("Object Detection | Face Detection | Sentiment Analysis | Entity Extraction")
        st.markdown("""---""")
        st.header("Meet Our Team Members")
        st.write("Aditya Patkar - Lead Developer")
        st.write("Rakesh Kumar - Frontend Developer")
        st.write("Radhika Sahastrabudhe - Design Expert")
        st.write("Apoorva Parashar - Backend and Diagrams ")
    elif app_mode == "Diagnose Yourself":
        choice = st.radio("", ("Show Demo", "Browse an Image"))
        st.write()

        if choice == "Browse an Image":
            st.set_option('deprecation.showfileUploaderEncoding', False)
            img= st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
            if img is not None:
                image = Image.open(img).convert('RGB')
                image = preprocess(image)
                model = torch.load('/Users/aditya/Desktop/Covidxray/covidxray', map_location=torch.device('cpu'))
                dataset = z()
                img = image
                col1, col2 = st.columns(2)

                col1.subheader("Original Image")
                fig = plt.figure(figsize = (15,15))
                plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
                col1.pyplot(fig, use_column_width=True)
                st.markdown("""---""")
                st.text(f"Predicted by AI :  {predict_image(img, model)}")
                st.markdown("""---""")
                st.text("This is just for preliminary Diagnosis")
                st.text("If you suffer from the symptoms, please see a doctor / radiologist.")
        else:
            image = Image.open("/Users/aditya/Downloads/data/Viral Pneumonia-1.png").convert('RGB')
            image = preprocess(image)
            model = torch.load('/Users/aditya/Desktop/Covidxray/covidxray', map_location=torch.device('cpu'))
            dataset = z()
            img = image
            col1, col2 = st.columns(2)

            col1.subheader("Your X-Ray")
            fig = plt.figure(figsize = (15,15))
            plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
            col1.pyplot(fig, use_column_width=True)
            st.markdown("""---""")
            st.text(f"Predicted by AI :  {predict_image(img, model)}")
            st.markdown("""---""")
            st.text("This is just for preliminary Diagnosis")
            st.text("If you suffer from the symptoms, please see a doctor / radiologist.")



if __name__ == '__main__':
    main()