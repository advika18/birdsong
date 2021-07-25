import streamlit as st
import time
import pandas as pd
import numpy as np
from PIL import Image
import glob
import plotly.express as px
st.set_page_config(page_title='Test',layout='wide')


@st.cache()
def load_data():
    df = pd.read_csv('train.csv')
    return df


# Read in the cereal data
df = load_data()


option = st.selectbox(
        'Bird',
        list(df.species.unique()))

c1, c2= st.beta_columns([1,2])

with c1:
    st.audio(glob.glob(f'downloads/{option}/*.mp3')[0], format='mp3')
    st.image(Image.open(glob.glob(f'downloads/{option}/*.jpg')[0]))


fig = px.scatter_geo(df[df.species==option],
                lat='latitude',lon='longitude', 
                hover_data=['elevation'],
                width=1000, height=600, template='seaborn')

with c2:
    st.plotly_chart(fig)

