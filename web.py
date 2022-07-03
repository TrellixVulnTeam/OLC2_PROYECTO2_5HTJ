from pyparsing import line
import streamlit as st
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from PIL import Image
regr_Linear = linear_model.LinearRegression()

df = ''
data = ''
cox = ''
coy = ''


def main():
    global df, data, x, y, cox, coy
    st.title('Modelamiento de Datos por @Brayan Prado')
    st.sidebar.header('entrada Parametros de Usuario')

    Algoritmo = ['Regresion Lineal', 'Regresion Polinomial', 'Clasificador Gaussiano', 'Arboles de Desicion', 'Redes Neuronales']
    model = st.sidebar.selectbox('Seleccione el tipo de Algoritmo', Algoritmo)

    Operaciones = ['Graficar puntos', 'Definir función de tendencia', 'Realizar predicción de la tendencia', 'Clasificar por Gauss o árboles de decisión o redes neuronales']
    ne = st.sidebar.selectbox('Seleccione la Operacion a Realizar', Operaciones)

    data = st.file_uploader("Seleccione el Archivo", type=["csv", "xls", "xlsx", "json"])
    if data is not None:
        spli = os.path.splitext(data.name)
        if spli[1] == '.csv':
            df = pd.read_csv(data)
            st.dataframe(df)
            x  = df.head()
            y  = df.head()
            cox = st.selectbox('seleccione X: ', x.columns)
            coy = st.selectbox('seleccione Y: ', y.columns) 
            if st.sidebar.button('Realizar accion'):
                if ((model == 'Regresion Lineal') or (model == 'Regresion Polinomial')) and (ne == 'Graficar puntos'):
                    fil = df[cox].tolist()
                    col = df[coy].tolist()
                    plt.scatter(fil, col)
                    plt.ylabel(coy)
                    plt.xlabel(cox)
                    plt.savefig('Dispersion.png')
                    plt.close()
                    image = Image.open('Dispersion.png')
                    st.image(image, caption="Grafica de Dispersion")
                    #st.pyplot('Dispersion.png')
        elif spli[1] == '.xls':
            df = pd.read_excel(data)
            x  = df.head()
            y  = df.head()
            cox = st.selectbox('seleccione X: ', x)
            coy = st.selectbox('seleccione Y: ', y)
        elif spli[1] == '.xlsx':
            df = pd.read_excel(data)
            x  = df.head()
            y  = df.head()
            cox = st.selectbox('seleccione X: ', x)
            coy = st.selectbox('seleccione Y: ', y)
        elif spli[1] == '.json':
            df = pd.read_json(data) 
            x  = df.head()
            y  = df.head()
            cox = st.selectbox('seleccione X: ', x)
            coy = st.selectbox('seleccione Y: ', y)

    

if __name__== '__main__':
    main()