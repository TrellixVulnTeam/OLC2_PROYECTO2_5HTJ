import streamlit as st
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from PIL import Image
regr_Linear = linear_model.LinearRegression()


df = ''
data = ''
cox = ''
coy = ''


def main():
    global df, data, x, y, cox, coy, regr_Linear
    st.title('Modelamiento de Datos por @Brayan Prado')
    st.sidebar.header('entrada Parametros de Usuario')

    Algoritmo = ['Regresion Lineal', 'Regresion Polinomial', 'Clasificador Gaussiano', 'Arboles de Desicion', 'Redes Neuronales']
    model = st.sidebar.selectbox('Seleccione el tipo de Algoritmo', Algoritmo)

    Operaciones = ['Graficar puntos', 'Definir función de tendencia', 'Realizar predicción de la tendencia', 'Clasificar por Gauss o árboles de decisión o redes neuronales']
    ne = st.sidebar.selectbox('Seleccione la Operacion a Realizar', Operaciones)

    nivel = st.sidebar.text_input('Ingrese en grado del polinomio')

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
                    plt.scatter(fil, col, color='blue')
                    plt.ylabel(coy)
                    plt.xlabel(cox)
                    if (model == 'Regresion Lineal'):
                        cx = np.array(df[cox]).reshape(-1,1)
                        cy = df[coy]
                        regr_Linear.fit(cx, cy)
                        R = regr_Linear.score(cx,cy)
                        b0 = regr_Linear.intercept_
                        b1 = regr_Linear.coef_
                        if (b0 < 0):
                            st.write(str(b1[0]) + 'x ' + str(b0))
                        elif (b0 > 0):
                            st.write(str(b1[0]) + 'x + ' + str(b0))
                        y_p = b1[0]*cx+b0
                        plt.plot(cx, y_p, color='red')
                    if (model == 'Regresion Polinomial'):
                        if nivel is not None: 
                            cx = np.array(df[cox]).reshape(-1, 1)
                            cy = df[coy]
                            X_train_p, X_test_p, Y_train_p, Y_test_p = train_test_split(cx, cy, test_size=0.15)
                            polgra = PolynomialFeatures(degree= int(nivel))
                            X_train_poli = polgra.fit_transform(X_train_p)
                            X_test_poli = polgra.fit_transform(X_test_p)
                            regr_Linear.fit(X_train_poli, Y_train_p)
                            Y_pred_p = regr_Linear.predict(X_test_poli)
                            plt.plot(X_test_p, Y_pred_p, color='red')
                    plt.savefig('Dispersion.png')
                    plt.close()
                    image = Image.open('Dispersion.png')
                    st.image(image, caption="Grafica de Dispersion")
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