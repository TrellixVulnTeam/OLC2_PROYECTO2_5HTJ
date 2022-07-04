import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from PIL import Image
regr_Linear = linear_model.LinearRegression()
regr_poly = linear_model.LinearRegression()

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

    pred = st.sidebar.text_input('Ingrese el valor a predecir :')

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
                            cx = np.asarray(df[cox]).reshape(-1, 1)
                            cy = df[coy]
                            polgra = PolynomialFeatures(degree= int(nivel))
                            x_t = polgra.fit_transform(cx)
                            regr_Linear.fit(x_t, cy)
                            #plt.plot(x_t, cy)
                            y_pred = regr_Linear.predict(x_t)
                            plt.plot(cx, y_pred, color='red')
                    plt.savefig('Dispersion.png')
                    plt.close()
                    image = Image.open('Dispersion.png')
                    st.image(image, caption="Grafica de Dispersion")
                elif ((model == 'Regresion Lineal') or (model == 'Regresion Polinomial')) and (ne == 'Definir función de tendencia'):
                    if (model == 'Regresion Lineal'):
                        cx = np.array(df[cox]).reshape(-1,1)
                        cy = df[coy]
                        regr_Linear.fit(cx, cy)
                        R = regr_Linear.score(cx,cy)
                        st.write('Pendiente : ')
                        b1 = regr_Linear.coef_
                        st.write(b1)
                        st.write('Intercepto : ')
                        b0 = regr_Linear.intercept_
                        st.write(b0)
                        st.write('Funcion de tendencia central')
                        if (b0 < 0):
                            st.write(str(b1[0]) + 'x ' + str(b0))
                        elif (b0 > 0):
                            st.write(str(b1[0]) + 'x + ' + str(b0))
                        st.write('Coeficiente de Correlacion')
                        st.write(R)
                    elif (model == 'Regresion Polinomial'):
                        if nivel is not None: 
                            cx = np.asarray(df[cox]).reshape(-1, 1)
                            cy = df[coy]
                            polgra = PolynomialFeatures(degree= int(nivel))
                            x_t = polgra.fit_transform(cx)
                            regr_poly.fit(x_t, cy)
                            y_pred = regr_poly.predict(x_t)
                            st.write('Valor de los Coeficientes :')
                            coer = regr_poly.coef_
                            st.write(coer)
                            st.write('Valor del Intercepto :')
                            st.write(regr_poly.intercept_)
                            st.write('Coeficiente de Correlacion :')
                            st.write(regr_poly.score(x_t, cy))
                            st.write('Funcion de Tendencia Central :')
                            concatenacion = ""
                            da = len(coer) - 1
                            while da>=0:
                                if da != 0:
                                    concatenacion = concatenacion + str(coer[da]) + 'X^' + str(da) + '+'
                                else:
                                    concatenacion = concatenacion + str(coer[da])
                                da = da - 1
                            st.write(concatenacion)
                elif ((model == 'Regresion Lineal') or (model == 'Regresion Polinomial')) and (ne == 'Realizar predicción de la tendencia'):
                    if (model == 'Regresion Lineal'):
                        cx = np.array(df[cox]).reshape(-1,1)
                        cy = df[coy]
                        pred1 = int(pred)
                        regr_Linear.fit(cx, cy)
                        b0 = regr_Linear.intercept_
                        b1 = regr_Linear.coef_
                        y_p = b1[0]*pred1+b0
                        st.write('El valor que se predijo es :')
                        st.write(y_p)
                    elif (model == 'Regresion Polinomial'):
                        if nivel is not None:
                            cx = np.asarray(df[cox]).reshape(-1, 1)
                            cy = df[coy]
                            polgra = PolynomialFeatures(degree= int(nivel))
                            x_t = polgra.fit_transform(cx)
                            regr_poly.fit(x_t, cy)
                            y_pred = regr_poly.predict(x_t)
                            concatenar = 0
                            coer = regr_poly.coef_
                            da = len(regr_poly.coef_)-1
                            while da>=0:
                                concatenar = concatenar + round((coer[da]),0)*int(pred)**(int(da))
                                da = da -1
                            st.write(concatenar)

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