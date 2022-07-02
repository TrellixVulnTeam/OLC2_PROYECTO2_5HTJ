import streamlit as st
import pickle
import pandas as pd
import os

data = ''
x = ''
y = ''

with open('lin_reg.pkl', 'rb') as li:
    lin_reg = pickle.load(li)

with open('log_reg.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)

with open('svc_m.pkl', 'rb') as sv:
    svm_mo = pickle.load(sv)

def main():
    global data, x, y
    st.title('Modelamiento de Datos por @Brayan Prado')
    st.sidebar.header('entrada Parametros de Usuario')

    Algoritmo = ['Regresion Lineal', 'Regresion Polinomial', 'Clasificador Gaussiano', 'Arboles de Desicion', 'Redes Neuronales']
    model = st.sidebar.selectbox('Seleccione el tipo de Algoritmo', Algoritmo)

    Operaciones = ['Graficar puntos', 'Definir función de tendencia', 'Realizar predicción de la tendencia', 'Clasificar por Gauss o árboles de decisión o redes neuronales']
    ne = st.sidebar.selectbox('Seleccione la Operacion a Realizar', Operaciones)

    data = st.file_uploader("Seleccione el Archivo", type=["csv", "xls", "xlsx", "json"])
    if data is not None:
        spli = os.path.splitext(data.name)
        st.write(spli[1])
        if spli[1] == '.csv':
            st.write(spli[0])
            df = pd.read_csv(data)
        elif spli[1] == '.xls':
            st.write(spli[0])
            df = pd.read_excel(data)
        elif spli[1] == '.xlsx':
            st.write(spli[0])
            df = pd.read_excel(data)
        elif spli[1] == '.json':
            st.write(spli[0])
            df = pd.read_json(data) 

    

if __name__== '__main__':
    main()