import streamlit as st
import pandas as pd
import mlflow.pyfunc
import requests
from mlflow.tracking import MlflowClient
import mlflow
import os

label_mapping = {
    0: 'Familia Catedral',
    1: 'Vanet',
    2: 'Haploborolis',
    3: 'Familia Ratake',
    4: 'Familias Wetmore',
    5: 'Vanet',
    6: 'Familia gótica',
}

def set_page_config():
    st.set_page_config(
        page_title="Kubernets Deployments",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yemoncada/mlops_covertype',
            'Report a bug': "https://github.com/yemoncada/mlops_covertype",
            'About': "Kubernets APP Deployment"
        }
    )

def create_header():
    st.title('MLFlow Kubernets (CoverType Dataset)')
    st.subheader('by Yefry Moncada Linares ([@yemoncad](https://github.com/yemoncada?tab=repositories))')

    st.markdown(
        """
        <br><br/>
        La siguiente aplicación proporciona una solución completa para el procesamiento y análisis de datos utilizando
        Streamlit, Python, Docker y Kubernetes, ofreciendo una experiencia de usuario interactiva y una infraestructura robusta
        y escalable para garantizar un rendimiento óptimo en entornos productivos.

        La aplicación se centra en la implementación de servicios que permiten cargar información desde archivos de texto plano
        hacia bases de datos, entrenar modelos de inteligencia artificial, realizar inferencias con modelos previamente entrenados, 
        almacenar información utilizada en el proceso de inferencia en archivos de texto plano y ofrecer una interfaz gráfica interactiva 
        para facilitar la interacción con estos servicios.
        """
        , unsafe_allow_html=True)

    st.markdown('---')

def cargar_base():

    base_url = "http://load-database:8502"

    if st.button('Cargar Base de Datos'):
        with st.spinner('Wait for it...'):
            response = requests.get(f"{base_url}/load_data")
            if response.status_code == 200:
                st.success('La base de Datos ha sido Creada y Cargada!', icon="✅")
                df = pd.read_json(response.json(), orient='records')
                st.dataframe(df.head(50))
            else:
                st.write(f"Ocurrió un error. Código de estado: {response.status_code}")

def create_inferencia():

    stage = "Production"
    os.environ['AWS_ACCESS_KEY_ID'] = '<AWS_ACCESS_KEY_ID>'
    os.environ['AWS_SECRET_ACCESS_KEY'] = '<AWS_ACCESS_KEY>'

    Elevation = st.slider("Elevation", 0, 5000, 2800)
    Slope = st.slider("Slope", 0, 90, 5)
    Horizontal_Distance_To_Hydrology = st.slider("Horizontal Distance To Hydrology", 0, 10000, 250)
    Vertical_Distance_To_Hydrology = st.slider("Vertical Distance To Hydrology", 0, 1000, 50)
    Horizontal_Distance_To_Roadways = st.slider("Horizontal Distance To Roadways", 0, 10000, 1000)
    Hillshade_9am = st.slider("Hillshade 9am", 0, 255, 220)
    Hillshade_Noon = st.slider("Hillshade Noon", 0, 255, 230)
    Horizontal_Distance_To_Fire_Points = st.slider("Horizontal Distance To Fire Points", 0, 10000, 500)

    st.markdown('---')

    cover_data = {
        "Elevation": Elevation,
        "Slope": Slope,
        "Horizontal_Distance_To_Hydrology": Horizontal_Distance_To_Hydrology,
        "Vertical_Distance_To_Hydrology": Vertical_Distance_To_Hydrology,
        "Horizontal_Distance_To_Roadways": Horizontal_Distance_To_Roadways,
        "Hillshade_9am": Hillshade_9am,
        "Hillshade_Noon": Hillshade_Noon,
        "Horizontal_Distance_To_Fire_Points": Horizontal_Distance_To_Fire_Points,
    }

    # connects to the Mlflow tracking server that you started above
    mlflow.set_tracking_uri("http://10.43.102.110:5000/")
    # Configurar el cliente de MLflow
    client = MlflowClient()

    model_names = []

    for rm in client.search_registered_models():
        model_name = rm.name
        model_names.append(model_name)
    
    if len(model_names) == 0:
        st.warning('No existen modelos registrados')
    else:
        data = pd.DataFrame([cover_data])
        st.write(data)

        st.markdown('---')

        model_name = st.selectbox('Seleccione el modelo para predecir', model_names)

        if st.button('Ejecutar predicción'):
            with st.spinner('Ejecutando...'):
                try:
                    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
                    predict = model.predict(data)[0]

                    # Obtener la última versión del modelo en la etapa de producción
                    model_version = client.get_latest_versions(model_name, stages=["Production"])[0]

                    # Obtener el run_id y el run_name del modelo en producción
                    run_id = model_version.run_id
                    run = client.get_run(run_id)
                    run_name = run.data.tags["mlflow.runName"]

                    st.markdown('---')
                    # Imprimir el run_name del modelo en producción
                    st.header(f"El tipo de cubierta forestal es: {predict} - {label_mapping[predict]}")
                    st.header(f"Nombre del Modelo en producción: {run_name}")  
                except:
                    st.warning('No existe ningun modelo en producción')


def main():
    set_page_config()
    create_header()
    create_inferencia()

if __name__ == "__main__":
    main()
