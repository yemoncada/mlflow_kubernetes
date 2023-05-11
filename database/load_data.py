from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from fastapi.responses import JSONResponse
from sqlalchemy_utils import database_exists, create_database
import sqlalchemy as db
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

app = FastAPI()

def create_db(data):

    """
    Esta función se utiliza para crear y cargar los datos de entrenamiento desde una base de datos MySQL. 
    
    Parámetros:
    -----------
    data : str
        El nombre de la tabla que contiene los datos de entrenamiento.
        
    Retorno:
    -------
    df_db : Pandas DataFrame
        Los datos de entrenamiento cargados desde la base de datos.
        
    Excepciones:
    ------------
    HTTPException: si la tabla especificada no existe en la base de datos.
    """

    if data == 'covertype_data':

        connection_string= "postgresql+psycopg2://"+ os.environ["POSTGRES_USER"] + ":" + os.environ["POSTGRES_PASSWORD"] + "@" + os.environ["POSTGRES_SERVER"] + "/" + os.environ["POSTGRES_DB"]
        query = 'SELECT * FROM '+data

        engine = create_engine(connection_string)

        # Create the database if it does not exist
        if not database_exists(engine.url):
            create_database(engine.url)

            print("[INFO] No existe la base de datos")

            # Re-create the engine to include the new database
            engine = create_engine(connection_string)

            with engine.connect() as conn:

                #impute the data
                df = pd.read_csv('data/covertype_train.csv')
                df.to_sql(con=engine, index_label='id', name='covertype_data', if_exists='replace')
                df_db = pd.read_sql_query(sql=text(query), con=conn)
        
        else:
            with engine.connect() as conn:
                if (db.inspect(conn).has_table('covertype_data')==True):
                    print("[INFO] Ya existe la base de datos")
                    df_db = pd.read_sql_query(sql=text(query), con=conn)
        
        return df_db
    
@app.get('/load_data')
def load_database():
    try:
        data = os.environ["POSTGRES_DB"]
        df_db = create_db(data)
        return JSONResponse(content=df_db.to_json(orient='records'))
  
    except:
        raise HTTPException(status_code=500, detail="Fallo Carga en la Base de Datos: "+data)

