# db.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Mot de passe vide
    'database': 'Sellams_namm',
    'port': 3306
}

@st.cache_resource
def get_db_engine():
    try:
        connection_string = f"mysql+pymysql://{DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"❌ Erreur de connexion MySQL : {str(e)}")
        return None

def get_data(query):
    engine = get_db_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        st.error(f"❌ Erreur SQL : {str(e)}")
        return pd.DataFrame()