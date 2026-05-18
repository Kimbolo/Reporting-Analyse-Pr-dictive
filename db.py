# # db.py
# import streamlit as st
# import pandas as pd
# from sqlalchemy import create_engine, text

# DB_CONFIG = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '',  # Mot de passe vide
#     'database': 'Sellams_namm',
#     'port': 3306
# }

# @st.cache_resource
# def get_db_engine():
#     try:
#         connection_string = f"mysql+pymysql://{DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
#         engine = create_engine(connection_string, pool_pre_ping=True)
#         with engine.connect() as conn:
#             conn.execute(text("SELECT 1"))
#         return engine
#     except Exception as e:
#         st.error(f"❌ Erreur de connexion MySQL : {str(e)}")
#         return None

# def get_data(query):
#     engine = get_db_engine()
#     if engine is None:
#         return pd.DataFrame()
#     try:
#         df = pd.read_sql_query(query, engine)
#         engine.dispose()
#         return df
#     except Exception as e:
#         st.error(f"❌ Erreur SQL : {str(e)}")
#         return pd.DataFrame()
# db.py - Version hybride (MySQL en local, SQLite sur le cloud)
import streamlit as st
import pandas as pd
import sqlite3
import os

# Détection de l'environnement
IS_CLOUD = os.environ.get('STREAMLIT_CLOUD', False) or 'STREAMLIT_SHARING' in os.environ

@st.cache_data(ttl=3600)
def get_data(query: str) -> pd.DataFrame:
    """Exécute une requête SQL - Adapté selon l'environnement"""
    
    if IS_CLOUD:
        # Mode Cloud : SQLite
        return get_data_sqlite(query)
    else:
        # Mode Local : MySQL
        return get_data_mysql(query)

def get_data_sqlite(query: str) -> pd.DataFrame:
    """Version SQLite pour le cloud"""
    try:
        db_path = "Sellams_namm.db"
        
        if not os.path.exists(db_path):
            st.error(f"❌ Fichier '{db_path}' introuvable")
            return pd.DataFrame()
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"❌ Erreur SQLite : {str(e)}")
        return pd.DataFrame()

def get_data_mysql(query: str) -> pd.DataFrame:
    """Version MySQL pour le développement local"""
    from sqlalchemy import create_engine, text
    
    try:
        connection_string = f"mysql+pymysql://root@localhost:3306/Sellams_namm"
        engine = create_engine(connection_string, pool_pre_ping=True)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        df = pd.read_sql_query(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        st.error(f"❌ Erreur MySQL : {str(e)}")
        return pd.DataFrame()

def test_connection():
    """Test la connexion selon l'environnement"""
    if IS_CLOUD:
        return os.path.exists("Sellams_namm.db")
    else:
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine("mysql+pymysql://root@localhost:3306/Sellams_namm")
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except:
            return False