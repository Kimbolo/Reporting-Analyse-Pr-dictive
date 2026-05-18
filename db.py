import streamlit as st
import pandas as pd
import sqlite3
import os

@st.cache_data(ttl=3600)
def get_data(query: str) -> pd.DataFrame:
    """Exécute une requête SQL sur SQLite"""
    try:
        db_path = "Sellams_namm.db"
        
        if not os.path.exists(db_path):
            st.error(f"Base de données introuvable")
            return pd.DataFrame()
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        # Ne pas afficher d'erreur ici, laisser le code appelant gérer
        # Retourner un DataFrame vide et lever une exception silencieuse
        return pd.DataFrame()

def get_data_safe(query: str, silent: bool = True) -> pd.DataFrame:
    """
    Exécute une requête SQL et retourne un DataFrame.
    Si silent=True, ne retourne pas d'erreur dans l'UI.
    """
    try:
        db_path = "Sellams_namm.db"
        
        if not os.path.exists(db_path):
            if not silent:
                st.error(f"Base de données introuvable")
            return pd.DataFrame()
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        if not silent:
            st.error(f"Erreur SQL : {str(e)}")
        return pd.DataFrame()

def table_exists(table_name: str) -> bool:
    """Vérifie si une table existe dans la base SQLite"""
    try:
        db_path = "Sellams_namm.db"
        if not os.path.exists(db_path):
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    except:
        return False

def get_all_tables() -> list:
    """Retourne la liste de toutes les tables dans la base"""
    try:
        db_path = "Sellams_namm.db"
        if not os.path.exists(db_path):
            return []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except:
        return []