# db.py - Version SQLite pour Streamlit Cloud
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
            st.error(f"❌ Base de données introuvable")
            return pd.DataFrame()
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        return pd.DataFrame()