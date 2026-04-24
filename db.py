from sqlalchemy import create_engine
import pandas as pd

# connexion MySQL
engine = create_engine("mysql+pymysql://root@127.0.0.1:3306/sellams_namm")

def get_data(query):
    """Exécute une requête SQL et retourne un DataFrame"""
    return pd.read_sql(query, engine)