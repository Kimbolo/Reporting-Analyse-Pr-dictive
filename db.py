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

# db.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse

def get_db_config():
    """
    Récupère la configuration de la base de données.
    Priorité : secrets Streamlit (déploiement) > configuration locale
    """
    # Si on est sur Streamlit Cloud (les secrets existent)
    try:
        if "database" in st.secrets:
            # Nouvelle structure : secrets.toml avec section [database]
            config = {
                'host': st.secrets["database"]["host"],
                'user': st.secrets["database"]["user"],
                'password': st.secrets["database"]["password"],
                'database': st.secrets["database"]["database"],
                'port': int(st.secrets["database"].get("port", 3306))
            }
            return config
        elif "DB_HOST" in st.secrets:
            # Structure alternative : variables individuelles
            config = {
                'host': st.secrets["DB_HOST"],
                'user': st.secrets["DB_USER"],
                'password': st.secrets["DB_PASSWORD"],
                'database': st.secrets["DB_NAME"],
                'port': int(st.secrets.get("DB_PORT", 3306))
            }
            return config
        else:
            # Fallback : configuration locale
            return {
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'database': 'Sellams_namm',
                'port': 3306
            }
    except:
        # Configuration locale par défaut
        return {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'Sellams_namm',
            'port': 3306
        }

@st.cache_resource
def get_db_engine():
    try:
        config = get_db_config()
        
        # Encoder le mot de passe pour éviter les caractères spéciaux
        password_encoded = urllib.parse.quote_plus(config['password'])
        
        # Construction de la chaîne de connexion
        if config['password']:
            connection_string = f"mysql+pymysql://{config['user']}:{password_encoded}@{config['host']}:{config['port']}/{config['database']}"
        else:
            connection_string = f"mysql+pymysql://{config['user']}@{config['host']}:{config['port']}/{config['database']}"
        
        engine = create_engine(
            connection_string, 
            pool_pre_ping=True,
            pool_recycle=3600  # Recycle les connexions après 1 heure
        )
        
        # Tester la connexion
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        return engine
    except Exception as e:
        st.error(f"❌ Erreur de connexion MySQL : {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache les résultats pendant 1 heure
def get_data(query: str) -> pd.DataFrame:
    """
    Exécute une requête SQL et retourne un DataFrame.
    Le résultat est mis en cache pour optimiser les performances.
    """
    engine = get_db_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"❌ Erreur SQL : {str(e)}")
        return pd.DataFrame()

def test_connection():
    """Fonction utilitaire pour tester la connexion"""
    engine = get_db_engine()
    if engine:
        st.success("✅ Connexion à la base de données réussie !")
        return True
    else:
        st.error("❌ Échec de la connexion à la base de données")
        return False