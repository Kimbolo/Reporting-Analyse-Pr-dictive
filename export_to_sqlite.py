# export_sqlite.py
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

print("🔄 Export de MySQL vers SQLite...")

# Connexion à votre MySQL local (celui qui fonctionne déjà)
mysql_engine = create_engine('mysql+pymysql://root@localhost:3306/Sellams_namm')

# Connexion SQLite (crée le fichier)
sqlite_conn = sqlite3.connect("Sellams_namm.db")

# Les tables de votre application
tables = ['facture_client', 'paiement_facture', 'stock', 'produit', 
          'conditionnement_production', 'personne', 'magasin']

for table in tables:
    try:
        print(f"📤 {table}...")
        df = pd.read_sql(f"SELECT * FROM {table}", mysql_engine)
        df.to_sql(table, sqlite_conn, if_exists='replace', index=False)
        print(f"   ✅ {len(df)} lignes")
    except Exception as e:
        print(f"   ⚠️ {table}: {e}")

sqlite_conn.close()
print("\n✅ Fichier 'Sellams_namm.db' créé avec succès !")