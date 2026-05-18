# export_to_sqlite.py
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

print("🔄 Export de la base MySQL vers SQLite...")

# Connexion à MySQL local
mysql_engine = create_engine('mysql+pymysql://root@localhost:3306/Sellams_namm')

# Création du fichier SQLite
sqlite_conn = sqlite3.connect('Sellams_namm.db')

# Liste des tables à exporter (ajustez selon vos tables)
tables = ['facture_client', 'paiement_facture', 'stock', 'personne', 'magasin', 'produit']

for table in tables:
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", mysql_engine)
        df.to_sql(table, sqlite_conn, if_exists='replace', index=False)
        print(f"✅ Table {table} exportée : {len(df)} lignes")
    except Exception as e:
        print(f"⚠️ Table {table} non trouvée : {e}")

sqlite_conn.close()
print("✅ Export terminé ! Fichier 'Sellams_namm.db' créé")