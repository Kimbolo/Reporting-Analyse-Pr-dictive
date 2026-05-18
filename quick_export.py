# quick_export.py - Export simple vers SQLite
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import os

print("Export rapide vers SQLite...")

# Connexion MySQL locale
try:
    mysql_engine = create_engine('mysql+pymysql://root@localhost:3306/Sellams_namm')
    print("Connexion MySQL OK")
except Exception as e:
    print(f"Erreur MySQL : {e}")
    exit(1)

# Supprimer l'ancien fichier SQLite
if os.path.exists("Sellams_namm.db"):
    os.remove("Sellams_namm.db")
    print("Ancien fichier supprimé")

# Créer le fichier SQLite
sqlite_conn = sqlite3.connect("Sellams_namm.db")
print("Fichier SQLite créé")

# Tables à exporter
tables = ['facture_client', 'paiement_facture', 'stock', 'produit', 'conditionnement_production', 'personne']

for table in tables:
    try:
        print(f"Export de {table}...")
        df = pd.read_sql(f"SELECT * FROM {table}", mysql_engine)
        df.to_sql(table, sqlite_conn, if_exists='replace', index=False)
        print(f"{len(df)} lignes exportées")
    except Exception as e:
        print(f"{table} : {e}")

sqlite_conn.close()

# Vérification
size = os.path.getsize("Sellams_namm.db") / (1024 * 1024)
print(f"\n Fichier SQLite créé : Sellams_namm.db ({size:.2f} MB)")

# Poussez ce fichier sur GitHub
print("\n Maintenant, exécutez :")
print("   git add Sellams_namm.db")
print("   git commit -m 'Ajout base SQLite'")
print("   git push")