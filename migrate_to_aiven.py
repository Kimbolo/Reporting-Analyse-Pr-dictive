# migrate_to_aiven.py - Exécutez ceci sur votre machine locale
import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

print("Migration des données vers Aiven...")

# Connexion à MySQL local
mysql_local = create_engine('mysql+pymysql://root@localhost:3306/Sellams_namm')

# Connexion à Aiven (REMPLACEZ PAR VOS IDENTIFIANTS AIVEN)
aiven_host = "pg-9cda73c-inseec-ce6c.e.aivencloud.com"
aiven_port = 17868
aiven_user = "avnadmin"
aiven_password = "AVNS_TCaa-56FGwcuh363s8e"

# Encoder le mot de passe
password_encoded = urllib.parse.quote_plus(aiven_password)
aiven_url = f"mysql+pymysql://{aiven_user}:{password_encoded}@{aiven_host}:{aiven_port}/defaultdb"
mysql_remote = create_engine(aiven_url)

# Liste des tables à migrer
tables = ['facture_client', 'paiement_facture', 'stock', 'produit', 'conditionnement_production', 'personne']

for table in tables:
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", mysql_local)
        df.to_sql(table, mysql_remote, if_exists='replace', index=False)
        print(f"Table {table} migrée : {len(df)} lignes")
    except Exception as e:
        print(f"Table {table} non trouvée : {e}")

print("Migration terminée !")