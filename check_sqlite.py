# check_sqlite.py
import sqlite3
import pandas as pd
import os

db_path = "Sellams_namm.db"

print("=" * 50)
print("🔍 VÉRIFICATION DE LA BASE SQLITE")
print("=" * 50)

# 1. Vérifier l'existence du fichier
if not os.path.exists(db_path):
    print(f"❌ Fichier '{db_path}' introuvable !")
    exit(1)

# 2. Taille du fichier
size = os.path.getsize(db_path)
print(f"📁 Fichier: {db_path}")
print(f"📏 Taille: {size / 1024:.2f} KB ({size / (1024*1024):.2f} MB)")

# 3. Connexion et inspection
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 4. Lister toutes les tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(f"\n📊 Tables trouvées ({len(tables)}):")
for table in tables:
    table_name = table[0]
    
    # Compter les lignes
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    
    # Afficher les colonnes
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    print(f"\n  📋 Table '{table_name}'")
    print(f"     Lignes: {count}")
    print(f"     Colonnes: {len(columns)}")
    print(f"     Structure: {', '.join([col[1] for col in columns[:5]])}")
    if len(columns) > 5:
        print(f"               ... et {len(columns) - 5} autres")

# 5. Aperçu des données pour chaque table
print("\n" + "=" * 50)
print("📊 APERÇU DES DONNÉES")
print("=" * 50)

for table in tables:
    table_name = table[0]
    df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn)
    print(f"\n📋 Table '{table_name}':")
    print(df.to_string())
    print("-" * 30)

conn.close()
print("\n✅ Vérification terminée !")