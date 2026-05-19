# check_user_table.py
import sqlite3

conn = sqlite3.connect("Sellams_namm.db")
cursor = conn.cursor()

# Lister toutes les tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables disponibles:")
for table in tables:
    print(f"  - {table[0]}")

# Vérifier spécifiquement la table utilisateur
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='utilisateur'")
if cursor.fetchone():
    print("\n Table 'utilisateur' trouvée")
    
    # Obtenir la structure de la table
    cursor.execute("PRAGMA table_info(utilisateur)")
    columns = cursor.fetchall()
    print("\n Structure de la table utilisateur:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Afficher quelques lignes
    cursor.execute("SELECT * FROM utilisateur LIMIT 5")
    rows = cursor.fetchall()
    print(f"\n Aperçu des données ({len(rows)} lignes):")
    for row in rows:
        print(f"  {row}")
else:
    print("\n Table 'utilisateur' non trouvée")

conn.close()