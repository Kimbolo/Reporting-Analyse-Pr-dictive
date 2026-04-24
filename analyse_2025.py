import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ===============================
# CHARGEMENT DU FICHIER
# ===============================
FILE_PATH = "reporting sellam.xlsx"
xls = pd.ExcelFile(FILE_PATH)

# ===============================
# 1. VENTES – FACTURE CLIENT
# ===============================
facture = pd.read_excel(xls, sheet_name="facture_client")
facture["DATE_CREATION"] = pd.to_datetime(facture["DATE_CREATION"], errors="coerce")

facture_2025 = facture[facture["DATE_CREATION"].dt.year == 2025]

# KPI
ca_total = facture_2025["MONTANT_NET"].sum()
nb_factures = facture_2025["ID_FACTURE_CLIENT"].nunique()
panier_moyen = ca_total / nb_factures if nb_factures > 0 else 0

print("=== KPI VENTES 2025 ===")
print(f"CA total : {ca_total:,.0f}")
print(f"Nombre de factures : {nb_factures}")
print(f"Panier moyen : {panier_moyen:,.0f}")

# --- Diagramme en barres : CA mensuel
facture_2025["MOIS"] = facture_2025["DATE_CREATION"].dt.month
ca_mensuel = facture_2025.groupby("MOIS")["MONTANT_NET"].sum()

plt.figure(figsize=(10, 5))
ca_mensuel.plot(kind="bar", color="steelblue")
plt.title("Chiffre d’affaires mensuel – 2025")
plt.xlabel("Mois")
plt.ylabel("CA")
plt.tight_layout()
plt.show()

# ===============================
# 2. PAIEMENTS – ENCAISSEMENTS
# ===============================
paiement = pd.read_excel(xls, sheet_name="PAIEMENT_FACTURE")
paiement["DATE_PAIEMENT"] = pd.to_datetime(paiement["DATE_PAIEMENT"], errors="coerce")

paiement_2025 = paiement[paiement["DATE_PAIEMENT"].dt.year == 2025]

encaissement_mensuel = (
    paiement_2025
    .groupby(paiement_2025["DATE_PAIEMENT"].dt.month)["MONTANT"]
    .sum()
)

# --- Courbe : encaissements
plt.figure(figsize=(10, 5))
encaissement_mensuel.plot(marker="o", color="green")
plt.title("Encaissements mensuels – 2025")
plt.xlabel("Mois")
plt.ylabel("Montant encaissé")
plt.tight_layout()
plt.show()

# ===============================
# 3. PRODUCTION
# ===============================
production = pd.read_excel(xls, sheet_name="PRODUCTION")
production["DATE_PRODUCTION"] = pd.to_datetime(production["DATE_PRODUCTION"], errors="coerce")

production_2025 = production[production["DATE_PRODUCTION"].dt.year == 2025]

prod_mensuelle = (
    production_2025
    .groupby(production_2025["DATE_PRODUCTION"].dt.month)
    .size()
)

# --- Histogramme : volume de production
plt.figure(figsize=(10, 5))
prod_mensuelle.plot(kind="bar", color="orange")
plt.title("Nombre de productions par mois – 2025")
plt.xlabel("Mois")
plt.ylabel("Nombre de lots")
plt.tight_layout()
plt.show()

# ===============================
# 4. STOCK – RÉPARTITION
# ===============================
stock = pd.read_excel(xls, sheet_name="STOCK")
stock["DATE_ENTREE"] = pd.to_datetime(stock["DATE_ENTREE"], errors="coerce")

stock_2025 = stock[stock["DATE_ENTREE"].dt.year == 2025]

stock_magasin = stock_2025.groupby("ID_MAGASIN")["QUANTITE"].sum()

# --- Camembert : répartition du stock
plt.figure(figsize=(8, 8))
stock_magasin.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Répartition du stock par magasin – 2025")
plt.ylabel("")
plt.tight_layout()
plt.show()

# ===============================
# 5. LIVRAISONS
# ===============================
livraison = pd.read_excel(xls, sheet_name="livraison_commande")
livraison["DATE_LIVRAISON"] = pd.to_datetime(livraison["DATE_LIVRAISON"], errors="coerce")

livraison_2025 = livraison[livraison["DATE_LIVRAISON"].dt.year == 2025]

livraison_etat = livraison_2025["ETAT_LIVRAISON"].value_counts()

# --- Diagramme en barres : état des livraisons
plt.figure(figsize=(8, 5))
livraison_etat.plot(kind="bar", color="purple")
plt.title("Statut des livraisons – 2025")
plt.xlabel("Statut")
plt.ylabel("Nombre")
plt.tight_layout()
plt.show()