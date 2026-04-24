import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===============================
# STYLE GLOBAL
# ===============================
sns.set_style("whitegrid")

st.set_page_config(
    page_title="Application Data – Sellams",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CONSTANTES
# ===============================
MOIS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}

def format_cfa(valeur):
    if valeur is None:
        return "0 FCFA"
    return f"{valeur:,.0f}".replace(",", " ") + " FCFA"

# ===============================
# CHARGEMENT DONNÉES
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "reporting sellam.xlsx")

@st.cache_data
def load_data():
    facture = pd.read_excel(FILE_PATH, sheet_name="facture_client", engine="openpyxl")
    paiement = pd.read_excel(FILE_PATH, sheet_name="PAIEMENT_FACTURE", engine="openpyxl")
    stock = pd.read_excel(FILE_PATH, sheet_name="STOCK", engine="openpyxl")

    facture["DATE_CREATION"] = pd.to_datetime(facture["DATE_CREATION"], errors="coerce")
    paiement["DATE_PAIEMENT"] = pd.to_datetime(paiement["DATE_PAIEMENT"], errors="coerce")

    return facture, paiement, stock

facture, paiement, stock = load_data()

# ===============================
# SIDEBAR — FILTRES & PÉRIMÈTRE
# ===============================
st.sidebar.markdown("## Périmètre d’analyse")

annees_dispo = sorted(facture["DATE_CREATION"].dt.year.dropna().unique())
annee = st.sidebar.selectbox("Année analysée", annees_dispo, index=0)

mois_dispo = sorted(
    facture[facture["DATE_CREATION"].dt.year == annee]["DATE_CREATION"].dt.month.unique()
)

mois_selectionnes = st.sidebar.multiselect(
    "Mois",
    options=mois_dispo,
    default=mois_dispo,
    format_func=lambda x: MOIS_FR.get(x, str(x))
)

st.sidebar.divider()

st.sidebar.markdown("### Activités analysées")
perimetre = st.sidebar.multiselect(
    "Périmètre métier",
    [
        "Factures",
        "Recouvrements",
        "Commandes",
        "Livraisons",
        "Clients",
        "Fournisseurs"
    ],
    default=["Factures", "Commandes", "Clients"]
)

# ===============================
# APPLICATION FILTRES
# ===============================
facture_f = facture[
    (facture["DATE_CREATION"].dt.year == annee) &
    (facture["DATE_CREATION"].dt.month.isin(mois_selectionnes)) &
    (facture["VALIDER"] == 1)
].copy()

paiement_f = paiement[paiement["DATE_PAIEMENT"].dt.year == annee].copy()

metier = st.selectbox(
    "Type d’analyse métier",
    ["Ventes", "Encaissements", "Stock", "Production", "Pertes"]
)

st.session_state["metier"] = metier

# ===============================
# PARTAGE SESSION
# ===============================
st.session_state.update({
    "annee": annee,
    "mois": mois_selectionnes,
    "perimetre": perimetre,
    "facture_f": facture_f,
    "paiement_f": paiement_f,
    "stock": stock,
    "metier": metier
})

# ===============================
# TITRE & CONTEXTE
# ===============================
st.title("Vue globale & Pilotage")
st.caption(f"Analyse consolidée – Année {annee}")

st.divider()

# ===============================
# KPI EXÉCUTIFS
# ===============================
ca_total = facture_f["MONTANT_NET"].sum()
nb_factures = facture_f["ID_FACTURE_CLIENT"].nunique()
encaisse = paiement_f["MONTANT"].sum()
taux_enc = (encaisse / ca_total * 100) if ca_total > 0 else 0
stock_total = stock["QUANTITE"].sum()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Chiffre d’affaires", format_cfa(ca_total))
col2.metric("Factures", nb_factures)
col3.metric("Encaissements", format_cfa(encaisse))
col4.metric("Taux d’encaissement", f"{taux_enc:.1f} %")
col5.metric("Stock global", f"{stock_total:,.0f}")

st.divider()

# ===============================
# CYCLE MÉTIER (LECTURE FLUX)
# ===============================
st.subheader("Lecture du cycle métier")

st.markdown(
    f"""
    **Commandes → Livraisons → Facturation → Encaissements**

    - **Périmètre actif** : {", ".join(perimetre)}
    - **Volume facturé** : {format_cfa(ca_total)}
    - **Trésorerie encaissée** : {format_cfa(encaisse)}
    """
)

st.divider()

# ===============================
# TENDANCES GLOBALES (MAX 2)
# ===============================
st.subheader("Tendances globales")

facture_f["MOIS_NUM"] = facture_f["DATE_CREATION"].dt.month
facture_f["MOIS_NOM"] = facture_f["MOIS_NUM"].map(MOIS_FR)

ca_mensuel = facture_f.groupby("MOIS_NOM", sort=False)["MONTANT_NET"].sum()

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ca_mensuel.index, ca_mensuel.values, marker="o", linewidth=3)
ax.set_ylabel("CA (FCFA)")
ax.set_xlabel("Mois")
ax.grid(axis="y", linestyle="--", alpha=0.6)
plt.xticks(rotation=45)
st.pyplot(fig)

st.divider()

# ===============================
# ALERTES & RISQUES
# ===============================
st.subheader("Alertes & points d’attention")

if taux_enc < 80:
    st.warning("Taux d’encaissement inférieur au seuil recommandé (80 %).")
else:
    st.success("Taux d’encaissement satisfaisant.")

if stock_total <= 0:
    st.warning("Stock global nul ou non renseigné.")

# ===============================
# LECTURE EXÉCUTIVE
# ===============================
st.subheader("Interprétation des résultats")

resume = f"""
Sur la période analysée ({annee}), l’activité montre un chiffre d’affaires de 
**{format_cfa(ca_total)}** avec un taux d’encaissement de **{taux_enc:.1f} %**.
Le périmètre étudié couvre principalement **{", ".join(perimetre)}**.
"""

st.info(resume)

st.divider()

# ===============================
# ORIENTATION UTILISATEUR
# ===============================
st.subheader("Explorer les analyses")

st.markdown("""
- **Dashboard** : Suivi opérationnel détaillé  
- **Analytics** : Compréhension des causes et écarts    
- **ML** : Prévisions et scénarios futurs  
""")