import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db import get_data
from sqlalchemy import create_engine, text

import plotly.express as px
import plotly.graph_objects as go

sns.set_style("whitegrid")

# ===============================
# STYLE GLOBAL
# ===============================

st.set_page_config(
    page_title="Application Data – Sellams",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CONFIGURATION UNIVERSELLE DES GRAPHIQUES PLOTLY
# ==========================================================

def create_standard_figure(title="", height=450):
    """
    Crée une figure Plotly avec une configuration standardisée.
    Tous les graphiques utiliseront cette même base pour être homogènes.
    """
    fig = go.Figure()
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        template='plotly_white',
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig

@st.cache_data(ttl=300)
def load_all_data():
    """Charge toutes les données nécessaires depuis la BDD"""
    
    # Factures
    facture = get_data("""
        SELECT 
            ID_FACTURE_CLIENT,
            DATE_CREATION,
            MONTANT_NET,
            VALIDER,
            STATUT,
            ID_PERSONNE
        FROM facture_client
    """)
    
    # Paiements
    paiement = get_data("""
        SELECT 
            ID_PAIEMENT_FACTURE,
            DATE_PAIEMENT,
            MONTANT,
            STATUT,
            ID_FACTURE_CLIENT
        FROM paiement_facture
    """)
    
    # Stock
    stock = get_data("""
        SELECT 
            DESIGNATION,
            ID_STOCK,
            ID_MAGASIN,
            QUANTITE,
            QUANTITE_ENTREE,
            DATE_ENTREE AS DATE_PRODUCTION
        FROM stock
    """)
    
    # Produits
    df_produit = get_data("""
        SELECT
            ID_PRODUIT,
            DESIGNATION
        FROM produit
    """)
    
    # Conditionnement production
    df_conditionnement = get_data("""
        SELECT
            DATE_PRODUCTION,
            ID_ARTICLE,
            PERDE_EN_BOUTEILLE,
            PERDE_EN_CAPSULE,
            PERDE_EN_ETIQUETTE,
            PERDE_EN_CARTON,
            QUANTITE_AVARIE,
            QUANTITE_ATTENDUE,
            QUANTITE_REELLE,
            QUANTITE_TOTALE
        FROM conditionnement_production
    """)
    
    # Clients (personnes)
    df_personne = get_data("""
        SELECT 
            ID_PERSONNE, 
            NOM 
        FROM personne
    """)

  # Conversion des dates
    if not facture.empty :
        facture["DATE_CREATION"] = pd.to_datetime(facture["DATE_CREATION"], errors="coerce")
    if not paiement.empty:
        paiement["DATE_PAIEMENT"] = pd.to_datetime(paiement["DATE_PAIEMENT"], errors="coerce")
    if not stock.empty:
        stock["DATE_PRODUCTION"] = pd.to_datetime(stock["DATE_PRODUCTION"], errors="coerce")
    if not df_conditionnement.empty:
        df_conditionnement["DATE_PRODUCTION"] = pd.to_datetime(df_conditionnement["DATE_PRODUCTION"], errors="coerce")

    # ← LE RETURN DOIT ÊTRE ICI, EN DEHORS DE TOUT IF
    return facture, paiement, stock, df_produit, df_conditionnement, df_personne
# Chargement unique des données
with st.spinner("Chargement des données..."):
    facture_all, paiement_all, stock_all, df_produit, df_conditionnement, df_personne = load_all_data()

    st.session_state["facture_all"] = facture_all.copy()
    st.session_state["paiement_all"] = paiement_all.copy()
    st.session_state["stock_all"] = stock_all.copy()

# Correction : Renommer les variables pour cohérenceif not facture.empty :
facture = facture_all
paiement = paiement_all
stock = stock_all

if facture_all.empty:
    st.error("Aucune donnée trouvée dans la table 'facture_client'.")
    st.info("""
    **Vérifiez dans phpMyAdmin :**
    1. La base de données contient-elle des données ?
    2. La table 'facture_client' existe-t-elle ?
    3. Les colonnes sont-elles correctes ?
    
    **Structure attendue pour 'facture_client' :**
    - ID_FACTURE_CLIENT
    - DATE_CREATION
    - MONTANT_NET
    - VALIDER
    - STATUT
    - ID_PERSONNE
    """)
    st.stop()

# ===============================
# MERGE PRODUIT ↔ CONDITIONNEMENT
# ===============================

if not df_conditionnement.empty and not df_produit.empty:
    df_prod = df_conditionnement.merge(
    df_produit,
    left_on="ID_ARTICLE",
    right_on="ID_PRODUIT",
    how="left"
)

    df_prod["DESIGNATION"] = df_prod["DESIGNATION"].fillna("Article inconnu")

# ===============================
# NORMALISATION DATE
# ===============================
    df_prod["DATE_PRODUCTION"] = pd.to_datetime(
        df_prod["DATE_PRODUCTION"],
        errors="coerce"
    )

# ===============================
# MÉTRIQUES MÉTIER
# ===============================
    df_prod["PERTES_TOTALES"] = (
        df_prod["PERDE_EN_BOUTEILLE"].fillna(0) +
        df_prod["PERDE_EN_CAPSULE"].fillna(0) +
        df_prod["PERDE_EN_ETIQUETTE"].fillna(0) +
        df_prod["PERDE_EN_CARTON"].fillna(0) +
        df_prod["QUANTITE_AVARIE"].fillna(0)
    )

    df_prod["ECART_STOCK"] = (
        df_prod["QUANTITE_REELLE"].fillna(0) -
        df_prod["QUANTITE_ATTENDUE"].fillna(0)
    )
else :
    df_prod = pd.DataFrame()

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
# SIDEBAR — FILTRES & PÉRIMÈTRE
# ===============================
st.sidebar.markdown("## Périmètre d’analyse")

# Vérifier que les données ne sont pas vides
if facture.empty or 'DATE_CREATION' not in facture.columns:
    st.error("Aucune donnée disponible. Vérifiez votre base de données.")
    st.stop()

annees_dispo = sorted(facture["DATE_CREATION"].dt.year.dropna().unique())
if len(annees_dispo) == 0:
    st.error("Aucune date valide trouvée dans les données.")
    st.stop()

annee = st.sidebar.selectbox("Année analysée", annees_dispo, index=0)

mois_dispo = sorted(
    facture[facture["DATE_CREATION"].dt.year == annee]["DATE_CREATION"].dt.month.unique()
)

if len(mois_dispo) == 0:
    st.warning(f"Aucune donnée pour l'année {annee}")
    mois_selectionnes = []
else:
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
if mois_selectionnes:
    facture_client = facture[
        (facture["DATE_CREATION"].dt.year == annee) &
        (facture["DATE_CREATION"].dt.month.isin(mois_selectionnes)) &
        (facture["VALIDER"] == 1)
    ].copy()
else:
    facture_client = pd.DataFrame()

if not paiement.empty and 'DATE_PAIEMENT' in paiement.columns:
    paiement_facture = paiement[paiement["DATE_PAIEMENT"].dt.year == annee].copy()
else:
    paiement_facture = pd.DataFrame()

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
    "facture_client": facture_client,
    "paiement_facture": paiement_facture,
    "stock": stock,
    "df_prod": df_prod,
    "df_personne": df_personne,
    "metier": metier
})

# ===============================
# TITRE & CONTEXTE
# ===============================
st.title("Vue globale & Pilotage")
st.caption(f"Analyse consolidée – Année {annee}")
st.caption("Source : Base de données MySQL – sellams_namm")

st.divider()

# ===============================
# KPI EXÉCUTIFS
# ===============================
ca_total = facture_client["MONTANT_NET"].sum() if not facture_client.empty else 0
nb_factures = facture_client["ID_FACTURE_CLIENT"].nunique() if not facture_client.empty else 0
encaisse = paiement_facture["MONTANT"].sum() if not paiement_facture.empty else 0
taux_enc = (encaisse / ca_total * 100) if ca_total > 0 else 0
stock_total = stock["QUANTITE"].sum() if not stock.empty and 'QUANTITE' in stock.columns else 0

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
# TENDANCES GLOBALES
# ===============================
st.subheader("Tendances globales")

if not facture_client.empty:
    facture_client["MOIS_NUM"] = facture_client["DATE_CREATION"].dt.month
    facture_client["MOIS_NOM"] = facture_client["MOIS_NUM"].map(MOIS_FR)
    
    ca_mensuel = facture_client.groupby("MOIS_NOM", sort=False)["MONTANT_NET"].sum().reset_index()
    
    if not ca_mensuel.empty:

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ca_mensuel["MOIS_NOM"],
            y=ca_mensuel["MONTANT_NET"],
            mode='lines+markers',
            name='CA MENSUEL',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, symbol='circle', color='#1f77b4'),
            hovertemplate='<b>%{x}</b><br>CA: %{y:,.0f} FCFA<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text="Évolution du chiffre d’affaires par mois", x=0.5),
            xaxis_title="Mois",
            yaxis_title="Chiffre d’affaires (FCFA)",
            hovermode="x unified",
            template="plotly_white",
            height=450,
            margin=dict(l=50, r=50, t=80, b=50),
            yaxis=dict(tickformat=",.0f", tickprefix='', ticksuffix=' FCFA')
        )

        # Configuration du zoom
        config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d','autoScale2d', 'resetScale2d'],
            'displaylogo': False,
            'responsive': True
        }

        st.plotly_chart(fig, use_container_width=True, config=config)
    else:
        st.info("Aucune donnée disponible pour afficher les tendances")
else:
    st.info("Aucune facture trouvée pour la période sélectionnée")

st.divider()

# ===============================
# ALERTES & RISQUES
# ===============================
st.subheader("Alertes & points d’attention")

if taux_enc < 80:
    st.warning("Taux d’encaissement inférieur au seuil recommandé (80 %).")
    st.caption("Action recommandée : Renforcer le recouvrement des créances clients.")
else:
    st.success("Taux d’encaissement satisfaisant.")

if stock_total <= 0:
    st.warning("Stock global nul ou non renseigné.")
    st.caption("Action recommandée : Vérifier l'inventaire physique.")
else:
    st.info(f"Niveau de stock actuel : {stock_total:,.0f} unités")

if ca_total == 0:
    st.error("Aucun chiffre d'affaires enregistré sur la période.")
elif ca_total < 1000000:
    st.warning("Chiffre d'affaires faible sur la période.")

# ===============================
# LECTURE EXÉCUTIVE
# ===============================
st.subheader("Interprétation des résultats")

if ca_total > 0:
    resume = f"""
    Sur la période analysée ({annee}), l’activité montre un chiffre d’affaires de 
    **{format_cfa(ca_total)}** avec un taux d’encaissement de **{taux_enc:.1f} %**.
    Le périmètre étudié couvre principalement **{", ".join(perimetre)}**.
    """
else:
    resume = f"""
    Aucune donnée de chiffre d'affaires n'est disponible pour l'année {annee}.
    Vérifiez que votre fichier Excel contient des données valides dans l'onglet 'facture_client'
    avec des factures validées (VALIDER = 1).
    """

st.info(resume)

st.divider()

# ===============================
# ORIENTATION UTILISATEUR
# ===============================
st.subheader("Explorer les analyses")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Dashboard**")
    st.caption("Suivi opérationnel détaillé des ventes et encaissements")
with col2:
    st.markdown("**Analytics**")
    st.caption("Analyse des causes et écarts de performance")
with col3:
    st.markdown("**ML**")
    st.caption("Prévisions des ventes et scénarios futurs")

# ===============================
# INFORMATIONS DE DÉBOGAGE
# ===============================
with st.expander(" Informations techniques"):
    st.write("**Statut des données :**")
    st.write(f"- Factures chargées : {len(facture)} lignes")
    st.write(f"- Paiements chargés : {len(paiement)} lignes")
    st.write(f"- Stock : {len(stock)} produits")
    st.write(f"- Produits : {len(df_produit)} références")
    st.write(f"- Conditionnement : {len(df_conditionnement)} enregistrements")
    st.write(f"- Clients : {len(df_personne)} personnes")
    st.write(f"- Période filtrée : {len(facture_client)} factures")