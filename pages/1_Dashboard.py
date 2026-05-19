import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db import get_data, table_exists, get_data_safe
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sns.set_style("whitegrid")

# ===============================
# CHARGEMENT DES DONNÉES BDD
# ===============================

# Chargement sécurisé de conditionnement_production
if table_exists("conditionnement_production"):
    df_conditionnement = get_data_safe("SELECT * FROM conditionnement_production")
    if not df_conditionnement.empty:
        # Convertir les colonnes en majuscules pour uniformité
        df_conditionnement.columns = [col.upper() for col in df_conditionnement.columns]
else:
    df_conditionnement = pd.DataFrame()

df_produit = get_data("""
    SELECT
        ID_PRODUIT,
        DESIGNATION
    FROM produit
""")

# ===============================
# MERGE PRODUIT ↔ CONDITIONNEMENT
# ===============================
if not df_conditionnement.empty and not df_produit.empty:
    # Vérifier que les colonnes existent
    if 'ID_ARTICLE' in df_conditionnement.columns:
        df_prod = df_conditionnement.merge(
            df_produit,
            left_on="ID_ARTICLE",
            right_on="ID_PRODUIT",
            how="left"
        )

        df_prod["DESIGNATION"] = df_prod["DESIGNATION"].fillna("Article inconnu")

        # NORMALISATION DATE
        if 'DATE_PRODUCTION' in df_prod.columns:
            df_prod["DATE_PRODUCTION"] = pd.to_datetime(
                df_prod["DATE_PRODUCTION"],
                errors="coerce"
            )

        # MÉTRIQUES MÉTIER
        colonnes_pertes = ["PERDE_EN_BOUTEILLE", "PERDE_EN_CAPSULE", 
                          "PERDE_EN_ETIQUETTE", "PERDE_EN_CARTON", "QUANTITE_AVARIE"]
        
        colonnes_presentes = [col for col in colonnes_pertes if col in df_prod.columns]
        
        if colonnes_presentes:
            df_prod["PERTES_TOTALES"] = df_prod[colonnes_presentes].fillna(0).sum(axis=1)
        else:
            df_prod["PERTES_TOTALES"] = 0

        if "QUANTITE_REELLE" in df_prod.columns and "QUANTITE_ATTENDUE" in df_prod.columns:
            df_prod["ECART_STOCK"] = (
                df_prod["QUANTITE_REELLE"].fillna(0) -
                df_prod["QUANTITE_ATTENDUE"].fillna(0)
            )
        else:
            df_prod["ECART_STOCK"] = 0
    else:
        st.warning("La colonne ID_ARTICLE est manquante dans conditionnement_production")
        df_prod = pd.DataFrame()
else:
    if df_conditionnement.empty:
        st.info("Données de conditionnement non disponibles")
    df_prod = pd.DataFrame()

MOIS_FR = {
    1: "Janvier",
    2: "Février",
    3: "Mars",
    4: "Avril",
    5: "Mai",
    6: "Juin",
    7: "Juillet",
    8: "Août",
    9: "Septembre",
    10: "Octobre",
    11: "Novembre",
    12: "Décembre"
}

def format_cfa(valeur):
    if valeur is None:
        return "0 FCFA"
    return f"{valeur:,.0f}".replace(",", " ") + " FCFA"

PLOTLY_CONFIG = {
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
    'displaylogo': False,
    'responsive': True,
}

COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb78',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf'
}

st.title("Dashboard interactif")

# ===============================
# SÉCURITÉ SESSION
# ===============================
required = ["facture_client", "paiement_facture", "stock"]
if not all(k in st.session_state for k in required):
    st.warning("Veuillez d’abord configurer les filtres dans l’onglet App.")
    st.stop()

facture_all = st.session_state["facture_client"].copy()
paiement_all = st.session_state["paiement_facture"].copy()
stock = st.session_state["stock"].copy()

# ===============================
# FILTRES DASHBOARD
# ===============================
st.sidebar.subheader("Filtres Dashboard")

annees_dispo = sorted(facture_all["DATE_CREATION"].dt.year.dropna().unique())
annee_dash = st.sidebar.selectbox("Année", annees_dispo, index=0)

mois_dispo = sorted(
    facture_all[facture_all["DATE_CREATION"].dt.year == annee_dash]
    ["DATE_CREATION"].dt.month.unique()
)

mois = st.sidebar.multiselect(
    "Mois",
    options=mois_dispo,
    default=mois_dispo,
    format_func=lambda x: MOIS_FR.get(x, str(x))
)

# Filtrage de df_prod si disponible
if not df_prod.empty and 'DATE_PRODUCTION' in df_prod.columns:
    df_prod_f = df_prod[
        (df_prod["DATE_PRODUCTION"].dt.year == annee_dash) &
        (df_prod["DATE_PRODUCTION"].dt.month.isin(mois))
    ].copy()
else:
    df_prod_f = pd.DataFrame()

# ===============================
# TYPE D’ANALYSE
# ===============================
analyse_type = st.sidebar.radio(
    "Type d’analyse",
    ["Ventes", "Encaissements", "Stock", "Production", "Pertes"],
    horizontal=True
)

# ===============================
# APPLICATION DES FILTRES
# ===============================
facture = facture_all[
    (facture_all["DATE_CREATION"].dt.year == annee_dash) &
    (facture_all["DATE_CREATION"].dt.month.isin(mois))
].copy()

paiement = paiement_all[
    paiement_all["DATE_PAIEMENT"].dt.year == annee_dash
].copy()

# ===============================
# KPI DYNAMIQUES
# ===============================
ca = facture["MONTANT_NET"].sum()
nb_factures = facture["ID_FACTURE_CLIENT"].nunique()
panier = ca / nb_factures if nb_factures else 0

col1, col2, col3 = st.columns(3)
col1.metric("CA", format_cfa(ca))
col2.metric("Factures", nb_factures)
col3.metric("Panier moyen", format_cfa(panier))

# ===============================
# PARAMÈTRES DE TRI
# ===============================
st.markdown("### Paramètres d’affichage")

col_a, col_b = st.columns(2)

with col_a:
    top_n = st.slider(
        "Nombre d’éléments à afficher",
        min_value=3,
        max_value=10,
        value=5
    )

with col_b:
    mode_valeur = st.radio(
        "Mesure",
        ["Chiffre d’affaires", "Volume"],
        horizontal=True
    )

# ===============================
# ANALYSE PAR TYPE
# ===============================
if analyse_type == "Ventes":
    st.markdown("## Analyse des ventes")
    st.caption("Évolution du chiffre d’affaires sur la période sélectionnée")

    facture["MOIS_NUM"] = facture["DATE_CREATION"].dt.month
    facture["MOIS_NOM"] = facture["MOIS_NUM"].map(MOIS_FR)

    ca_mensuel = (
        facture
        .groupby("MOIS_NOM", sort=False)["MONTANT_NET"]
        .sum()
    )

    if not ca_mensuel.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ca_mensuel.index.tolist(),
            y=ca_mensuel.values.tolist(),
            mode="lines+markers",
            name='CA mensuel',
            line=dict(color=COLORS['primary'],width=3),
            marker=dict(size=8, symbol='circle', color="#1f77b4"),
            hovertemplate='<b>%{x}</b><br>CA: %{y:,.0f} FCFA<extra></extra>'
        ))
        fig.update_layout(
            title=dict(text="Évolution mensuelle du chiffre d’affaires", x=0.5, font=dict(size=16)),
            xaxis_title="Mois",
            yaxis_title="Chiffre d’affaires (FCFA)",
            hovermode="x unified",
            template="plotly_white",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(tickangle=45),
            yaxis=dict(tickformat=",.0f", tickprefix='', ticksuffix=' FCFA')
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.info("Cette courbe met en évidence les périodes de croissance ou de ralentissement de l’activité commerciale.")
    else:
        st.info("Aucune donnée de ventes disponible pour la période sélectionnée")

    # ... (le reste du code pour Ventes reste similaire, juste avec les vérifications) ...

elif analyse_type == "Encaissements":
    st.markdown("## CA vs Encaissements")

    facture["MOIS_NUM"] = facture["DATE_CREATION"].dt.month
    facture["MOIS_NOM"] = facture["MOIS_NUM"].map(MOIS_FR)

    paiement["MOIS_NUM"] = paiement["DATE_PAIEMENT"].dt.month
    paiement["MOIS_NOM"] = paiement["MOIS_NUM"].map(MOIS_FR)

    ca_mensuel = facture.groupby("MOIS_NOM", sort=False)["MONTANT_NET"].sum()
    enc_mensuel = paiement.groupby("MOIS_NOM", sort=False)["MONTANT"].sum()

    df_compare = pd.DataFrame({
        "CA": ca_mensuel,
        "Encaissements": enc_mensuel
    }).fillna(0)

    if not df_compare.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_compare.index.tolist(),
            y=df_compare["CA"].tolist(),
            mode='lines+markers',
            name='CA',
            line=dict(color=COLORS['primary'],width=3),
            marker=dict(size=8, symbol='circle', color="#1f77b4"),
            hovertemplate='<b>%{x}</b><br>CA: %{y:,.0f} FCFA<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=df_compare.index.tolist(),
            y=df_compare["Encaissements"].tolist(),
            mode='lines+markers',
            name='Encaissements',
            line=dict(color=COLORS['success'], width=3, dash='dash'),
            marker=dict(size=8, symbol='square', color="#ff7f0e"),
            hovertemplate='<b>%{x}</b><br>Encaissements: %{y:,.0f} FCFA<extra></extra>'
        ))
        fig.update_layout(
            title=dict(text="CA vs Encaissements", x=0.5, font=dict(size=16)),
            xaxis_title="Mois",
            yaxis_title="Montant (FCFA)",
            hovermode="x unified",
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            template="plotly_white",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(tickangle=45),
            yaxis=dict(tickformat=",.0f", tickprefix='', ticksuffix=' FCFA')
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        st.info("Aucune donnée disponible")

elif analyse_type == "Stock":
    st.subheader("Stock par produit")
    st.info("Module Stock en cours de développement.")

elif analyse_type == "Production":
    st.subheader("Production")
    st.info("Module Production en cours de développement.")

elif analyse_type == "Pertes":
    st.subheader("Pertes - Top articles par volume de pertes")
    st.info("Module Pertes en cours de développement.")

# ===============================
# INTERPRÉTATION AUTOMATIQUE
# ===============================
if ca > 0:
    msg = f"Pour l’année {annee_dash}, sur la période sélectionnée, le chiffre d’affaires est de {format_cfa(ca)} avec un panier moyen de {format_cfa(panier)}."
    st.info(msg)
else:
    st.info("Aucune donnée disponible avec ces filtres.")

st.success("Dashboard mis à jour dynamiquement en fonction des filtres sélectionnés.")