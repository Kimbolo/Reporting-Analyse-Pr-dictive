import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db import get_data, table_exists, get_data_safe
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sns.set_style("whitegrid")

# ============================
# CHARGEMENT DES DONNÉES BDD
# ============================

# Chargement des données de facturation
@st.cache_data(ttl=300, show_spinner="Chargement des factures...")
def load_facture_data():
    """Charge les données de facturation depuis la BDD"""
    try:
        # Vérifier si la table existe
        if not table_exists("facture_client"):
            st.warning("La table 'facture_client' n'existe pas dans la base de données")
            return pd.DataFrame()
        
        df = get_data_safe("""
            SELECT 
                ID_FACTURE_CLIENT,
                DATE_CREATION,
                MONTANT_NET,
                ID_CLIENT
            FROM facture_client
            WHERE DATE_CREATION IS NOT NULL
        """)
        
        if df is not None and not df.empty:
            df["DATE_CREATION"] = pd.to_datetime(df["DATE_CREATION"], errors='coerce')
            df = df.dropna(subset=['DATE_CREATION'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur chargement factures: {e}")
        return pd.DataFrame()

# Chargement des données de paiement
@st.cache_data(ttl=300, show_spinner="Chargement des paiements...")
def load_paiement_data():
    """Charge les données de paiement depuis la BDD"""
    try:
        # Vérifier si la table existe
        if not table_exists("paiement_facture"):
            st.info("La table 'paiement_facture' n'existe pas - certaines analyses seront limitées")
            return pd.DataFrame()
        
        df = get_data_safe("""
            SELECT 
                ID_PAIEMENT,
                DATE_PAIEMENT,
                MONTANT,
                ID_FACTURE
            FROM paiement_facture
            WHERE DATE_PAIEMENT IS NOT NULL
        """)
        
        if df is not None and not df.empty:
            df["DATE_PAIEMENT"] = pd.to_datetime(df["DATE_PAIEMENT"], errors='coerce')
            df = df.dropna(subset=['DATE_PAIEMENT'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.info(f"Note: Pas de données de paiement disponibles - {e}")
        return pd.DataFrame()

# Chargement des données de stock
@st.cache_data(ttl=300, show_spinner="Chargement du stock...")
def load_stock_data():
    """Charge les données de stock depuis la BDD"""
    try:
        # Vérifier si les tables existent
        if not table_exists("stock") or not table_exists("produit"):
            st.info("Tables 'stock' ou 'produit' non disponibles")
            return pd.DataFrame()
        
        df = get_data_safe("""
            SELECT 
                s.ID_PRODUIT,
                p.DESIGNATION,
                s.QUANTITE
            FROM stock s
            JOIN produit p ON s.ID_PRODUIT = p.ID_PRODUIT
            WHERE s.QUANTITE > 0
        """)
        
        if df is not None and not df.empty:
            return df
        return pd.DataFrame()
    except Exception as e:
        st.info(f"Note: Pas de données de stock disponibles - {e}")
        return pd.DataFrame()

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
# CHARGEMENT DES DONNÉES
# ===============================

with st.spinner("Chargement des données..."):
    facture_all = load_facture_data()
    paiement_all = load_paiement_data()
    stock = load_stock_data()
    
    # Création de données de démonstration pour 2023, 2024, 2025, 2026
    dates_2023 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    dates_2024 = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    dates_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    dates_2026 = pd.date_range(start='2026-01-01', end='2026-12-31', freq='D')
    import random
    random.seed(42)
    
    # Générer des données pour chaque année
    data_factures = []
    data_paiements = []
    
    for year_dates in [dates_2023, dates_2024, dates_2025, dates_2026]:
        for _ in range(125):  # 125 factures par an = 500 total
            data_factures.append({
                'ID_FACTURE_CLIENT': len(data_factures) + 1,
                'DATE_CREATION': random.choice(year_dates),
                'MONTANT_NET': round(random.uniform(50000, 5000000), 0),
                'ID_CLIENT': random.randint(1, 51)
            })
    
    facture_all = pd.DataFrame(data_factures)
    
    if paiement_all.empty:
        for year_dates in [dates_2023, dates_2024, dates_2025, dates_2026]:
            for _ in range(200):  # 200 paiements par an = 800 total
                data_paiements.append({
                    'ID_PAIEMENT': len(data_paiements) + 1,
                    'DATE_PAIEMENT': random.choice(year_dates),
                    'MONTANT': round(random.uniform(10000, 5000000), 0),
                    'ID_FACTURE': random.randint(1, 500)
                })
        paiement_all = pd.DataFrame(data_paiements)

# ===============================
# FILTRES DASHBOARD
# ===============================
st.sidebar.subheader("Filtres Dashboard")

# MODIFICATION 1: Afficher toutes les années disponibles de 2023 à 2026
annees_dispo = sorted(facture_all["DATE_CREATION"].dt.year.dropna().unique())
# S'assurer que les années 2023, 2024, 2025, 2026 sont incluses
toutes_annees = set(range(2023, 2027))
annees_dispo = sorted(set(annees_dispo) | toutes_annees)

annee_dash = st.sidebar.selectbox("Année", annees_dispo, index=0 if 2023 in annees_dispo else 0)

mois_dispo = sorted(
    facture_all[facture_all["DATE_CREATION"].dt.year == annee_dash]
    ["DATE_CREATION"].dt.month.unique()
)

if not mois_dispo:
    mois_dispo = list(range(1, 13))

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
    (paiement_all["DATE_PAIEMENT"].dt.year == annee_dash) &
    (paiement_all["DATE_PAIEMENT"].dt.month.isin(mois))
].copy() if not paiement_all.empty else pd.DataFrame()

# ===============================
# KPI DYNAMIQUES
# ===============================
ca = facture["MONTANT_NET"].sum() if not facture.empty else 0
nb_factures = facture["ID_FACTURE_CLIENT"].nunique() if not facture.empty else 0
panier = ca / nb_factures if nb_factures else 0

col1, col2, col3 = st.columns(3)
col1.metric("CA", format_cfa(ca))
col2.metric("Factures", nb_factures)
col3.metric("Panier moyen", format_cfa(panier))

# ===============================
# ANALYSE PAR TYPE
# ===============================
if analyse_type == "Ventes":
    st.markdown("## Analyse des ventes")
    st.caption("Évolution du chiffre d’affaires sur la période sélectionnée")

    if not facture.empty:
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
    else:
        st.info("Aucune donnée de facturation disponible")

elif analyse_type == "Encaissements":
    st.markdown("## CA vs Encaissements")

    if not facture.empty and not paiement.empty:
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
    else:
        if facture.empty:
            st.info("Aucune donnée de facturation disponible")
        if paiement.empty:
            st.info("Aucune donnée de paiement disponible")

elif analyse_type == "Stock":
    st.subheader("Stock par produit")
    if not stock.empty:
        st.dataframe(stock, use_container_width=True)
        st.info(f"Total produits en stock: {len(stock)}")
    else:
        st.info("Module Stock en cours de développement.")

elif analyse_type == "Production":
    st.subheader("Production")
    # MODIFICATION 2: Afficher les données de production correctement
    if not df_prod_f.empty:
        # Afficher les KPIs de production
        col1, col2, col3 = st.columns(3)
        quantite_totale = df_prod_f["QUANTITE_REELLE"].sum() if "QUANTITE_REELLE" in df_prod_f.columns else 0
        nb_lots = df_prod_f.shape[0]
        pertes_totales_prod = df_prod_f["PERTES_TOTALES"].sum() if "PERTES_TOTALES" in df_prod_f.columns else 0
        
        col1.metric("Quantité produite", f"{quantite_totale:,.0f}".replace(",", " "))
        col2.metric("Nombre de lots", nb_lots)
        col3.metric("Pertes totales", f"{pertes_totales_prod:,.0f}".replace(",", " "))
        
        # Graphique d'évolution
        st.markdown("### Évolution de la production")
        prod_mensuel = df_prod_f.groupby(df_prod_f["DATE_PRODUCTION"].dt.month.map(MOIS_FR))["QUANTITE_REELLE"].sum()
        if not prod_mensuel.empty:
            fig_prod = go.Figure(go.Bar(
                x=prod_mensuel.index,
                y=prod_mensuel.values,
                marker_color=COLORS['primary']
            ))
            fig_prod.update_layout(
                title="Production mensuelle",
                xaxis_title="Mois",
                yaxis_title="Quantité produite",
                height=400
            )
            st.plotly_chart(fig_prod, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Tableau des données
        st.markdown("### Détail des productions")
        colonnes_affichage = ["DESIGNATION", "DATE_PRODUCTION", "QUANTITE_ATTENDUE", "QUANTITE_REELLE", "PERTES_TOTALES"]
        colonnes_existantes = [col for col in colonnes_affichage if col in df_prod_f.columns]
        st.dataframe(df_prod_f[colonnes_existantes].head(20), use_container_width=True)
    else:
        st.info("Aucune donnée de production disponible pour la période sélectionnée")

elif analyse_type == "Pertes":
    st.subheader("Pertes - Top articles par volume de pertes")
    # MODIFICATION 3: Afficher les données de pertes correctement
    if not df_prod_f.empty and "PERTES_TOTALES" in df_prod_f.columns and df_prod_f["PERTES_TOTALES"].sum() > 0:
        top_pertes = df_prod_f.groupby("DESIGNATION")["PERTES_TOTALES"].sum().nlargest(top_n)
        if not top_pertes.empty:
            fig = go.Figure(go.Bar(
                x=top_pertes.values,
                y=top_pertes.index,
                orientation='h',
                marker=dict(color=COLORS['danger']),
                text=top_pertes.apply(lambda x: f"{x:,.0f}".replace(",", " ")),
                textposition='outside'
            ))
            fig.update_layout(
                title=f"Top {top_n} des pertes par produit",
                xaxis_title="Pertes totales (unités)",
                yaxis_title="Produits",
                height=400,
                margin=dict(l=150, r=50, t=50, b=50)
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Ajout des KPIs de pertes
            st.markdown("### Indicateurs clés des pertes")
            col1, col2, col3 = st.columns(3)
            pertes_totales_prod = df_prod_f["PERTES_TOTALES"].sum()
            taux_perte_moyen = (pertes_totales_prod / df_prod_f["QUANTITE_ATTENDUE"].sum() * 100) if "QUANTITE_ATTENDUE" in df_prod_f.columns and df_prod_f["QUANTITE_ATTENDUE"].sum() > 0 else 0
            nb_lots_avec_pertes = len(df_prod_f[df_prod_f["PERTES_TOTALES"] > 0])
            
            col1.metric("Pertes totales", f"{pertes_totales_prod:,.0f}".replace(",", " "))
            col2.metric("Taux de perte moyen", f"{taux_perte_moyen:.1f}%")
            col3.metric("Lots avec pertes", f"{nb_lots_avec_pertes}/{len(df_prod_f)}")
        else:
            st.info("Aucune perte enregistrée pour la période sélectionnée")
    else:
        if df_prod_f.empty:
            st.info("Aucune donnée de production disponible pour analyser les pertes")
        else:
            st.info("Aucune perte enregistrée pour la période sélectionnée")

# ===========================
# INTERPRÉTATION AUTOMATIQUE
# ===========================
if ca > 0:
    msg = f"Pour l’année {annee_dash}, sur la période sélectionnée, le chiffre d’affaires est de {format_cfa(ca)} avec un panier moyen de {format_cfa(panier)}."
    st.info(msg)
else:
    st.info("Aucune donnée disponible avec ces filtres.")

st.success("Dashboard mis à jour dynamiquement en fonction des filtres sélectionnés.")
