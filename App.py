import streamlit as st
import pandas as pd
import seaborn as sns
from db import get_data, get_data_safe, table_exists

import plotly.graph_objects as go
import plotly.express as px

sns.set_style("whitegrid")

# STYLE GLOBAL
st.set_page_config(
    page_title="Application Data – Sellams",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour éviter les points de suspension et forcer l'affichage complet
st.markdown("""
<style>
    /* Forcer l'affichage complet des métriques */
    .stMetricValue {
        font-size: 1.8rem !important;
        white-space: normal !important;
        word-break: keep-all !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    
    /* Ajustement des colonnes pour les métriques */
    div[data-testid="column"] {
        min-width: 150px !important;
    }
    
    /* Forcer l'affichage complet dans les dataframes */
    .stDataFrame div[data-testid="stHorizontalBlock"] {
        overflow-x: auto !important;
    }
    
    /* Éviter la troncature des textes */
    .stMarkdown, .stCaption, .stText {
        white-space: normal !important;
        word-wrap: break-word !important;
    }
</style>
""", unsafe_allow_html=True)

# CONFIGURATION UNIVERSELLE DES GRAPHIQUES PLOTLY
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
    if table_exists("conditionnement_production"):
        df_conditionnement = get_data_safe("SELECT * FROM conditionnement_production")
    else:
        st.info("Table conditionnement_production non disponible")
        df_conditionnement = pd.DataFrame()

    # Clients (personnes)
    df_personne = get_data("""
        SELECT 
            ID_PERSONNE, 
            NOM 
        FROM personne
    """)

    # Conversion des dates
    if not facture.empty:
        facture["DATE_CREATION"] = pd.to_datetime(facture["DATE_CREATION"], errors="coerce")
    if not paiement.empty:
        paiement["DATE_PAIEMENT"] = pd.to_datetime(paiement["DATE_PAIEMENT"], errors="coerce")
    if not stock.empty:
        stock["DATE_PRODUCTION"] = pd.to_datetime(stock["DATE_PRODUCTION"], errors="coerce")

    # Pour conditionnement - utiliser le bon nom de colonne
    if not df_conditionnement.empty and 'date_production' in df_conditionnement.columns:
        df_conditionnement["date_production"] = pd.to_datetime(
            df_conditionnement["date_production"],
            errors="coerce"
        )

    return facture, paiement, stock, df_produit, df_conditionnement, df_personne


# Chargement unique des données
with st.spinner("Chargement des données..."):
    facture_all, paiement_all, stock_all, df_produit, df_conditionnement, df_personne = load_all_data()

# Renommer les variables pour cohérence
if not facture_all.empty:
    facture = facture_all
else:
    facture = pd.DataFrame()

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

# MERGE PRODUIT & CONDITIONNEMENT
if not df_conditionnement.empty and not df_produit.empty:
    try:
        df_prod = df_conditionnement.merge(
            df_produit,
            left_on="id_article",
            right_on="ID_PRODUIT",
            how="left"
        )

        df_prod["DESIGNATION"] = df_prod["DESIGNATION"].fillna("Article inconnu")

        # NORMALISATION DATE
        if 'date_production' in df_prod.columns:
            df_prod["date_production"] = pd.to_datetime(
                df_prod["date_production"],
                errors="coerce"
            )

        # MÉTRIQUES MÉTIER 
        colonnes_pertes = [
            "perde_en_bouteille",
            "perde_en_capsule",
            "perde_en_etiquette",
            "perde_en_carton",
            "quantite_avarie"
        ]

        colonnes_presentes = [col for col in colonnes_pertes if col in df_prod.columns]

        if colonnes_presentes:
            df_prod["PERTES_TOTALES"] = df_prod[colonnes_presentes].fillna(0).sum(axis=1)
        else:
            df_prod["PERTES_TOTALES"] = 0

        if "quantite_reelle" in df_prod.columns and "quantite_attendue" in df_prod.columns:
            df_prod["ECART_STOCK"] = (
                df_prod["quantite_reelle"].fillna(0) -
                df_prod["quantite_attendue"].fillna(0)
            )
        else:
            df_prod["ECART_STOCK"] = 0

    except Exception as e:
        st.warning(f"Erreur lors du traitement des données de conditionnement: {e}")
        df_prod = pd.DataFrame()
else:
    if df_conditionnement.empty:
        st.info("Données de conditionnement non disponibles pour cette base")
    df_prod = pd.DataFrame()

# CONSTANTES
MOIS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}


def format_cfa(valeur):
    if valeur is None or pd.isna(valeur):
        return "0 FCFA"
    # Format sans abréviation, chiffres complets
    return f"{valeur:,.0f}".replace(",", " ") + " FCFA"


def format_nombre(valeur):
    """Formate les nombres sans abréviation (pas de K, M, etc.)"""
    if valeur is None or pd.isna(valeur):
        return "0"
    return f"{valeur:,.0f}".replace(",", " ")


# SIDEBAR — FILTRES & PÉRIMÈTRE
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

# ========================
# TYPE D'ANALYSE MÉTIER 
# ========================
metier = st.selectbox(
    "Type d’analyse métier",
    ["Ventes", "Encaissements", "Stock", "Production", "Pertes"],
    key="metier_select"
)

# ==================================================
# STOCKAGE DES MÉTADONNÉES POUR LE RAPPORT PDF
# ==================================================

st.session_state["metier_app"] = metier
st.session_state["mois_selectionnes_app"] = mois_selectionnes
st.session_state["annee_app"] = annee

# Stockage des DataFrames filtrés pour le rapport
st.session_state["facture_f"] = facture_client.copy() if not facture_client.empty else pd.DataFrame()
st.session_state["paiement_facture"] = paiement_facture.copy() if not paiement_facture.empty else pd.DataFrame()
st.session_state["df_prod"] = df_prod.copy() if not df_prod.empty else pd.DataFrame()
st.session_state["stock_data"] = stock.copy() if not stock.empty else pd.DataFrame()

# TITRE & CONTEXTE
st.title("Vue globale & Pilotage")
st.caption(f"Analyse consolidée – Année {annee} | Analyse : {metier}")
st.caption("Source : Base de données MySQL – sellams_namm")

st.divider()

# ===============
# KPI EXÉCUTIFS 
# ===============

if metier == "Ventes":
    # KPIs Ventes
    ca_total = facture_client["MONTANT_NET"].sum() if not facture_client.empty else 0
    nb_factures = facture_client["ID_FACTURE_CLIENT"].nunique() if not facture_client.empty else 0
    panier_moyen = ca_total / nb_factures if nb_factures > 0 else 0
    
    # ===== AJOUTER CES LIGNES =====
    st.session_state["ca_total_ventes"] = ca_total
    st.session_state["nb_factures_ventes"] = nb_factures
    st.session_state["panier_moyen_ventes"] = panier_moyen
    
    # Stocker les données mensuelles pour le tableau
    if not facture_client.empty:
        facture_client["MOIS_NOM"] = facture_client["DATE_CREATION"].dt.month.map(MOIS_FR)
        ventes_mensuelles = facture_client.groupby("MOIS_NOM", sort=False)["MONTANT_NET"].sum().reset_index()
        st.session_state["ventes_mensuelles"] = ventes_mensuelles
    
    # Utilisation de colonnes avec largeur automatique
    col1, col2, col3 = st.columns(3, gap="large")
    col1.metric("Chiffre d'affaires", format_cfa(ca_total), help="Montant total des ventes sur la période")
    col2.metric("Nombre de factures", format_nombre(nb_factures), help="Nombre total de factures validées")
    col3.metric("Panier moyen", format_cfa(panier_moyen), help="Montant moyen par facture")
    
    # Graphique Ventes
    if not facture_client.empty:
        facture_client["MOIS_NOM"] = facture_client["DATE_CREATION"].dt.month.map(MOIS_FR)
        ventes_mensuelles = facture_client.groupby("MOIS_NOM", sort=False)["MONTANT_NET"].sum().reset_index()
        
        if not ventes_mensuelles.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ventes_mensuelles["MOIS_NOM"],
                y=ventes_mensuelles["MONTANT_NET"],
                name='Ventes',
                marker_color='#1f77b4'
            ))
            fig.update_layout(
                title="Évolution mensuelle des ventes",
                xaxis_title="Mois",
                yaxis_title="Montant (FCFA)",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des données
            st.markdown("### Détail mensuel des ventes")
            tableau_ventes = ventes_mensuelles.copy()
            tableau_ventes["MONTANT_NET"] = tableau_ventes["MONTANT_NET"].apply(format_cfa)
            tableau_ventes.columns = ["Mois", "Chiffre d'affaires"]
            st.dataframe(tableau_ventes, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune donnée de ventes disponible")
    else:
        st.info("Aucune facture trouvée pour la période sélectionnée")

elif metier == "Encaissements":
    encaisse = paiement_facture["MONTANT"].sum() if not paiement_facture.empty else 0
    ca_total = facture_client["MONTANT_NET"].sum() if not facture_client.empty else 0
    taux_enc = (encaisse / ca_total * 100) if ca_total > 0 else 0
    nb_paiements = paiement_facture["ID_PAIEMENT_FACTURE"].nunique() if not paiement_facture.empty else 0
    
    # ===== AJOUTER CES LIGNES =====
    st.session_state["total_encaisse"] = encaisse
    st.session_state["taux_encaissement"] = taux_enc
    st.session_state["nb_paiements"] = nb_paiements
    
    if not paiement_facture.empty:
        paiement_facture["MOIS_NOM"] = paiement_facture["DATE_PAIEMENT"].dt.month.map(MOIS_FR)
        encaissements_mensuels = paiement_facture.groupby("MOIS_NOM", sort=False)["MONTANT"].sum().reset_index()
        st.session_state["encaissements_mensuels"] = encaissements_mensuels
    
    col1, col2, col3, col4 = st.columns(4, gap="large")
    col1.metric("Total encaissé", format_cfa(encaisse), help="Montant total des paiements reçus")
    col2.metric("Taux d'encaissement", f"{taux_enc:.1f} %", help="Pourcentage du CA encaissé")
    col3.metric("Nombre de paiements", format_nombre(nb_paiements), help="Nombre total de transactions")
    col4.metric("CA facturé", format_cfa(ca_total), help="Chiffre d'affaires total facturé")
    
    # Graphique Encaissements
    if not paiement_facture.empty and 'DATE_PAIEMENT' in paiement_facture.columns:
        paiement_facture["MOIS_NOM"] = paiement_facture["DATE_PAIEMENT"].dt.month.map(MOIS_FR)
        encaissements_mensuels = paiement_facture.groupby("MOIS_NOM", sort=False)["MONTANT"].sum().reset_index()
        
        if not encaissements_mensuels.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=encaissements_mensuels["MOIS_NOM"],
                y=encaissements_mensuels["MONTANT"],
                name='Encaissements',
                marker_color='#2ecc71'
            ))
            fig.update_layout(
                title="Évolution mensuelle des encaissements",
                xaxis_title="Mois",
                yaxis_title="Montant (FCFA)",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des données
            st.markdown("### Détail mensuel des encaissements")
            tableau_encaissements = encaissements_mensuels.copy()
            tableau_encaissements["MONTANT"] = tableau_encaissements["MONTANT"].apply(format_cfa)
            tableau_encaissements.columns = ["Mois", "Montant encaissé"]
            st.dataframe(tableau_encaissements, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune donnée d'encaissements disponible")
    else:
        st.info("Aucun paiement trouvé pour la période")

elif metier == "Stock":
    # KPIs Stock
    stock_total = stock["QUANTITE"].sum() if not stock.empty and 'QUANTITE' in stock.columns else 0
    nb_produits = stock["ID_STOCK"].nunique() if not stock.empty else 0
    stock_moyen = stock_total / nb_produits if nb_produits > 0 else 0
    
    # ===== AJOUTER CES LIGNES =====
    st.session_state["stock_total"] = stock_total
    st.session_state["nb_produits"] = nb_produits
    st.session_state["stock_data_detail"] = stock[["DESIGNATION", "QUANTITE"]].head(20) if not stock.empty else pd.DataFrame()
    
    col1, col2, col3 = st.columns(3, gap="large")
    col1.metric("Stock total (unités)", format_nombre(stock_total), help="Nombre total d'unités en stock")
    col2.metric("Nombre de références", format_nombre(nb_produits), help="Nombre de produits différents")
    col3.metric("Stock moyen par référence", f"{stock_moyen:.1f}", help="Moyenne d'unités par produit")
    
    # Graphique Top produits
    if not stock.empty:
        top_stock = stock.nlargest(10, "QUANTITE")[["DESIGNATION", "QUANTITE"]].dropna()
        if not top_stock.empty:
            fig = px.bar(
                top_stock,
                x="QUANTITE",
                y="DESIGNATION",
                orientation='h',
                title="Top 10 des produits en stock",
                labels={"QUANTITE": "Quantité", "DESIGNATION": "Produit"}
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé du stock
            st.markdown("### Détail du stock par produit")
            stock_display = stock[["DESIGNATION", "QUANTITE"]].dropna().sort_values("QUANTITE", ascending=False)
            stock_display["QUANTITE"] = stock_display["QUANTITE"].apply(format_nombre)
            stock_display.columns = ["Produit", "Quantité en stock"]
            st.dataframe(stock_display.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("Données de stock insuffisantes")
    else:
        st.info("Aucune donnée de stock disponible")

elif metier == "Production":
    # KPIs Production
    if not df_prod.empty:
        quantite_totale = df_prod["quantite_reelle"].sum() if "quantite_reelle" in df_prod.columns else 0
        pertes_totales = df_prod["PERTES_TOTALES"].sum() if "PERTES_TOTALES" in df_prod.columns else 0
        taux_pertes = (pertes_totales / quantite_totale * 100) if quantite_totale > 0 else 0
        nb_lots = df_prod["id_article"].nunique() if "id_article" in df_prod.columns else 0
        
        st.session_state["quantite_produite"] = quantite_totale
        st.session_state["pertes_prod"] = pertes_totales
        st.session_state["taux_pertes"] = taux_pertes
        st.session_state["nb_lots_production"] = nb_lots
        
        col1, col2, col3, col4 = st.columns(4, gap="large")
        col1.metric("Production totale", f"{format_nombre(quantite_totale)} unités", help="Nombre total d'unités produites")
        col2.metric("Pertes totales", f"{format_nombre(pertes_totales)} unités", help="Nombre total d'unités perdues")
        col3.metric("Taux de pertes", f"{taux_pertes:.1f} %", help="Pourcentage de production perdue")
        col4.metric("Nombre de lots", format_nombre(nb_lots), help="Nombre total de lots de production")
        
        # Graphique Pertes par lot
        if "PERTES_TOTALES" in df_prod.columns and "DESIGNATION" in df_prod.columns:
            pertes_par_produit = df_prod.groupby("DESIGNATION")["PERTES_TOTALES"].sum().reset_index()
            pertes_par_produit = pertes_par_produit.nlargest(10, "PERTES_TOTALES")
            
            if not pertes_par_produit.empty:
                fig = px.bar(
                    pertes_par_produit,
                    x="DESIGNATION",
                    y="PERTES_TOTALES",
                    title="Pertes par produit (Top 10)",
                    labels={"PERTES_TOTALES": "Pertes (unités)", "DESIGNATION": "Produit"}
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau détaillé des pertes
                st.markdown("### Détail des pertes par produit")
                pertes_display = pertes_par_produit.copy()
                pertes_display["PERTES_TOTALES"] = pertes_display["PERTES_TOTALES"].apply(format_nombre)
                pertes_display.columns = ["Produit", "Pertes (unités)"]
                st.dataframe(pertes_display, use_container_width=True, hide_index=True)
            else:
                st.info("Aucune donnée de pertes disponible")
    else:
        st.info("Aucune donnée de production disponible")
        col1, col2 = st.columns(2)
        col1.metric("Production totale", "N/A")
        col2.metric("Pertes totales", "N/A")

elif metier == "Pertes":
    # KPIs Pertes
    if not df_prod.empty and "PERTES_TOTALES" in df_prod.columns:
        pertes_totales = df_prod["PERTES_TOTALES"].sum()
        nb_lots_avec_pertes = (df_prod["PERTES_TOTALES"] > 0).sum()
        pertes_moyennes = df_prod["PERTES_TOTALES"].mean()
        
        st.session_state["pertes_totales"] = pertes_totales
        st.session_state["nb_lots_avec_pertes"] = nb_lots_avec_pertes
        st.session_state["pertes_moyennes"] = pertes_moyennes
        
        # Stocker le détail des pertes par type
        colonnes_pertes = ["perde_en_bouteille", "perde_en_capsule", "perde_en_etiquette", "perde_en_carton", "quantite_avarie"]
        colonnes_presentes = [col for col in colonnes_pertes if col in df_prod.columns]
        if colonnes_presentes:
            pertes_par_type = df_prod[colonnes_presentes].sum().reset_index()
            pertes_par_type.columns = ["Type de perte", "Quantité"]
            st.session_state["pertes_par_type"] = pertes_par_type
        
        col1, col2, col3 = st.columns(3, gap="large")
        col1.metric("Pertes totales", f"{format_nombre(pertes_totales)} unités", help="Nombre total d'unités perdues")
        col2.metric("Lots avec pertes", format_nombre(nb_lots_avec_pertes), help="Nombre de lots ayant subi des pertes")
        col3.metric("Pertes moyennes/lot", f"{pertes_moyennes:.1f}", help="Moyenne des pertes par lot")
        
        # Détail des pertes par type
        colonnes_pertes = ["perde_en_bouteille", "perde_en_capsule", "perde_en_etiquette", "perde_en_carton", "quantite_avarie"]
        colonnes_presentes = [col for col in colonnes_pertes if col in df_prod.columns]
        
        if colonnes_presentes:
            pertes_par_type = df_prod[colonnes_presentes].sum().reset_index()
            pertes_par_type.columns = ["Type de perte", "Quantité"]
            
            # Nettoyer les noms pour l'affichage
            type_labels = {
                "perde_en_bouteille": "Bouteilles",
                "perde_en_capsule": "Capsules",
                "perde_en_etiquette": "Étiquettes",
                "perde_en_carton": "Cartons",
                "quantite_avarie": "Marchandise avariée"
            }
            pertes_par_type["Type de perte"] = pertes_par_type["Type de perte"].map(type_labels).fillna(pertes_par_type["Type de perte"])
            
            fig = px.pie(
                pertes_par_type,
                values="Quantité",
                names="Type de perte",
                title="Répartition des pertes par type",
                hole=0.3
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé des pertes par type
            st.markdown("### Détail des pertes par type")
            pertes_type_display = pertes_par_type.copy()
            pertes_type_display["Quantité"] = pertes_type_display["Quantité"].apply(format_nombre)
            st.dataframe(pertes_type_display, use_container_width=True, hide_index=True)
        else:
            st.info("Données détaillées des pertes non disponibles")
    else:
        st.info("Aucune donnée de pertes disponible")
        st.metric("Pertes totales", "N/A")

st.divider()

# ==================
# ALERTES & RISQUES
# ==================

st.subheader("Alertes & points d’attention")

if metier == "Ventes":
    ca_total = facture_client["MONTANT_NET"].sum() if not facture_client.empty else 0
    if ca_total == 0:
        st.error("Aucun chiffre d'affaires enregistré sur la période.")
    elif ca_total < 1000000:
        st.warning("Chiffre d'affaires faible sur la période.")
    else:
        st.success("Niveau de chiffre d'affaires satisfaisant.")

elif metier == "Encaissements":
    encaisse = paiement_facture["MONTANT"].sum() if not paiement_facture.empty else 0
    ca_total = facture_client["MONTANT_NET"].sum() if not facture_client.empty else 0
    taux_enc = (encaisse / ca_total * 100) if ca_total > 0 else 0
    
    if taux_enc < 80:
        st.warning(f"Taux d’encaissement inférieur au seuil recommandé (80%) : {taux_enc:.1f}%")
        st.caption("Action recommandée : Renforcer le recouvrement des créances clients.")
    else:
        st.success(f"Taux d’encaissement satisfaisant : {taux_enc:.1f}%")

elif metier == "Stock":
    stock_total = stock["QUANTITE"].sum() if not stock.empty and 'QUANTITE' in stock.columns else 0
    if stock_total <= 0:
        st.warning("Stock global nul ou non renseigné.")
        st.caption("Action recommandée : Vérifier l'inventaire physique.")
    elif stock_total < 1000:
        st.warning(f"Stock faible : {format_nombre(stock_total)} unités. Risque de rupture.")
    else:
        st.info(f"Niveau de stock actuel : {format_nombre(stock_total)} unités")

elif metier == "Production":
    if not df_prod.empty and "PERTES_TOTALES" in df_prod.columns:
        quantite_totale = df_prod["quantite_reelle"].sum() if "quantite_reelle" in df_prod.columns else 0
        pertes_totales = df_prod["PERTES_TOTALES"].sum()
        taux_pertes = (pertes_totales / quantite_totale * 100) if quantite_totale > 0 else 0
        
        if taux_pertes > 10:
            st.warning(f"Taux de pertes élevé : {taux_pertes:.1f}% (seuil recommandé < 10%)")
            st.caption("Action recommandée : Auditer les processus de production.")
        else:
            st.success(f"Taux de pertes maîtrisé : {taux_pertes:.1f}%")
    else:
        st.info("Données de production insuffisantes pour générer des alertes")

elif metier == "Pertes":
    if not df_prod.empty and "PERTES_TOTALES" in df_prod.columns:
        pertes_totales = df_prod["PERTES_TOTALES"].sum()
        if pertes_totales > 1000:
            st.warning(f"Pertes totales élevées : {format_nombre(pertes_totales)} unités")
            st.caption("Action recommandée : Analyser les causes racines des pertes.")
        elif pertes_totales > 0:
            st.info(f"Pertes totales : {format_nombre(pertes_totales)} unités - À surveiller.")
        else:
            st.success("Aucune perte enregistrée sur la période.")
    else:
        st.info("Données de pertes non disponibles")

# ======================
# LECTURE DES RESULTATS
# ======================

st.subheader("Interprétation des résultats")

resume_metier = {
    "Ventes": "analyse les performances commerciales et le chiffre d'affaires",
    "Encaissements": "examine le recouvrement et la trésorerie",
    "Stock": "évalue les niveaux d'inventaire et la rotation",
    "Production": "analyse les volumes produits et les pertes de fabrication",
    "Pertes": "détaille les pertes par type et par produit"
}

st.info(f"**Mode {metier}** : Cette vue {resume_metier.get(metier, 'analyse le métier sélectionné')} sur la période {annee}.")

st.divider()

# ========================
# ORIENTATION UTILISATEUR
# ========================

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

# ========================
# INFORMATIONS DE DÉBOGAGE
# ========================

with st.expander("Informations techniques"):
    st.write("**Statut des données :**")
    st.write(f"- Factures chargées : {len(facture)} lignes")
    st.write(f"- Paiements chargés : {len(paiement)} lignes")
    st.write(f"- Stock : {len(stock)} produits")
    st.write(f"- Produits : {len(df_produit)} références")
    st.write(f"- Conditionnement : {len(df_conditionnement)} enregistrements")
    st.write(f"- Clients : {len(df_personne)} personnes")
    st.write(f"- Période filtrée : {len(facture_client)} factures")
    st.write(f"- Mode actif : {metier}")
