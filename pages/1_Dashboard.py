import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db import get_data
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sns.set_style("whitegrid")

# ===============================
# CHARGEMENT DES DONNÉES BDD
# ===============================

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

df_produit = get_data("""
    SELECT
        ID_PRODUIT,
        DESIGNATION
    FROM produit
""")

# ===============================
# MERGE PRODUIT ↔ CONDITIONNEMENT
# ===============================
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
    """
    Formate un nombre en Franc CFA
    Exemple : 1250000 -> 1 250 000 FCFA
    """
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

sns.set_style("whitegrid")

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

# ---- Filtre année (local au dashboard)
annees_dispo = sorted(facture_all["DATE_CREATION"].dt.year.dropna().unique())
annee_dash = st.sidebar.selectbox("Année", annees_dispo, index=0)

# ---- Filtre mois
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


df_prod_f = df_prod[
    (df_prod["DATE_PRODUCTION"].dt.year == annee_dash) &
    (df_prod["DATE_PRODUCTION"].dt.month.isin(mois))
].copy()

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
# PARAMÈTRES DE TRI (JUMELÉS)
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

    # ---------- ANALYSE DES VENTES ----------
    st.markdown("## Analyse des ventes")
    st.caption("Évolution du chiffre d’affaires sur la période sélectionnée")

    facture["MOIS_NUM"] = facture["DATE_CREATION"].dt.month
    facture["MOIS_NOM"] = facture["MOIS_NUM"].map(MOIS_FR)

    ca_mensuel = (
        facture
        .groupby("MOIS_NOM", sort=False)["MONTANT_NET"]
        .sum()
    )

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
 
    st.info(
        "Cette courbe met en évidence les périodes de croissance ou de ralentissement "
        "de l’activité commerciale."
    )

    # ===============================
    # ÉVOLUTION DES PRODUCTIONS
    # ===============================
    st.subheader("Évolution des productions")

    if "QUANTITE_PRODUITE" in stock.columns and "DATE_PRODUCTION" in stock.columns:
        stock["DATE_PRODUCTION"] = pd.to_datetime(
            stock["DATE_PRODUCTION"], errors="coerce"
        )
        stock["MOIS_NUM"] = stock["DATE_PRODUCTION"].dt.month
        stock["MOIS_NOM"] = stock["MOIS_NUM"].map(MOIS_FR)

        prod_mensuelle = (
            stock
            .groupby("MOIS_NOM", sort=False)["QUANTITE_PRODUITE"]
            .sum()
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=prod_mensuelle.index.tolist(),
            y=prod_mensuelle.values.tolist(),
            mode="lines+markers",
            name='Production',
            line=dict(color=COLORS['secondary'],width=3),
            marker=dict(size=8, symbol='square', color="#ff7f0e"),
            hovertemplate='<b>%{x}</b><br>Production: %{y:,.0f} unités<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text="Évolution mensuelle de la production", x=0.5, font=dict(size=16)),
            xaxis_title="Mois",
            yaxis_title="Quantité produite",
            hovermode="x unified",
            template="plotly_white",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(tickangle=45),
        )

        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.divider()

    # ===============================
    # FLUX COMMANDES / LIVRAISONS
    # ===============================
    st.subheader("Flux commandes & livraisons")

    if not facture.empty:
        nb_commandes = facture["ID_FACTURE_CLIENT"].nunique()
        nb_livraisons = stock["ID_MAGASIN"].nunique() if "ID_MAGASIN" in stock.columns else 0

        col_x, col_y = st.columns(2)
        col_x.metric("Commandes traitées", nb_commandes)
        col_y.metric("Points de livraison actifs", nb_livraisons)

        st.caption(
            "Ces indicateurs donnent une vision synthétique du flux opérationnel "
            "entre commandes clients et livraisons."
        )

    # ===============================
    # COMPARATIF VENTES VS PRODUCTIONS
    # ===============================
    st.subheader("Ventes vs Productions")

    if "QUANTITE_PRODUITE" in stock.columns and "DATE_PRODUCTION" in stock.columns:
        stock["DATE_PRODUCTION"] = pd.to_datetime(stock["DATE_PRODUCTION"], errors="coerce")
        stock["MOIS_NUM"] = stock["DATE_PRODUCTION"].dt.month
        stock["MOIS_NOM"] = stock["MOIS_NUM"].map(MOIS_FR)
        prod_mensuelle = stock.groupby("MOIS_NOM", sort=False)["QUANTITE_PRODUITE"].sum()
        
        if not prod_mensuelle.empty and not ca_mensuel.empty:
            df_compare = pd.DataFrame({
                "Ventes": ca_mensuel,
                "Production": prod_mensuelle
            }).fillna(0)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_compare.index.tolist(),
                y=df_compare["Ventes"].values.tolist(),
                mode="lines+markers",
                name='Ventes',
                line=dict(color=COLORS['primary'],width=3),
                marker=dict(size=8, symbol='circle', color="#1f77b4"),
                hovertemplate='<b>%{x}</b><br>Ventes: %{y:,.0f} FCFA<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=df_compare.index.tolist(),
                y=df_compare["Production"].tolist(),
                mode="lines+markers",
                name='Production',
                line=dict(color=COLORS['secondary'],width=3, dash='dash'),
                marker=dict(size=8, symbol='square', color="#ff7f0e"),
                hovertemplate='<b>%{x}</b><br>Production: %{y:,.0f} unités<extra></extra>'
            ))

            fig.update_layout(
                title=dict(text="Ventes vs Productions", x=0.5, font=dict(size=16)),
                xaxis_title="Mois",
                yaxis_title="Valeur / Volume",
                hovermode="x unified",
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                template="plotly_white",
                height=450,
                margin=dict(l=50, r=50, t=80, b=50),
                xaxis=dict(tickangle=45)
            )

            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Données de production insuffisantes pour comparaison.")
    
    # ---------- TOP CLIENTS ----------
    st.markdown("## Concentration du chiffre d’affaires par client")
    st.caption("Identification des clients stratégiques")

    df_personne = get_data("SELECT ID_PERSONNE, NOM FROM personne")

    facture_client = facture.merge(
        df_personne,
        on="ID_PERSONNE",
        how="left"
    )
    facture_client["NOM"] = facture_client["NOM"].fillna("Client inconnu")

    clients_dispo = sorted(facture_client["NOM"].unique())
    clients_selectionnes = st.multiselect(
        "Sélectionner un ou plusieurs clients",
        options=clients_dispo,
        default=clients_dispo
    )

    facture_client = facture_client[
        facture_client["NOM"].isin(clients_selectionnes)
    ]

    top_clients = (
        facture_client
        .groupby("NOM")["MONTANT_NET"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig = go.Figure(go.Bar(
        x=top_clients.values,
        y=top_clients.index,
        orientation='h',
        marker=dict(
            color=top_clients.values,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="CA (FCFA)")
            ),
        hovertemplate='<b>%{y}</b><br>CA: %{x:,.0f} FCFA<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=f"Top {top_n} clients par chiffre d’affaires", x=0.5, font=dict(size=16)),
        xaxis_title="Chiffre d’affaires (FCFA)",
        yaxis_title="Client",
        template="plotly_white",
        height=400,
        margin=dict(l=100, r=20, t=50, b=20),
        xaxis=dict(tickformat=",.0f", tickprefix='', ticksuffix=' FCFA')
    )

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    total_ca = facture_client["MONTANT_NET"].sum()
    part_top = (top_clients.sum() / total_ca * 100) if total_ca > 0 else 0

    st.info(
        f"Les 5 principaux clients représentent {part_top:.1f} % du chiffre d’affaires."
    )

    if part_top > 60:
        st.warning(
            "Forte dépendance commerciale à un nombre limité de clients."
        )
    else:
        st.success(
            "Répartition équilibrée du chiffre d’affaires."
        )

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

elif analyse_type == "Stock":
    st.subheader("Stock par produit")

    if "ID_PRODUIT" in stock.columns or "ID_ARTICLE" in stock.columns:
        id_col = "ID_PRODUIT" if "ID_PRODUIT" in stock.columns else "ID_ARTICLE"
        
        df_stock_produit = stock.merge(
            df_produit,
            left_on=id_col,
            right_on="ID_PRODUIT",
            how="left"
        )
        df_stock_produit["DESIGNATION"] = df_stock_produit["DESIGNATION"].fillna("Produit inconnu")

        stock_par_produit = (
            df_stock_produit
            .groupby("DESIGNATION")["QUANTITE"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
    
        fig = go.Figure(go.Bar(
            x=stock_par_produit.values,
            y=stock_par_produit.index,
            orientation='h',
            marker_color=COLORS['primary'],
            hovertemplate='<b>%{y}</b><br>Stock: %{x:,.0f} unités<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text=f"Top {top_n} produits en stock", x=0.5, font=dict(size=16)),
            xaxis_title="Quantité en stock",
            yaxis_title="Produit",
            height=400,
            margin=dict(l=120, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        st.warning("Données de stock non disponibles pour la période sélectionnée.")

elif analyse_type == "Production":
    st.subheader("Production")
    st.info("Module Production en cours de développement.")

elif analyse_type == "Pertes":
    st.subheader("Pertes - Top articles par volume de pertes")
    
    if not df_prod_f.empty and "PERTES_TOTALES" in df_prod_f.columns:
        pertes_article = (
            df_prod_f
            .groupby("DESIGNATION")["PERTES_TOTALES"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        
        if not pertes_article.empty:
            fig = go.Figure(go.Bar(
                x=pertes_article.values,
                y=pertes_article.index,
                orientation='h',
                marker=dict(
                    color=pertes_article.values,
                    colorscale='Oranges',
                    showscale=True,
                    colorbar=dict(title="Volume de pertes")
                ),
                hovertemplate='<b>%{y}</b><br>Volume de pertes: %{x:,.0f} unités<extra></extra>'
            ))

            fig.update_layout(
                title=dict(text=f"Top {top_n} articles par volume de pertes", x=0.5, font=dict(size=16)),
                xaxis_title="Volume de pertes",
                yaxis_title="Article",
                height=400,
                margin=dict(l=120, r=20, t=50, b=20)
            )

            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Aucune perte enregistrée sur la période sélectionnée.")
    else:
        st.warning("Données de pertes non disponibles pour la période sélectionnée.")

# ===============================
# INTERPRÉTATION AUTOMATIQUE
# ===============================
if ca > 0:
    msg = (
        f"Pour l’année {annee_dash}, sur la période sélectionnée, "
        f"le chiffre d’affaires est de {format_cfa(ca)} "
        f"avec un panier moyen de {format_cfa(panier)}."
    )

    if panier < facture_all["MONTANT_NET"].mean():
        msg += " Le panier moyen est inférieur à la moyenne globale."

    st.info(msg)
else:
    st.info("Aucune donnée disponible avec ces filtres.")

st.success("Dashboard mis à jour dynamiquement en fonction des filtres sélectionnés.")

# ===============================
# CONTEXTE POUR EXPORT PDF
# ===============================

if analyse_type in ["Production", "Pertes", "Stock"]:
    data_for_report = df_prod_f.copy()
elif analyse_type == "Ventes":
    data_for_report = facture.copy()
elif analyse_type == "Encaissements":
    data_for_report = paiement.copy()
else:
    data_for_report = None

st.session_state["report_context"] = {
    "type_analyse": analyse_type,
    "annee": annee_dash,
    "mois": mois,
    "data": data_for_report
}