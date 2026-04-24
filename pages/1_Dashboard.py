import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db import get_data

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

sns.set_style("whitegrid")

st.title("Dashboard interactif")

# ===============================
# SÉCURITÉ SESSION
# ===============================
required = ["facture_f", "paiement_f", "stock"]
if not all(k in st.session_state for k in required):
    st.warning("Veuillez d’abord configurer les filtres dans l’onglet App.")
    st.stop()

facture_all = st.session_state["facture_f"].copy()
paiement_all = st.session_state["paiement_f"].copy()
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

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(
        x=ca_mensuel.index,
        y=ca_mensuel.values,
        marker="o",
        linewidth=3,
        color="#1f77b4",
        ax=ax
    )

    ax.set_title("Évolution mensuelle du chiffre d’affaires", fontsize=14, weight="bold")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Chiffre d’affaires (FCFA)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)

    st.pyplot(fig)
 
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

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(
        x=prod_mensuelle.index,
        y=prod_mensuelle.values,
        marker="o",
        linewidth=3,
        color="#ff7f0e",
        ax=ax
    )
    ax.set_ylabel("Quantité produite")
    ax.set_xlabel("Mois")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.info("Données de production non disponibles.")

    st.divider()

# ===============================
# FLUX COMMANDES / LIVRAISONS
# ===============================
    st.subheader("Flux commandes & livraisons")

if not facture.empty:
    nb_commandes = facture["ID_FACTURE_CLIENT"].nunique()
    nb_livraisons = stock["ID_MAGASIN"].nunique()

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

if "QUANTITE_PRODUITE" in stock.columns:
    df_compare = pd.DataFrame({
        "Ventes": ca_mensuel,
        "Production": prod_mensuelle
    }).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(x=df_compare.index, y=df_compare["Ventes"], label="Ventes", ax=ax)
    sns.lineplot(x=df_compare.index, y=df_compare["Production"], label="Production", ax=ax)

    ax.set_ylabel("Volume / Valeur")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.caption(
        "Ce graphique permet d’identifier les écarts entre capacité de production "
        "et volumes effectivement vendus."
    )

    # ---------- TOP CLIENTS ----------
    st.markdown("## Concentration du chiffre d’affaires par client")
    st.caption("Identification des clients stratégiques")

    from db import get_data
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

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        x=top_clients.values,
        y=top_clients.index,
        palette="Blues_r",
        ax=ax
    )

    ax.set_title("Top 5 clients par chiffre d’affaires", fontsize=14, weight="bold")
    ax.set_xlabel("Chiffre d’affaires (FCFA)")
    ax.set_ylabel("Client")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    st.pyplot(fig)

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

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(x=df_compare.index, y=df_compare["CA"], label="CA", ax=ax)
    sns.lineplot(x=df_compare.index, y=df_compare["Encaissements"], label="Encaissements", ax=ax)

    ax.set_title("CA vs Encaissements")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif analyse_type == "Stock":
        st.subheader("Stock")
        st.bar_chart(stock.groupby("ID_MAGASIN")["QUANTITE"].sum())

elif analyse_type == "Production":
    st.subheader("Production")

elif analyse_type == "Pertes":
    st.subheader("Pertes")

    
    pertes_article = (
        df_prod_f
        .groupby("DESIGNATION")["PERTES_TOTALES"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        x=pertes_article.values,
        y=pertes_article.index,
        palette="Oranges_r",
        ax=ax
    )
    ax.set_xlabel("Volume de pertes")
    ax.set_ylabel("Article")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    st.pyplot(fig)

# ===============================
# INTERPRÉTATION AUTOMATIQUE
# ===============================
    st.subheader("Lecture automatique")
    
else:
    st.info("Veuillez sélectionner un type d’analyse.")


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