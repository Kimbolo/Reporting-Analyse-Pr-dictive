import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db import get_data

sns.set_style("whitegrid")

# =========================================================
# CONFIG & CONTEXTE
# =========================================================
st.title("Analytics — Audit de performance")
st.caption("Analyse expliquée et diagnostique des performances commerciales")

required = ["facture_f", "paiement_f", "stock", "annee"]
if not all(k in st.session_state for k in required):
    st.warning("Veuillez d’abord configurer les filtres dans l’onglet App.")
    st.stop()

facture = st.session_state["facture_f"].copy()
paiement = st.session_state["paiement_f"].copy()
stock = st.session_state["stock"].copy()
annee = st.session_state["annee"]

MOIS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}

facture["MOIS"] = facture["DATE_CREATION"].dt.month
facture["MOIS_NOM"] = facture["MOIS"].map(MOIS_FR)
paiement["MOIS"] = paiement["DATE_PAIEMENT"].dt.month
paiement["MOIS_NOM"] = paiement["MOIS"].map(MOIS_FR)

# =========================================================
# 1. CONTEXTE D’AUDIT
# =========================================================
st.markdown("## Contexte analytique")

st.info(
    f"Analyse de la performance commerciale pour l’année **{annee}**\n"
    "Données basées sur ventes validées, encaissements réels et niveaux de stock\n"
    "Objectif : comprendre les causes, les risques et les leviers d’amélioration"
)

# =========================================================
# 2. DYNAMIQUE DES VENTES
# =========================================================
st.markdown("## Dynamique du chiffre d’affaires")

ca_mensuel = facture.groupby("MOIS_NOM", sort=False)["MONTANT_NET"].sum()
variation = ca_mensuel.pct_change() * 100

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(
    x=ca_mensuel.index,
    y=ca_mensuel.values,
    marker="o",
    linewidth=3,
    color="#1f77b4",
    ax=ax
)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.set_title("Évolution mensuelle du chiffre d’affaires")
ax.set_xlabel("Mois")
ax.set_ylabel("CA")
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("### Lecture")
st.markdown(
    "- Identification des périodes de croissance et de ralentissement\n"
    "- Les variations importantes indiquent un effet saisonnier ou opérationnel\n"
    "- Une baisse prolongée peut signaler une contrainte structurelle"
)

# =========================================================
# 3. MEILLEUR CLIENT & CONCENTRATION
# =========================================================
st.markdown("## Analyse clients — concentration du CA")

df_personne = get_data("SELECT ID_PERSONNE, NOM FROM personne")
facture_client = facture.merge(df_personne, on="ID_PERSONNE", how="left")
facture_client["NOM"] = facture_client["NOM"].fillna("Client inconnu")

ca_client = (
    facture_client.groupby("NOM")["MONTANT_NET"]
    .sum()
    .sort_values(ascending=False)
)

top5 = ca_client.head(5)
top10 = ca_client.head(10)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    x=top10.values,
    y=top10.index,
    palette="Blues_r",
    ax=ax
)
ax.grid(axis="x", linestyle="--", alpha=0.6)
ax.set_title("Top 10 clients par chiffre d’affaires")
ax.set_xlabel("CA")
ax.set_ylabel("Client")
st.pyplot(fig)

part_top5 = top5.sum() / ca_client.sum() * 100
part_top10 = top10.sum() / ca_client.sum() * 100

st.markdown("### Lecture")
st.markdown(
    f"- Meilleur client = **{top5.index[0]}**\n"
    f"- Top 5 clients = **{part_top5:.1f} %** du CA\n"
    f"- Top 10 clients = **{part_top10:.1f} %** du CA\n"
)

if part_top5 > 60:
    st.warning("Forte dépendance commerciale détectée.")
else:
    st.success("Répartition client relativement équilibrée.")

# =========================================================
# 4. MEILLEURS ARTICLES VENDUS
# =========================================================
st.markdown("## Articles générateurs de performance")

if "ID_PRODUIT" in facture.columns:
    ca_produit = (
        facture.groupby("ID_PRODUIT")["MONTANT_NET"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(
        x=ca_produit.values,
        y=ca_produit.index.astype(str),
        palette="Greens_r",
        ax=ax
    )
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.set_title("Top articles par chiffre d’affaires")
    st.pyplot(fig)

    st.markdown(
        "- Quelques articles tirent majoritairement la performance\n"
        "- Risque de dépendance produit si portefeuille trop concentré"
    )
else:
    st.info("Données produits non disponibles.")

# =========================================================
# 5. STOCK ↔ VENTES
# =========================================================
st.markdown("## Stock & impact sur les ventes")

stock_mag = stock.groupby("ID_MAGASIN")["QUANTITE"].sum()
st.bar_chart(stock_mag)

st.markdown(
    "- Des ventes faibles combinées à un faible stock suggèrent une contrainte d’approvisionnement\n"
    "- Un surstock sans hausse de ventes indique une immobilisation de trésorerie"
)

# =========================================================
# 6. VENTES VS ENCAISSEMENTS
# =========================================================
st.markdown("## Ventes vs Encaissements")

enc_mensuel = paiement.groupby("MOIS_NOM", sort=False)["MONTANT"].sum()
df_cash = pd.DataFrame({
    "CA": ca_mensuel,
    "Encaissements": enc_mensuel
}).fillna(0)

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(x=df_cash.index, y=df_cash["CA"], label="CA", ax=ax)
sns.lineplot(x=df_cash.index, y=df_cash["Encaissements"], label="Encaissements", ax=ax)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.set_title("Décalage CA vs Trésorerie")
plt.xticks(rotation=45)
st.pyplot(fig)

taux_enc = df_cash["Encaissements"].sum() / df_cash["CA"].sum() * 100

st.markdown(
    f"- Taux d’encaissement global : **{taux_enc:.1f} %**\n"
    "- Un écart durable indique un risque de trésorerie"
)

# =========================================================
# 7. SYNTHÈSE D’AUDIT
# =========================================================
st.markdown("## Synthèse & diagnostic final")

st.success(
    f"- La performance {annee} est portée par un nombre limité de clients\n"
    f"- Une dépendance client significative est observée (Top 5 = {part_top5:.1f} %)\n"
    f"- Les ventes et encaissements présentent un décalage à surveiller\n"
    f"- Le stock semble influencer directement la capacité de vente\n"
    f"- Des leviers existent : diversification client, pilotage stock, réduction des retards de paiement"
)