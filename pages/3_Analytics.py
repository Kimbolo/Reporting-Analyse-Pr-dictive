import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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

ca_mensuel = facture.groupby("MOIS_NOM", sort=False)["MONTANT_NET"].sum().reset_index()
variation = ca_mensuel["MONTANT_NET"].pct_change() * 100

# Graphique interactif avec Plotly
fig_ca = px.line(
    ca_mensuel, 
    x="MOIS_NOM", 
    y="MONTANT_NET",
    markers=True,
    title="Évolution mensuelle du chiffre d’affaires"
)
fig_ca.update_traces(
    hovertemplate='<b>%{x}</b><br>' +
                  'CA: %{y:,.0f} €<br>' +
                  '<extra></extra>'
)
fig_ca.update_layout(
    xaxis_title="Mois",
    yaxis_title="Chiffre d'affaires (€)",
    hovermode='x unified'
)
st.plotly_chart(fig_ca, use_container_width=True)

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

df_magasin = get_data("SELECT ID_MAGASIN, NOM_MAGASIN FROM magasin")

ca_client = (
    facture_client.groupby("NOM")["MONTANT_NET"]
    .sum()
    .sort_values(ascending=False)
)

top5 = ca_client.head(5)
top10 = ca_client.head(10)

# Graphique interactif avec Plotly
fig_clients = px.bar(
    x=top10.values,
    y=top10.index,
    color_discrete_sequence=["blue"],
    title="Top 10 clients par chiffre d’affaires"
)
fig_clients.update_layout(
    xaxis_title="Chiffre d'affaires (€)",
    yaxis_title="Client",
    hovermode='y unified'
)
st.plotly_chart(fig_clients, use_container_width=True)

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
    
    # Afficher les top produits
    df_produits = pd.DataFrame({
        'Produit': ca_produit.index.astype(str),
        'CA': ca_produit.values
    })
    
    fig_produits = px.bar(
        df_produits,
        x='CA',
        y='Produit',
        orientation='h',
        title="Top 10 produits par chiffre d'affaires",
        color='CA',
        color_continuous_scale='Blues'
    )
    fig_produits.update_traces(
        hovertemplate='<b>%{y}</b><br>CA: %{x:,.0f} €<extra></extra>'
    )
    st.plotly_chart(fig_produits, use_container_width=True)

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

# Fusionner stock avec les noms des magasins
if 'df_magasin' in locals():
    stock_with_names = stock.merge(df_magasin, on="ID_MAGASIN", how="left")
    # Remplacer les ID manquants par "Magasin inconnu"
    stock_with_names["NOM_MAGASIN"] = stock_with_names["NOM_MAGASIN"].fillna(f"Magasin {stock_with_names['ID_MAGASIN']}")
    
    # Grouper par nom de magasin
    stock_mag = stock_with_names.groupby("NOM_MAGASIN")["QUANTITE"].sum().sort_values(ascending=False)
else:
    # Fallback si la table magasin n'existe pas
    stock_mag = stock.groupby("ID_MAGASIN")["QUANTITE"].sum()
    stock_mag.index = [f"Magasin {idx}" for idx in stock_mag.index]

# Créer un graphique Plotly pour meilleur affichage (avec tooltips)
fig_stock = px.bar(
    x=stock_mag.values,
    y=stock_mag.index,
    orientation='h',
    title="Niveau de stock par magasin",
    color=stock_mag.values,
    color_continuous_scale='Reds',
    labels={'x': 'Quantité en stock', 'y': 'Magasin'}
)
fig_stock.update_traces(
    hovertemplate='<b>%{y}</b><br>Stock: %{x:,.0f} unités<extra></extra>'
)
fig_stock.update_layout(
    height=400,
    xaxis_title="Quantité en stock",
    yaxis_title="Magasin"
)
st.plotly_chart(fig_stock, use_container_width=True)

st.markdown(
    "- Des ventes faibles combinées à un faible stock suggèrent une contrainte d’approvisionnement\n"
    "- Un surstock sans hausse de ventes indique une immobilisation de trésorerie"
)

# =========================================================
# 6. VENTES VS ENCAISSEMENTS
# =========================================================
st.markdown("## Ventes vs Encaissements")

# Calculer les encaissements par mois
enc_mensuel = paiement.groupby("MOIS_NOM", sort=False)["MONTANT"].sum().reset_index()
enc_mensuel.columns = ['MOIS_NOM', 'Encaissements']

# S'assurer que ca_mensuel est un DataFrame avec les bonnes colonnes
if isinstance(ca_mensuel, pd.Series):
    ca_mensuel = ca_mensuel.reset_index()
    ca_mensuel.columns = ['MOIS_NOM', 'CA']
elif 'MONTANT_NET' in ca_mensuel.columns:
    ca_mensuel = ca_mensuel[['MOIS_NOM', 'MONTANT_NET']]
    ca_mensuel.columns = ['MOIS_NOM', 'CA']

df_cash = pd.merge(ca_mensuel, enc_mensuel, on='MOIS_NOM', how='outer').fillna(0)

# Vérifier que l'ordre des mois est correct
ordre_mois = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", 
              "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
df_cash['MOIS_NOM'] = pd.Categorical(df_cash['MOIS_NOM'], categories=ordre_mois, ordered=True)
df_cash = df_cash.sort_values('MOIS_NOM')

# Graphique avec deux lignes séparées
fig_cash = go.Figure()

# Ajouter la ligne du CA
fig_cash.add_trace(go.Scatter(
    x=df_cash['MOIS_NOM'],
    y=df_cash['CA'],
    name='Chiffre d\'affaires',
    mode='lines+markers',
    line=dict(color='blue', width=2),
    marker=dict(size=8),
    hovertemplate='<b>%{x}</b><br>CA: %{y:,.0f} €<extra></extra>'
))

# Ajouter la ligne des encaissements
fig_cash.add_trace(go.Scatter(
    x=df_cash['MOIS_NOM'],
    y=df_cash['Encaissements'],
    name='Encaissements',
    mode='lines+markers',
    line=dict(color='green', width=2),
    marker=dict(size=8),
    hovertemplate='<b>%{x}</b><br>Encaissements: %{y:,.0f} €<extra></extra>'
))

fig_cash.update_layout(
    title="Décalage CA vs Trésorerie",
    xaxis_title="Mois",
    yaxis_title="Montant (€)",
    hovermode='x unified',
    legend=dict(x=0, y=1, orientation='h')
)

st.plotly_chart(fig_cash, use_container_width=True)

taux_enc = df_cash["Encaissements"].sum() / df_cash["CA"].sum() * 100 if df_cash["CA"].sum() > 0 else 0

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