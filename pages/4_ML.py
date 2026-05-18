import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error
from db import get_data

# ==========================================================
# STYLE ET CONFIGURATION
# ==========================================================
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Palette de couleurs professionnelle
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

# Configuration Plotly pour un rendu homogene
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
    'displaylogo': False,
    'responsive': True
}

# ==========================================================
# TITRE
# ==========================================================
st.title("Machine Learning")
st.caption("Previsions, detection d'anomalies et insights strategiques")

# ==========================================================
# SECURITE DES DONNEES PARTAGEES
# ==========================================================
required_keys = ["facture_f", "stock", "df_prod", "annee", "mois"]
if not all(k in st.session_state for k in required_keys):
    st.warning("Veuillez d'abord charger les donnees dans l'onglet App.")
    st.stop()

# Récupérer les données filtrées de l'onglet principal
facture_filtered = st.session_state["facture_f"].copy()
stock = st.session_state["stock"].copy()
df_prod = st.session_state.get("df_prod", pd.DataFrame())
annee_courante = st.session_state.get("annee", 2024)
mois_selectionnes = st.session_state.get("mois", [])

# IMPORTANT: Récupérer TOUTES les factures (non filtrées) pour l'analyse YoY
if "facture_all" in st.session_state:
    facture_all = st.session_state["facture_all"].copy()
else:
    # Fallback: utiliser les données filtrées si indisponibles
    facture_all = facture_filtered
    st.warning("Données complètes non disponibles - Utilisation des données filtrées uniquement")

# Pour l'affichage, on garde le filtre actuel
facture = facture_filtered

# Conversion des dates pour facture_all également
facture_all["DATE_CREATION"] = pd.to_datetime(facture_all["DATE_CREATION"], errors="coerce")
facture_all = facture_all.dropna(subset=["DATE_CREATION"])

# ==========================================================
# CONSTANTES
# ==========================================================
MOIS_FR = {
    1: "Janvier", 2: "Fevrier", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Aout",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Decembre"
}

MOIS_ABBR = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Avr",
    5: "Mai", 6: "Juin", 7: "Juil", 8: "Aou",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

def format_cfa(valeur):
    if valeur is None or pd.isna(valeur):
        return "0 FCFA"
    return f"{valeur:,.0f}".replace(",", " ") + " FCFA"

def format_percentage(valeur):
    return f"{valeur:+.1f}%"

# ==========================================================
# PREPARATION DES DONNEES POUR ANALYSES
# ==========================================================

# Creation d'une colonne mois pour les aggregations
facture["MOIS"] = facture["DATE_CREATION"].dt.month
facture["ANNEE"] = facture["DATE_CREATION"].dt.year
facture["TRIMESTRE"] = facture["DATE_CREATION"].dt.quarter
facture["JOUR_SEM"] = facture["DATE_CREATION"].dt.dayofweek

facture_all["MOIS"] = facture_all["DATE_CREATION"].dt.month
facture_all["ANNEE"] = facture_all["DATE_CREATION"].dt.year
facture_all["TRIMESTRE"] = facture_all["DATE_CREATION"].dt.quarter
facture_all["JOUR_SEM"] = facture_all["DATE_CREATION"].dt.dayofweek

# CA mensuel par annee en utilisant TOUTES les factures (facture_all)
ca_mensuel_all = (
    facture_all
    .groupby(["ANNEE", "MOIS"])["MONTANT_NET"]
    .sum()
    .reset_index()
)

# Top produits
if "ID_PRODUIT" in facture.columns and "MONTANT_NET" in facture.columns:
    top_produits = (
        facture
        .groupby("ID_PRODUIT")["MONTANT_NET"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
else:
    top_produits = pd.Series()

# ==========================================================
# SECTION 1 : COMPARAISON YoY (Year over Year)
# ==========================================================

st.markdown("---")
st.markdown("## Analyse comparative YoY (Year over Year)")

# Récupérer toutes les années disponibles dans les données
toutes_annees_disponibles = sorted(facture_all["DATE_CREATION"].dt.year.dropna().unique())

if len(toutes_annees_disponibles) >= 2:
    annee_reference = toutes_annees_disponibles[-1]  # Par défaut, on prend l'année la plus récente comme référence
    annee_comparee = toutes_annees_disponibles[-2]  # Et l'année précédente pour la comparaison

    with st.expander("Choisir les années à comparer"):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            annee_1_adv = st.selectbox(
                "Année de référence", 
                toutes_annees_disponibles, 
                index=len(toutes_annees_disponibles)-1,
                key="yoy_ref_adv",
            )
        with col_adv2:
            autres_annees = [a for a in toutes_annees_disponibles if a != annee_1_adv]
            annee_2_adv = st.selectbox(
                    "Année à comparer", 
                    autres_annees, 
                    index=0,
                    key="yoy_comp_adv",
                )
            
        if annee_1_adv != annee_reference or annee_2_adv != annee_comparee :
            annee_reference = annee_1_adv
            annee_comparee = annee_2_adv
            st.info(f"Mode avancé actif : comparaison {annee_reference} vs {annee_comparee}")
        else:
            st.info(f"Mode standard : comparaison {annee_reference} vs {annee_comparee}")


# Déterminer quelle année est la plus récente pour l'affichage
    if annee_reference > annee_comparee:
        annee_plus_recente = annee_reference
        annee_plus_ancienne = annee_comparee
    else:
        annee_plus_recente = annee_comparee
        annee_plus_ancienne = annee_reference

    annee_1 = annee_reference
    annee_2 = annee_comparee

# ==========================================================
# FILTRAGE DES DONNEES POUR LES 2 ANNEES
# ==========================================================

ca_annee_1 = ca_mensuel_all[ca_mensuel_all["ANNEE"] == annee_reference]
ca_annee_2 = ca_mensuel_all[ca_mensuel_all["ANNEE"] == annee_comparee]

# ==========================================================
# AFFICHAGE COMPARAISON
# ==========================================================

if not facture.empty and len(toutes_annees_disponibles) >= 2:
    
    # Calcul des métriques pour l'année 1
    facture_annee_1 = facture[facture["DATE_CREATION"].dt.year == annee_1]
    ca_1 = facture_annee_1["MONTANT_NET"].sum()
    nb_factures_1 = facture_annee_1["ID_FACTURE_CLIENT"].nunique()
    panier_1 = ca_1 / nb_factures_1 if nb_factures_1 > 0 else 0
    
    # Calcul des métriques pour l'année 2
    facture_annee_2 = facture[facture["DATE_CREATION"].dt.year == annee_2]
    ca_2 = facture_annee_2["MONTANT_NET"].sum()
    nb_factures_2 = facture_annee_2["ID_FACTURE_CLIENT"].nunique()
    panier_2 = ca_2 / nb_factures_2 if nb_factures_2 > 0 else 0
    
    # Calcul des évolutions (depuis l'année la plus ancienne vers la plus récente)
    if annee_plus_recente == annee_1:
        evolution_ca = ((ca_1 - ca_2) / ca_2 * 100) if ca_2 > 0 else 0
        evolution_factures = ((nb_factures_1 - nb_factures_2) / nb_factures_2 * 100) if nb_factures_2 > 0 else 0
        evolution_panier = ((panier_1 - panier_2) / panier_2 * 100) if panier_2 > 0 else 0
        annee_ref = annee_2
        annee_comp = annee_1
        ca_ref = ca_2
        ca_comp = ca_1
        nb_ref = nb_factures_2
        panier_ref = panier_2
    else:
        evolution_ca = ((ca_2 - ca_1) / ca_1 * 100) if ca_1 > 0 else 0
        evolution_factures = ((nb_factures_2 - nb_factures_1) / nb_factures_1 * 100) if nb_factures_1 > 0 else 0
        evolution_panier = ((panier_2 - panier_1) / panier_1 * 100) if panier_1 > 0 else 0
        annee_ref = annee_1
        annee_comp = annee_2
        ca_ref = ca_1
        ca_comp = ca_2
        nb_ref = nb_factures_1
        panier_ref = panier_1
    
    # AFFICHAGE DES KPI
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"CA {annee_comp}", 
            format_cfa(ca_comp),
            delta=f"{evolution_ca:+.1f}% vs {annee_ref}",
            delta_color="normal" if evolution_ca >= 0 else "inverse"
        )
    
    with col2:
        st.metric(
            f"Factures {annee_comp}",
            f"{nb_factures_2 if annee_comp == annee_2 else nb_factures_1:,}",
            delta=f"{evolution_factures:+.1f}% vs {annee_ref}",
            delta_color="normal" if evolution_factures >= 0 else "inverse"
        )
    
    with col3:
        st.metric(
            f"Panier moyen {annee_comp}",
            format_cfa(panier_2 if annee_comp == annee_2 else panier_1),
            delta=f"{evolution_panier:+.1f}% vs {annee_ref}",
            delta_color="normal" if evolution_panier >= 0 else "inverse"
        )
    
    # GRAPHIQUE COMPARATIF
    st.markdown("### Evolution mensuelle comparative")
    
    # Préparer les valeurs pour les 12 mois
    mois_labels = [MOIS_ABBR.get(m, m) for m in range(1, 13)]
    
    # Valeurs année 1
    valeurs_annee_1 = []
    for m in range(1, 13):
        val = ca_annee_1[ca_annee_1["MOIS"] == m]["MONTANT_NET"].values
        valeurs_annee_1.append(val[0] if len(val) > 0 else 0)
    
    # Valeurs année 2
    valeurs_annee_2 = []
    for m in range(1, 13):
        val = ca_annee_2[ca_annee_2["MOIS"] == m]["MONTANT_NET"].values
        valeurs_annee_2.append(val[0] if len(val) > 0 else 0)
    
    # Création du graphique
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=mois_labels,
        y=valeurs_annee_1,
        mode='lines+markers',
        name=f'{annee_1}',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=mois_labels,
        y=valeurs_annee_2,
        mode='lines+markers',
        name=f'{annee_2}',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8, symbol='square')
    ))
    
    # Ajout des écarts significatifs
    for i, (val1, val2) in enumerate(zip(valeurs_annee_1, valeurs_annee_2)):
        if val1 > val2 and val2 > 0:
            fig.add_annotation(
                x=mois_labels[i], y=max(val1, val2),
                text=f"▲ +{((val1-val2)/val2*100):.0f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor='#2ca02c',
                font=dict(size=9, color='#2ca02c')
            )
        elif val2 > val1 and val1 > 0:
            fig.add_annotation(
                x=mois_labels[i], y=max(val1, val2),
                text=f"▲ +{((val2-val1)/val1*100):.0f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor='#2ca02c',
                font=dict(size=9, color='#2ca02c')
            )
    
    fig.update_layout(
        title=f"Comparaison du CA mensuel - {annee_1} vs {annee_2}",
        xaxis_title="Mois",
        yaxis_title="Chiffre d'affaires (FCFA)",
        hovermode='x unified',
        template='plotly_white',
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interprétation
    st.markdown("Synthèse de la comparaison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        meilleur_mois_1 = max(zip(mois_labels, valeurs_annee_1), key=lambda x: x[1])
        meilleur_mois_2 = max(zip(mois_labels, valeurs_annee_2), key=lambda x: x[1])
        st.info(f"**{annee_1}** : Meilleur mois = {meilleur_mois_1[0]} ({format_cfa(meilleur_mois_1[1])})")
        st.info(f"**{annee_2}** : Meilleur mois = {meilleur_mois_2[0]} ({format_cfa(meilleur_mois_2[1])})")
    
    with col2:
        total_1 = sum(valeurs_annee_1)
        total_2 = sum(valeurs_annee_2)
        difference_abs = total_2 - total_1
        difference_rel = ((total_2 - total_1) / total_1 * 100) if total_1 > 0 else 0
        
        if difference_abs > 0:
            st.success(f"**{annee_2}** est en hausse de {format_cfa(difference_abs)} (+{difference_rel:.1f}%) par rapport à {annee_1}")
        elif difference_abs < 0:
            st.error(f"**{annee_2}** est en baisse de {format_cfa(abs(difference_abs))} ({difference_rel:.1f}%) par rapport à {annee_1}")
        else:
            st.info("Les deux années ont un chiffre d'affaires équivalent")

else:
    if len(toutes_annees_disponibles) < 2:
        st.warning(f"Données insuffisantes pour la comparaison YoY. Seule {len(toutes_annees_disponibles)} année(s) disponible(s) : {toutes_annees_disponibles}")
    else:
        st.info(f"Aucune donnée disponible")

# ==========================================================
# SECTION 2 : PREVISIONS POUR LES MOIS A VENIR
# ==========================================================
st.markdown("---")
st.markdown("## Previsions pour les mois a venir")

if len(ca_mensuel_all) >= 6:
    
    # Preparation des donnees pour la regression
    ca_mensuel_all = ca_mensuel_all.sort_values(["ANNEE", "MOIS"])
    ca_mensuel_all["ORDRE"] = range(len(ca_mensuel_all))
    
    # Ajout des features cycliques
    ca_mensuel_all["MOIS_SIN"] = np.sin(2 * np.pi * ca_mensuel_all["MOIS"] / 12)
    ca_mensuel_all["MOIS_COS"] = np.cos(2 * np.pi * ca_mensuel_all["MOIS"] / 12)
    
    X = ca_mensuel_all[["ORDRE", "MOIS_SIN", "MOIS_COS"]]
    y = ca_mensuel_all["MONTANT_NET"]
    
    # Modele de regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Prevision pour les 6 prochains mois
    derniere_position = ca_mensuel_all["ORDRE"].max()
    mois_futur = pd.DataFrame({
        "ORDRE": range(derniere_position + 1, derniere_position + 7),
        "MOIS_SIN": np.sin(2 * np.pi * (ca_mensuel_all.iloc[-1]["MOIS"] + np.arange(1, 7)) / 12),
        "MOIS_COS": np.cos(2 * np.pi * (ca_mensuel_all.iloc[-1]["MOIS"] + np.arange(1, 7)) / 12)
    })
    previsions = model.predict(mois_futur)
    
    # Creation des labels pour les mois futurs
    dernier_mois = ca_mensuel_all.iloc[-1]["MOIS"]
    derniere_annee = ca_mensuel_all.iloc[-1]["ANNEE"]
    
    mois_futurs_labels = []
    for i in range(6):
        mois_fut = dernier_mois + i + 1
        annee_fut = derniere_annee
        if mois_fut > 12:
            mois_fut -= 12
            annee_fut += 1
        mois_futurs_labels.append(f"{MOIS_ABBR.get(mois_fut, mois_fut)} {annee_fut}")
    
    # Calcul de l'erreur historique du modèle (validation croisée simple)
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

    y_pred_train = model.predict(X)
    mae = mean_absolute_error(y, y_pred_train)
    mape = mean_absolute_percentage_error(y, y_pred_train) * 100

    borne_inf = previsions - mae
    borne_sup = previsions + mae

    # Affichage des métriques de qualité du modèle
    col_precision1, col_precision2, col_precision3 = st.columns(3)
    with col_precision1:
        st.metric("Précision du modèle", f"±{mae:,.0f} FCFA", help="Erreur absolue moyenne sur l'historique")
    with col_precision2:
        st.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error - plus bas = meilleur")
    with col_precision3:
        if mape < 10:
            st.success("Modèle fiable")
        elif mape < 20:
            st.warning("Modèle acceptable")
        else:
            st.error("Modèle peu fiable")

    # Affichage des previsions en metriques
    st.markdown("### Previsions mensuelles avec intervalle de confiance")
    
    cols = st.columns(6)
    for i, (col, mois_label, prev) in enumerate(zip(cols, mois_futurs_labels, previsions)):
        with col:
            st.metric(
                mois_label,
                format_cfa(prev),
                delta=f"±{mae:,.0f} FCFA",
                delta_color="off"
            )
            st.caption(f"Fourchette: {format_cfa(borne_inf[i])} - {format_cfa(borne_sup[i])} FCFA")

            
    # Graphique historique + previsions avec Plotly
    st.markdown("### Evolution historique et previsionnelle")
    
    # Preparer les donnees historiques (12 derniers mois)
    historique_labels = [f"{MOIS_ABBR.get(row['MOIS'], row['MOIS'])} {row['ANNEE']}" for _, row in ca_mensuel_all.tail(12).iterrows()]
    historique_vals = ca_mensuel_all.tail(12)["MONTANT_NET"].values
    
    # Creation du graphique
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historique_labels,
        y=historique_vals,
        mode='lines+markers',
        name='Historique',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>Historique</b><br>Periode: %{x}<br>CA: %{y:,.0f} FCFA<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=mois_futurs_labels,
        y=previsions,
        mode='lines+markers',
        name='Previsions',
        line=dict(color=COLORS['secondary'], width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='<b>Prevision</b><br>Periode: %{x}<br>CA estime: %{y:,.0f} FCFA<extra></extra>'
    ))
    
    # Ajout d'une zone de confiance simple
    std_dev = y.std()
    fig.add_trace(go.Scatter(
        x=mois_futurs_labels + mois_futurs_labels[::-1],
        y=list(borne_sup) + list(borne_inf[::-1]),
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255, 127, 14, 0)'),
        name=f'Zone de confiance (±{mae:,.0f} FCFA)',
        showlegend=True,
        hoverinfo='none'
    ))
    
    fig.update_layout(
        title=dict(text="Prevision du chiffre d'affaires - 6 mois", x=0.5, font=dict(size=16)),
        xaxis_title="Periode",
        yaxis_title="Chiffre d'affaires (FCFA)",
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
        yaxis=dict(tickformat=',.0f', tickprefix='', ticksuffix=' FCFA')
    )
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # Interpretation
    moyenne_previsions = previsions.mean()
    tendance = "HAUSSIERE" if moyenne_previsions > y.mean() else "BAISSIERE"
    
    st.markdown(f"""
    <div style='background-color: {"#d4edda" if moyenne_previsions > y.mean() else "#fff3cd"}; 
                padding: 15px; 
                border-radius: 10px;
                margin: 10px 0;'>
        <b>Interpretation :</b> Tendance <b style='color: {"#28a745" if moyenne_previsions > y.mean() else "#ffc107"}'>
        {tendance}</b><br>
        Prevision moyenne : {format_cfa(moyenne_previsions)} vs moyenne historique : {format_cfa(y.mean())}
    </div>
    """, unsafe_allow_html=True)
    
else:
    st.warning("Donnees historiques insuffisantes pour les previsions (minimum 6 mois requis).")

st.markdown("---")

# ==========================================================
# SECTION 3 : TOP PRODUITS & RECOMMANDATIONS
# ==========================================================
if not top_produits.empty:
    st.markdown("## Top produits & recommandations")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Graphique avec Plotly
        top_df = pd.DataFrame({
            'Produit': [f"Produit {i}" for i in top_produits.index],
            'CA': top_produits.values
        }).head(10)
        
        fig = go.Figure(go.Bar(
            x=top_df['CA'],
            y=top_df['Produit'],
            orientation='h',
            marker=dict(
                color=top_df['CA'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="CA (FCFA)")
            ),
            hovertemplate='<b>%{y}</b><br>CA: %{x:,.0f} FCFA<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="Top 10 des produits", x=0.5),
            xaxis_title="Chiffre d'affaires (FCFA)",
            yaxis_title="Produit",
            height=400,
            template='plotly_white',
            margin=dict(l=100, r=20, t=50, b=20),
            xaxis=dict(tickformat=',.0f', tickprefix='', ticksuffix=' FCFA')
        )
        
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    with col2:
        part_top3 = top_produits.head(3).sum() / top_produits.sum() * 100 if top_produits.sum() > 0 else 0
        part_top5 = top_produits.head(5).sum() / top_produits.sum() * 100 if top_produits.sum() > 0 else 0
        
        st.markdown("### Recommandations")
        
        if part_top5 > 60:
            st.warning(f"Forte concentration : Top 5 = {part_top5:.0f}% du CA")
            st.markdown("-> Action : Diversifier l'offre produits")
        else:
            st.success(f"Bonne repartition : Top 5 = {part_top5:.0f}% du CA")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Marketing**")
            st.caption("Mettre en avant les produits a forte marge")
        with col_b:
            st.markdown("**Stock**")
            st.caption("Assurer la disponibilite des meilleures ventes")

# ==========================================================
# SECTION 4 : PRODUCTION & PERTES
# ==========================================================
if not df_prod.empty and "QUANTITE_TOTALE" in df_prod.columns:
    st.markdown("## Analyse Production & Pertes")
    
    # Donnees de production par mois
    if "DATE_PRODUCTION" in df_prod.columns:
        df_prod["DATE_PROD"] = pd.to_datetime(df_prod["DATE_PRODUCTION"], errors="coerce")
        df_prod["MOIS_PROD"] = df_prod["DATE_PROD"].dt.month
        df_prod["ANNEE_PROD"] = df_prod["DATE_PROD"].dt.year
        
        # Production mensuelle
        prod_mensuelle = df_prod.groupby(["ANNEE_PROD", "MOIS_PROD"])["QUANTITE_TOTALE"].sum().reset_index()
        
        # Verifier si PERTES_TOTALES existe
        if "PERTES_TOTALES" in df_prod.columns:
            pertes_mensuelle = df_prod.groupby(["ANNEE_PROD", "MOIS_PROD"])["PERTES_TOTALES"].sum().reset_index()
        else:
            pertes_mensuelle = pd.DataFrame()
        
        # Graphique Production vs Pertes avec Plotly
        st.markdown("### Evolution production & pertes")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Production
        prod_recent = prod_mensuelle.tail(12)
        labels = [f"{MOIS_ABBR.get(row['MOIS_PROD'], row['MOIS_PROD'])} {row['ANNEE_PROD']}" for _, row in prod_recent.iterrows()]
        
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=prod_recent["QUANTITE_TOTALE"],
                mode='lines+markers',
                name='Production',
                line=dict(color=COLORS['primary'], width=3),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='<b>Production</b><br>Periode: %{x}<br>Quantite: %{y:,.0f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Pertes (si disponibles)
        if not pertes_mensuelle.empty:
            pertes_recent = pertes_mensuelle.tail(12)
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=pertes_recent["PERTES_TOTALES"],
                    mode='lines+markers',
                    name='Pertes',
                    line=dict(color=COLORS['danger'], width=3, dash='dot'),
                    marker=dict(size=8, symbol='square'),
                    hovertemplate='<b>Pertes</b><br>Periode: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
                ),
                secondary_y=True
            )
        
        fig.update_layout(
            title=dict(text="Production vs Pertes", x=0.5),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white',
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        fig.update_yaxes(title_text="Quantite produite", secondary_y=False)
        fig.update_yaxes(title_text="Volume de pertes", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Metriques de performance
        total_prod = df_prod["QUANTITE_TOTALE"].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Production totale", f"{total_prod:,.0f}")
        
        if "PERTES_TOTALES" in df_prod.columns:
            total_pertes = df_prod["PERTES_TOTALES"].sum()
            taux_perte = (total_pertes / total_prod * 100) if total_prod > 0 else 0
            
            with col2:
                st.metric("Pertes totales", f"{total_pertes:,.0f}")
            with col3:
                delta_color = "inverse" if taux_perte > 5 else "normal"
                st.metric("Taux de perte", f"{taux_perte:.1f}%", delta="Objectif < 5%", delta_color=delta_color)
            
            if taux_perte > 5:
                st.warning("Alerte : Taux de perte eleve - Investir dans la maintenance preventive")
            else:
                st.success("Performance : Taux de perte maitrise")


    # Taux de perte détaillé avec objectif
    if "PERTES_TOTALES" in df_prod.columns and "QUANTITE_TOTALE" in df_prod.columns:
        st.markdown("### Analyse detaillée du taux de perte")

        # Calculer le taux de perte par mois
        if "MOIS_PROD" in df_prod.columns and "ANNEE_PROD" in df_prod.columns:
            pertes_par_mois = df_prod.groupby(["ANNEE_PROD", "MOIS_PROD"]).agg({
                "PERTES_TOTALES": "sum",
                "QUANTITE_TOTALE": "sum"
            }).reset_index()

            pertes_par_mois["TAUX_PERTE"] = (pertes_par_mois["PERTES_TOTALES"] / pertes_par_mois["QUANTITE_TOTALE"] * 100).fillna(0)

            # Objectif paramétrable
            col_obj1, col_obj2 = st.columns(2)
            with col_obj1:
                objectif_taux_perte = st.number_input(
                    "Objectif taux de perte (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    help="Seuil cible pour le taux de perte. Au-delà, une alerte sera déclenchée.",
                    key="taux_perte_objectif")
                
            with col_obj2:
                taux_moyen = pertes_par_mois["TAUX_PERTE"].mean()
                delta_taux = taux_moyen - objectif_taux_perte
                st.metric(
                    "Taux de perte moyen",
                    f"{taux_moyen:.1f}%",
                    delta=f"{delta_taux:+.1f}% vs objectif",
                    delta_color="inverse" if delta_taux > 0 else "normal"
                )

            # Graphique d'évolution du taux de perte
            fig_taux = go.Figure()

            labels_taux = [f"{MOIS_ABBR.get(row['MOIS_PROD'], row['MOIS_PROD'])} {row['ANNEE_PROD']}" for _, row in pertes_par_mois.tail(12).iterrows()]
            valeurs_taux = pertes_par_mois.tail(12)["TAUX_PERTE"].values

            fig_taux.add_trace(go.Scatter(
                x=labels_taux,
                y=valeurs_taux,
                mode='lines+markers',
                name='Taux de perte',
                line=dict(color=COLORS['danger'], width=3),
                marker=dict(size=8, symbol='circle'),
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.1)',
                hovertemplate='<b>%{x}</b><br>Taux de perte: %{y:.1f}%<extra></extra>'
            ))

            # Ligne d'objectif
            fig_taux.add_hline(
                y=objectif_taux_perte,
                line_dash="dash",
                line_color=COLORS['success'],
                annotation_text=f"Objectif: {objectif_taux_perte}%",
                annotation_position="top right"
            )

            # Ligne d'alerte (objectif + 2%)
            seuil_alerte = objectif_taux_perte + 2
            fig_taux.add_hline(
                y=seuil_alerte,
                line_dash="dot",
                line_color=COLORS['warning'],
                annotation_text=f"Alerte: {seuil_alerte}%",
                annotation_position="bottom right"
            )

            fig_taux.update_layout(
                title=dict(text="Evolution du taux de perte", x=0.5),
                xaxis_title="Période",
                yaxis_title="Taux de perte (%)",
                template='plotly_white',
                height=350,
                yaxis=dict(range=[0, max(valeurs_taux.max(), seuil_alerte) + 2])
            )

            st.plotly_chart(fig_taux, use_container_width=True, config=PLOTLY_CONFIG)

            # Détection des mois avec taux anormal
            mois_anormaux = pertes_par_mois[pertes_par_mois["TAUX_PERTE"] > seuil_alerte]
            if not mois_anormaux.empty:
                st.warning(f"{len(mois_anormaux)} mois depassent le seuil d'alerte ({seuil_alerte}%)")
                for _, row in mois_anormaux.tail(3).iterrows():
                    st.markdown(f"- {MOIS_FR.get(row['MOIS_PROD'], row['MOIS_PROD'])} {row['ANNEE_PROD']}: **{row['TAUX_PERTE']:.1f}%**")
            else :
                st.success("Tous les mois respectent le seuil d'alerte")

        # Analyse des pertes par type
        cols_pertes = ["PERDE_EN_BOUTEILLE", "PERDE_EN_CAPSULE", "PERDE_EN_ETIQUETTE", "PERDE_EN_CARTON", "QUANTITE_AVARIE"]
        cols_existants = [c for c in cols_pertes if c in df_prod.columns]
        
        if cols_existants:
            st.markdown("### Repartition des pertes")
            
            pertes_by_type = df_prod[cols_existants].sum()
            pertes_df = pd.DataFrame({
                'Type': [c.replace('PERDE_EN_', '').replace('_', ' ').title() for c in cols_existants],
                'Volume': pertes_by_type.values
            })
            
            fig = px.pie(
                pertes_df, 
                values='Volume', 
                names='Type',
                title='Repartition des pertes par type',
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.3
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Volume: %{value:,.0f}<br>Part: %{percent}<extra></extra>'
            )
            
            fig.update_layout(
                template='plotly_white',
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Recommandation sur le principal type de perte
            type_max = cols_existants[pertes_by_type.argmax()]
            type_name = type_max.replace('PERDE_EN_', '').replace('_', ' ').title()
            st.info(f"Point d'attention : {type_name} represente la plus grande part des pertes - Prioriser l'amelioration sur ce poste")

st.markdown("---")

# ==========================================================
# SECTION 5 : SAISONNALITE
# ==========================================================
if not ca_mensuel_all.empty:
    st.markdown("## Analyse de saisonnalite")
    
    # Calcul des moyennes par mois sur toutes les annees
    moyennes_par_mois = ca_mensuel_all.groupby("MOIS")["MONTANT_NET"].mean()
    mois_labels = [MOIS_FR.get(m, m) for m in range(1, 13)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=mois_labels,
        y=moyennes_par_mois.values,
        marker=dict(
            color=moyennes_par_mois.values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="CA moyen (FCFA)")
        ),
        hovertemplate='<b>%{x}</b><br>CA moyen: %{y:,.0f} FCFA<extra></extra>'
    ))
    
    # Ajout de la ligne de moyenne
    moyenne_globale = moyennes_par_mois.mean()
    fig.add_hline(
        y=moyenne_globale,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Moyenne: {format_cfa(moyenne_globale)}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=dict(text="Saisonnalite du chiffre d'affaires", x=0.5, font=dict(size=16)),
        xaxis_title="Mois",
        yaxis_title="Chiffre d'affaires moyen (FCFA)",
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        yaxis=dict(tickformat=',.0f', tickprefix='', ticksuffix=' FCFA')
    )
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # Identifier les pics et creux
    mois_pic = moyennes_par_mois.idxmax()
    mois_creux = moyennes_par_mois.idxmin()
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Pic d'activite : {MOIS_FR.get(mois_pic, mois_pic)}")
        st.caption("-> Anticiper les stocks et renforcer les equipes")
    with col2:
        st.warning(f"Creux d'activite : {MOIS_FR.get(mois_creux, mois_creux)}")
        st.caption("-> Planifier la maintenance et les formations")

st.markdown("---")

# ==========================================================
# SECTION 5.5 : PREVISIONS DES PERTES (modèle dédié)
# ==========================================================
st.markdown("---")
st.markdown("## Prévisions des pertes")

if not df_prod.empty and "PERTES_TOTALES" in df_prod.columns and "QUANTITE_TOTALE" in df_prod.columns:
    
    # Préparer les données de pertes par mois
    if "DATE_PRODUCTION" in df_prod.columns:
        df_prod["DATE_PROD"] = pd.to_datetime(df_prod["DATE_PRODUCTION"], errors="coerce")
        df_prod["MOIS_PROD"] = df_prod["DATE_PROD"].dt.month
        df_prod["ANNEE_PROD"] = df_prod["DATE_PROD"].dt.year
        
        # Agréger les pertes par mois
        pertes_mensuelles = df_prod.groupby(["ANNEE_PROD", "MOIS_PROD"]).agg({
            "PERTES_TOTALES": "sum",
            "QUANTITE_TOTALE": "sum"
        }).reset_index()
        
        pertes_mensuelles["TAUX_PERTE"] = (pertes_mensuelles["PERTES_TOTALES"] / pertes_mensuelles["QUANTITE_TOTALE"] * 100).fillna(0)
        
        if len(pertes_mensuelles) >= 6:
            
            # Créer un ordre temporel
            pertes_mensuelles = pertes_mensuelles.sort_values(["ANNEE_PROD", "MOIS_PROD"])
            pertes_mensuelles["ORDRE"] = range(len(pertes_mensuelles))
            
            # Features cycliques pour les pertes
            pertes_mensuelles["MOIS_SIN"] = np.sin(2 * np.pi * pertes_mensuelles["MOIS_PROD"] / 12)
            pertes_mensuelles["MOIS_COS"] = np.cos(2 * np.pi * pertes_mensuelles["MOIS_PROD"] / 12)
            
            # Modèle pour volume de pertes
            X_loss = pertes_mensuelles[["ORDRE", "MOIS_SIN", "MOIS_COS"]]
            y_loss = pertes_mensuelles["PERTES_TOTALES"]
            
            model_loss = LinearRegression()
            model_loss.fit(X_loss, y_loss)
            
            # Prévision des pertes sur 6 mois
            derniere_position_loss = pertes_mensuelles["ORDRE"].max()
            dernier_mois_loss = pertes_mensuelles.iloc[-1]["MOIS_PROD"]
            derniere_annee_loss = pertes_mensuelles.iloc[-1]["ANNEE_PROD"]
            
            mois_futur_loss = pd.DataFrame({
                "ORDRE": range(derniere_position_loss + 1, derniere_position_loss + 7),
                "MOIS_SIN": np.sin(2 * np.pi * (dernier_mois_loss + np.arange(1, 7)) / 12),
                "MOIS_COS": np.cos(2 * np.pi * (dernier_mois_loss + np.arange(1, 7)) / 12)
            })
            
            previsions_pertes = model_loss.predict(mois_futur_loss)
            previsions_pertes = np.maximum(previsions_pertes, 0)  # Pas de pertes négatives
            
            # Calcul de l'erreur du modèle
            y_pred_loss = model_loss.predict(X_loss)
            mae_loss = mean_absolute_error(y_loss, y_pred_loss)
            
            # Labels pour les mois futurs
            mois_futurs_loss_labels = []
            for i in range(6):
                mois_fut = dernier_mois_loss + i + 1
                annee_fut = derniere_annee_loss
                if mois_fut > 12:
                    mois_fut -= 12
                    annee_fut += 1
                mois_futurs_loss_labels.append(f"{MOIS_ABBR.get(mois_fut, mois_fut)} {annee_fut}")
            
            # Affichage des prévisions
            st.markdown("### Prévisions des pertes - 6 mois")
            
            col_loss1, col_loss2, col_loss3 = st.columns(3)
            with col_loss1:
                st.metric("Précision du modèle pertes", f"±{mae_loss:,.0f} unités")
            with col_loss2:
                pertes_moyennes_historiques = y_loss.mean()
                st.metric("Pertes moyennes historiques", f"{pertes_moyennes_historiques:,.0f} unités")
            with col_loss3:
                tendance_pertes = "HAUSSIÈRE" if previsions_pertes[-1] > previsions_pertes[0] else "BAISSIÈRE"
                st.metric("Tendance prévue", tendance_pertes)
            
            # Graphique historique + prévision des pertes
            fig_loss = go.Figure()
            
            # Labels historiques (12 derniers mois)
            historiques_loss_labels = [f"{MOIS_ABBR.get(row['MOIS_PROD'], row['MOIS_PROD'])} {row['ANNEE_PROD']}" 
                                       for _, row in pertes_mensuelles.tail(12).iterrows()]
            historiques_loss_vals = pertes_mensuelles.tail(12)["PERTES_TOTALES"].values
            
            fig_loss.add_trace(go.Scatter(
                x=historiques_loss_labels,
                y=historiques_loss_vals,
                mode='lines+markers',
                name='Pertes historiques',
                line=dict(color=COLORS['danger'], width=3),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='<b>Historique</b><br>Periode: %{x}<br>Pertes: %{y:,.0f}<extra></extra>'
            ))
            
            fig_loss.add_trace(go.Scatter(
                x=mois_futurs_loss_labels,
                y=previsions_pertes,
                mode='lines+markers',
                name='Prévisions pertes',
                line=dict(color=COLORS['warning'], width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='<b>Prévision</b><br>Periode: %{x}<br>Pertes estimées: %{y:,.0f}<extra></extra>'
            ))
            
            # Zone de confiance
            fig_loss.add_trace(go.Scatter(
                x=mois_futurs_loss_labels + mois_futurs_loss_labels[::-1],
                y=list(previsions_pertes + mae_loss) + list((previsions_pertes - mae_loss)[::-1]),
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255, 127, 14, 0)'),
                name=f'Zone de confiance (±{mae_loss:,.0f})',
                showlegend=True
            ))
            
            fig_loss.update_layout(
                title=dict(text="Prévision des pertes - 6 mois", x=0.5, font=dict(size=16)),
                xaxis_title="Période",
                yaxis_title="Volume de pertes",
                hovermode='x unified',
                template='plotly_white',
                height=400,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            st.plotly_chart(fig_loss, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Interprétation et recommandations
            ratio_perte_ca = 0
            if "MONTANT_NET" in facture.columns:
                ca_moyen_mensuel = facture.groupby(facture["DATE_CREATION"].dt.to_period("M"))["MONTANT_NET"].sum().mean()
                valeur_moyenne_par_perte = ca_moyen_mensuel / max(y_loss.mean(), 1)
                cout_prevu_pertes = previsions_pertes.mean() * valeur_moyenne_par_perte
                
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                    <b>Impact business estimé</b><br>
                    - Pertes moyennes prévues : <b>{previsions_pertes.mean():.0f} unités/mois</b><br>
                    - Coût estimé des pertes : <b>{format_cfa(cout_prevu_pertes)}/mois</b><br>
                    - Recommandation : {"Urgence - Action corrective nécessaire" if previsions_pertes.mean() > pertes_moyennes_historiques * 1.2 else "🟢 Situation sous contrôle"}
                </div>
                """, unsafe_allow_html=True)
            
            # Alerte si prévision anormalement haute
            if previsions_pertes.mean() > pertes_moyennes_historiques * 1.3:
                st.error("ALERTE : Hausse significative des pertes prévue dans les mois à venir !")
            elif previsions_pertes.mean() > pertes_moyennes_historiques * 1.1:
                st.warning("Attention : Légère hausse des pertes prévue à surveiller")
            else:
                st.success("Prévision des pertes stable")
        else:
            st.warning("Données insuffisantes pour la prévision des pertes (minimum 6 mois requis)")
    else:
        st.info("Colonne DATE_PRODUCTION manquante pour l'analyse temporelle des pertes")
else:
    st.info("Données de production/pertes insuffisantes pour la prévision")

# ==========================================================
# SECTION 6 : MODELES MACHINE LEARNING
# ==========================================================
st.markdown("## Modeles Machine Learning avances")

cas_ml = st.radio(
    "Choisissez un cas d'usage",
    [
        "Prevision des ventes",
        "Risque de rupture de stock",
        "Detection d'anomalies",
        "ML Utilisateurs"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

# ==========================================================
# 6.1 PREVISION DES VENTES DETAILLEE
# ==========================================================
if cas_ml == "Prevision des ventes (detaillee)":
    st.markdown("### Prevision des ventes - Modele avance")
    
    # Preparation des donnees avec features cycliques
    ca_mensuel = (
        facture
        .groupby(["ANNEE", "MOIS"])["MONTANT_NET"]
        .sum()
        .reset_index()
    )
    
    if len(ca_mensuel) >= 6:
        st.markdown("#### Donnees historiques (12 derniers mois)")
        
        # Afficher les donnees historiques formatees
        ca_mensuel_display = ca_mensuel.tail(12).copy()
        ca_mensuel_display["CA"] = ca_mensuel_display["MONTANT_NET"].apply(format_cfa)
        ca_mensuel_display["Periode"] = ca_mensuel_display.apply(
            lambda x: f"{MOIS_FR.get(x['MOIS'], x['MOIS'])} {x['ANNEE']}", axis=1
        )
        st.dataframe(ca_mensuel_display[["Periode", "CA"]], use_container_width=True, hide_index=True)
        
        # Modele avec features cycliques
        ca_mensuel["MOIS_SIN"] = np.sin(2 * np.pi * ca_mensuel["MOIS"] / 12)
        ca_mensuel["MOIS_COS"] = np.cos(2 * np.pi * ca_mensuel["MOIS"] / 12)
        
        X = ca_mensuel[["MOIS", "MOIS_SIN", "MOIS_COS"]]
        y = ca_mensuel["MONTANT_NET"]
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Prevision pour 24 mois
        mois_futur = pd.DataFrame({
            "MOIS": list(range(1, 13)) * 2,
            "MOIS_SIN": np.sin(2 * np.pi * np.arange(1, 25) / 12),
            "MOIS_COS": np.cos(2 * np.pi * np.arange(1, 25) / 12)
        })
        ca_prevu = model.predict(mois_futur)
        
        # Graphique avec Plotly
        historique_x = [f"{MOIS_ABBR.get(row['MOIS'], row['MOIS'])} {row['ANNEE']}" for _, row in ca_mensuel.iterrows()]
        
        # Labels pour les previsions
        annee_fin = ca_mensuel.iloc[-1]["ANNEE"]
        dernier_mois = ca_mensuel.iloc[-1]["MOIS"]
        
        prevision_x = []
        for i in range(24):
            mois = dernier_mois + i + 1
            annee = annee_fin
            if mois > 12:
                mois -= 12
                annee += 1
            prevision_x.append(f"{MOIS_ABBR.get(mois, mois)} {annee}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historique_x,
            y=ca_mensuel["MONTANT_NET"],
            mode='lines+markers',
            name='Historique',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=6, symbol='circle'),
            hovertemplate='<b>Historique</b><br>Periode: %{x}<br>CA: %{y:,.0f} FCFA<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=prevision_x,
            y=ca_prevu,
            mode='lines+markers',
            name='Prevision',
            line=dict(color=COLORS['secondary'], width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate='<b>Prevision</b><br>Periode: %{x}<br>CA estime: %{y:,.0f} FCFA<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="Prevision du chiffre d'affaires - 24 mois", x=0.5, font=dict(size=16)),
            xaxis_title="Periode",
            yaxis_title="Chiffre d'affaires (FCFA)",
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white',
            height=450,
            margin=dict(l=50, r=50, t=80, b=100),
            xaxis=dict(tickangle=45),
            yaxis=dict(tickformat=',.0f', tickprefix='', ticksuffix=' FCFA')
        )
        
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Interpretation
        derniere_valeur = ca_mensuel["MONTANT_NET"].iloc[-1]
        premiere_prevision = ca_prevu[0]
        
        if premiere_prevision > derniere_valeur:
            st.success("Tendance HAUSSIERE detectee - Recommandation : Augmenter les capacites de production et renforcer les stocks")
        else:
            st.warning("Tendance BAISSIERE detectee - Recommandation : Optimiser les stocks et la tresorerie, actions marketing renforcees")
    else:
        st.warning("Donnees insuffisantes pour la prevision (minimum 6 mois d'historique requis)")

# ==========================================================
# 6.2 RISQUE DE RUPTURE DE STOCK
# ==========================================================
elif cas_ml == "Risque de rupture de stock":
    st.markdown("### Analyse du risque de rupture de stock")
    
    if not stock.empty and "QUANTITE" in stock.columns and len(stock) > 0:
        seuil = stock["QUANTITE"].quantile(0.2)
        stock["RISQUE_RUPTURE"] = stock["QUANTITE"] < seuil
        stock["NIVEAU_RISQUE"] = pd.cut(
            stock["QUANTITE"],
            bins=[-float('inf'), seuil/2, seuil, float('inf')],
            labels=["CRITIQUE", "ELEVE", "OK"]
        )
        
        # Graphique de distribution
        fig = go.Figure()
        
        risque_counts = stock["NIVEAU_RISQUE"].value_counts()
        colors_risque = {"CRITIQUE": COLORS['danger'], "ELEVE": COLORS['warning'], "OK": COLORS['success']}
        
        fig.add_trace(go.Bar(
            x=risque_counts.index,
            y=risque_counts.values,
            marker_color=[colors_risque.get(x, COLORS['gray']) for x in risque_counts.index],
            hovertemplate='<b>%{x}</b><br>Nombre: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="Distribution du risque stock", x=0.5),
            xaxis_title="Niveau de risque",
            yaxis_title="Nombre de produits",
            template='plotly_white',
            height=350,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Affichage des produits a risque
        produits_risque = stock[stock["RISQUE_RUPTURE"]].copy()
        
        if not produits_risque.empty:
            st.warning(f"{len(produits_risque)} produits en risque de rupture")
            
            # Ajout d'une colonne de priorisation
            produits_risque["PRIORITE"] = pd.cut(
                produits_risque["QUANTITE"],
                bins=[-float('inf'), seuil/4, seuil/2, seuil],
                labels=["URGENT", "HAUTE", "MOYENNE"]
            )
            
            cols_affichage = ["DESIGNATION", "QUANTITE", "PRIORITE"] if "DESIGNATION" in produits_risque.columns else ["QUANTITE", "PRIORITE"]
            st.dataframe(produits_risque[cols_affichage], use_container_width=True)
            
            if len(produits_risque) > len(stock) * 0.3:
                st.error("URGENCE : Plus de 30% du stock est critique !")
        else:
            st.success("Niveau de stock satisfaisant - Aucun risque de rupture detecte")
    else:
        st.info("Donnees de stock insuffisantes pour l'analyse")

# ==========================================================
# 6.3 DETECTION D'ANOMALIES
# ==========================================================
elif cas_ml == "Detection d'anomalies (pertes)":
    st.markdown("### Detection d'anomalies - Pertes & avaries")
    
    # Chercher les colonnes de pertes
    perte_cols = []
    
    if not df_prod.empty:
        perte_cols = [c for c in df_prod.columns if "PERDE" in c.upper() or "AVARIE" in c.upper()]
    
    if not perte_cols and not facture.empty:
        perte_cols = [c for c in facture.columns if "PERD" in c.upper() or "AVARIE" in c.upper()]
    
    if not perte_cols:
        st.info("Aucune donnee de pertes disponible pour l'analyse.")
    else:
        if not df_prod.empty and perte_cols:
            pertes = df_prod[perte_cols].fillna(0)
            df_to_use = df_prod.copy()
        else:
            pertes = facture[perte_cols].fillna(0)
            df_to_use = facture.copy()
        
        model = IsolationForest(contamination=0.1, random_state=42)
        df_to_use["ANOMALIE"] = model.fit_predict(pertes)
        
        anomalies = df_to_use[df_to_use["ANOMALIE"] == -1]
        
        # Metriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total lignes analysees", len(df_to_use))
        with col2:
            st.metric("Anomalies detectees", len(anomalies))
        with col3:
            taux_anomalies = len(anomalies) / len(df_to_use) * 100 if len(df_to_use) > 0 else 0
            st.metric("Taux d'anomalies", f"{taux_anomalies:.1f}%")
        
        if not anomalies.empty:
            st.warning(f"{len(anomalies)} anomalies detectees - A analyser en priorite")
            
            # Affichage des anomalies
            cols_affichage = perte_cols + (["DATE_PRODUCTION"] if "DATE_PRODUCTION" in df_to_use.columns else [])
            cols_affichage = [c for c in cols_affichage if c in anomalies.columns]
            st.dataframe(anomalies[cols_affichage].head(20), use_container_width=True)
            
            if len(anomalies) > len(df_to_use) * 0.15:
                st.error("Alerte critique : Taux d'anomalies eleve - Audit qualite urgent necessaire")
        else:
            st.success("Aucune anomalie detectee - Processus de production stable")

# ==========================================================
# 6.4 ML UTILISATEURS
# ==========================================================
elif cas_ml == "ML Utilisateurs (admin)":
    st.markdown("### Classification utilisateurs")
    
    try:
        from db import get_data
        df = get_data("SELECT * FROM utilisateur")
        
        if df.empty:
            st.info("Aucune donnee utilisateur disponible")
        else:
            cols_dispo = [c for c in ["ID_PROFIL", "EMPLOYE"] if c in df.columns]
            if len(cols_dispo) >= 2:
                df = df[cols_dispo].dropna()
                
                X = df[[cols_dispo[0]]]
                y = df[cols_dispo[1]]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                
                acc = accuracy_score(y_test, model.predict(X_test))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy du modele", f"{acc:.2%}")
                with col2:
                    st.metric("Taille dataset", len(df))
                
                df["prediction_employe"] = model.predict(X)
                st.dataframe(df, use_container_width=True)
                
                st.info(
                    "Ce modele est un cas administratif illustratif, "
                    "sans impact direct sur les decisions business."
                )
            else:
                st.warning("Colonnes necessaires manquantes dans la table utilisateur")
    except ImportError:
        st.warning("Module db.py non disponible - Fonctionnalite desactivee")
    except Exception as e:
        st.error(f"Erreur : {str(e)}")

st.markdown("---")

# ==========================================================
# SECTION 7 : ALERTES AUTOMATIQUES
# ==========================================================
st.markdown("## Alertes & recommandations automatiques")

alertes = []

# Alerte ventes
if len(ca_mensuel_all) >= 3 and not ca_mensuel_all.empty:
    ca_recent = ca_mensuel_all.tail(3)["MONTANT_NET"].mean()
    ca_historique = ca_mensuel_all.head(-3)["MONTANT_NET"].mean() if len(ca_mensuel_all) > 6 else ca_mensuel_all["MONTANT_NET"].mean()
    
    if ca_recent < ca_historique * 0.8:
        alertes.append(("Ventes", f"Baisse significative detectee sur les 3 derniers mois (-{((1 - ca_recent/ca_historique)*100):.0f}%)"))

# Alerte stock
if not stock.empty and "QUANTITE" in stock.columns and len(stock) > 0:
    seuil_stock = stock["QUANTITE"].quantile(0.15)
    stocks_critiques = stock[stock["QUANTITE"] < seuil_stock]
    if not stocks_critiques.empty:
        alertes.append(("Stock", f"{len(stocks_critiques)} produits en risque de rupture"))

# Alerte pertes
if not df_prod.empty and "PERTES_TOTALES" in df_prod.columns:
    if df_prod["PERTES_TOTALES"].sum() > 0:
        pertes_moyennes = df_prod["PERTES_TOTALES"].mean()
        if pertes_moyennes > 100:
            alertes.append(("Pertes", f"Niveau de pertes anormalement eleve ({pertes_moyennes:.0f} unites moyenne)"))

# Alerte saisonniere
if not ca_mensuel_all.empty and 'moyennes_par_mois' in locals():
    dernier_mois_actuel = datetime.now().month
    ca_mois_actuel = ca_mensuel_all[ca_mensuel_all["MOIS"] == dernier_mois_actuel]["MONTANT_NET"].values
    if len(ca_mois_actuel) > 0:
        ca_moyen_mois = moyennes_par_mois[dernier_mois_actuel] if dernier_mois_actuel in moyennes_par_mois.index else 0
        if ca_moyen_mois > 0 and ca_mois_actuel[0] < ca_moyen_mois * 0.7:
            alertes.append(("Saisonnalite", f"Performance anormalement faible pour {MOIS_FR.get(dernier_mois_actuel, dernier_mois_actuel)}"))

if alertes:
    for alerte in alertes:
        st.error(f"{alerte[0]} : {alerte[1]}")
else:
    st.success("Aucune alerte critique - Performance stable, tous les indicateurs sont au vert")

# ==========================================================
# EXPORT DES PREVISIONS (version robuste)
# ==========================================================

# Téléchargement des prévisions
if 'previsions' in locals() and len(previsions) > 0:
    previsions_df = pd.DataFrame({
        "Periode": mois_futurs_labels if 'mois_futurs_labels' in locals() else [f"M+{i+1}" for i in range(len(previsions))],
        "Prevision_CA_CFA": previsions,
        "Borne_Inferieure": previsions - mae if 'mae' in locals() else previsions * 0.9,
        "Borne_Superieure": previsions + mae if 'mae' in locals() else previsions * 1.1
    })

    csv_data = previsions_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Exporter les prévisions CA (CSV)",
        data=csv_data,
        file_name=f"previsions_ca_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

if 'previsions_pertes' in locals() and len(previsions_pertes) > 0:
    previsions_pertes_df = pd.DataFrame({
        "Periode": mois_futurs_loss_labels if 'mois_futurs_loss_labels' in locals() else [f"M+{i+1}" for i in range(len(previsions_pertes))],
        "Prevision_Pertes_Unites": previsions_pertes,
        "Borne_Inferieure": previsions_pertes - mae_loss if 'mae_loss' in locals() else previsions_pertes * 0.9,
        "Borne_Superieure": previsions_pertes + mae_loss if 'mae_loss' in locals() else previsions_pertes * 1.1
    })

    csv_data_pertes = previsions_pertes_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Exporter les prévisions pertes (CSV)",
        data=csv_data_pertes,
        file_name=f"previsions_pertes_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption("Machine Learning & Analytics - Powered by Streamlit | Donnees actualisees en temps reel")