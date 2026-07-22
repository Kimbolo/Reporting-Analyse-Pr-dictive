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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error
from db import get_data

# ==========================================================
# STYLE ET CONFIGURATION
# ==========================================================
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

COLORS = {
    'primary': '#1f77b4', 'secondary': '#ff7f0e', 'success': '#2ca02c',
    'danger': '#d62728', 'warning': '#ffbb78', 'purple': '#9467bd',
    'brown': '#8c564b', 'pink': '#e377c2', 'gray': '#7f7f7f',
    'olive': '#bcbd22', 'cyan': '#17becf'
}

PLOTLY_CONFIG = {
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
    'displaylogo': False,
    'responsive': True
}

def format_cfa(valeur):
    if valeur is None or pd.isna(valeur): return "0 FCFA"
    return f"{valeur:,.0f}".replace(",", " ") + " FCFA"

def format_nombre(valeur):
    if valeur is None or pd.isna(valeur): return "0"
    return f"{valeur:,.0f}".replace(",", " ")

def format_percentage(valeur):
    return f"{valeur:+.1f}%"

def create_time_series_chart(data, x_values, y_values, title, y_label, 
                              line_color=COLORS['primary'], line_dash='solid',
                              show_fill=False, fill_color=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, y=y_values, mode='lines+markers', name='Valeurs',
        line=dict(color=line_color, width=3, dash=line_dash),
        marker=dict(size=8, symbol='circle'),
        fill='tozeroy' if show_fill else 'none', fillcolor=fill_color
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title="Période", yaxis_title=y_label,
        hovermode='x unified', template='plotly_white', height=450,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def safe_get_moyennes_par_mois(ca_mensuel_all):
    if ca_mensuel_all is not None and not ca_mensuel_all.empty:
        return ca_mensuel_all.groupby("MOIS")["MONTANT_NET"].mean()
    return pd.Series()

# ==========================================================
# TITRE
# ==========================================================
st.title("Machine Learning")
st.caption("Previsions, detection d'anomalies et insights strategiques")

# ==========================================================
# CHARGEMENT AUTONOME DES DONNEES
# ==========================================================
try:
    facture_all = get_data("SELECT * FROM facture_client")
    if facture_all is not None and not facture_all.empty:
        facture_all["DATE_CREATION"] = pd.to_datetime(facture_all["DATE_CREATION"], errors="coerce")
        facture_all = facture_all.dropna(subset=["DATE_CREATION"])
        facture = facture_all.copy()
    else:
        st.error("Impossible de charger les données des factures")
        st.stop()
    
    stock = get_data("SELECT * FROM stock")
    if stock is None: stock = pd.DataFrame()
    
    df_prod = get_data("SELECT * FROM production")
    if df_prod is None: df_prod = pd.DataFrame()
    
    st.session_state["facture_client"] = facture
    st.session_state["facture_all"] = facture_all
    st.session_state["stock"] = stock
    st.session_state["df_prod"] = df_prod
    st.session_state["annee"] = datetime.now().year
    st.session_state["mois"] = []
    
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {str(e)}")
    st.info("Veuillez vérifier votre connexion à la base de données et que les tables existent")
    st.stop()

facture_clientiltered = st.session_state["facture_client"].copy()
stock = st.session_state["stock"].copy()
df_prod = st.session_state.get("df_prod", pd.DataFrame())
annee_courante = st.session_state.get("annee", 2024)
mois_selectionnes = st.session_state.get("mois", [])

if "facture_all" in st.session_state:
    facture_all = st.session_state["facture_all"].copy()
else:
    facture_all = facture_clientiltered

facture = facture_clientiltered

# ==========================================================
# CONSTANTES
# ==========================================================
MOIS_FR = {1:"Janvier", 2:"Fevrier", 3:"Mars", 4:"Avril", 5:"Mai", 6:"Juin",
           7:"Juillet", 8:"Aout", 9:"Septembre", 10:"Octobre", 11:"Novembre", 12:"Decembre"}

MOIS_ABBR = {1:"Jan", 2:"Fev", 3:"Mar", 4:"Avr", 5:"Mai", 6:"Juin",
             7:"Juil", 8:"Aou", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

# ==========================================================
# PREPARATION DES DONNEES
# ==========================================================
facture["MOIS"] = facture["DATE_CREATION"].dt.month
facture["ANNEE"] = facture["DATE_CREATION"].dt.year
facture_all["MOIS"] = facture_all["DATE_CREATION"].dt.month
facture_all["ANNEE"] = facture_all["DATE_CREATION"].dt.year

ca_mensuel_all = facture_all.groupby(["ANNEE", "MOIS"])["MONTANT_NET"].sum().reset_index()

if "ID_PRODUIT" in facture.columns and "MONTANT_NET" in facture.columns:
    top_produits = facture.groupby("ID_PRODUIT")["MONTANT_NET"].sum().sort_values(ascending=False).head(10)
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
    annee_reference = toutes_annees_disponibles[-1]
    annee_comparee = toutes_annees_disponibles[-2]

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

    if annee_reference > annee_comparee:
        annee_plus_recente = annee_reference
        annee_plus_ancienne = annee_comparee
    else:
        annee_plus_recente = annee_comparee
        annee_plus_ancienne = annee_reference

    annee_1 = annee_reference
    annee_2 = annee_comparee

    ca_annee_1 = ca_mensuel_all[ca_mensuel_all["ANNEE"] == annee_reference]
    ca_annee_2 = ca_mensuel_all[ca_mensuel_all["ANNEE"] == annee_comparee]

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
    
    # Calcul des évolutions
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
    
    mois_labels = [MOIS_ABBR.get(m, m) for m in range(1, 13)]
    
    valeurs_annee_1 = []
    for m in range(1, 13):
        val = ca_annee_1[ca_annee_1["MOIS"] == m]["MONTANT_NET"].values
        valeurs_annee_1.append(val[0] if len(val) > 0 else 0)
    
    valeurs_annee_2 = []
    for m in range(1, 13):
        val = ca_annee_2[ca_annee_2["MOIS"] == m]["MONTANT_NET"].values
        valeurs_annee_2.append(val[0] if len(val) > 0 else 0)
    
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
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
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

# Initialisation des variables pour éviter les erreurs
previsions = None
mae = None
mois_futurs_labels = []

if len(ca_mensuel_all) >= 6:
    
    ca_mensuel_all = ca_mensuel_all.sort_values(["ANNEE", "MOIS"])
    ca_mensuel_all["ORDRE"] = range(len(ca_mensuel_all))
    
    ca_mensuel_all["MOIS_SIN"] = np.sin(2 * np.pi * ca_mensuel_all["MOIS"] / 12)
    ca_mensuel_all["MOIS_COS"] = np.cos(2 * np.pi * ca_mensuel_all["MOIS"] / 12)
    
    # Séparation entraînement/test pour validation réaliste
    train_size = min(len(ca_mensuel_all) - 3, int(len(ca_mensuel_all) * 0.8))
    train_data = ca_mensuel_all.iloc[:train_size]
    test_data = ca_mensuel_all.iloc[train_size:]
    
    X_train = train_data[["ORDRE", "MOIS_SIN", "MOIS_COS"]]
    y_train = train_data["MONTANT_NET"]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Validation sur données de test
    if len(test_data) > 0:
        X_test = test_data[["ORDRE", "MOIS_SIN", "MOIS_COS"]]
        y_test = test_data["MONTANT_NET"]
        y_pred_test = model.predict(X_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    else:
        # Fallback si pas assez de données pour test
        y_pred_train = model.predict(X_train)
        mae_test = mean_absolute_error(y_train, y_pred_train)
        mape_test = mean_absolute_percentage_error(y_train, y_pred_train) * 100
    
    # Prévision pour les 6 prochains mois
    derniere_position = ca_mensuel_all["ORDRE"].max()
    mois_futur = pd.DataFrame({
        "ORDRE": range(derniere_position + 1, derniere_position + 7),
        "MOIS_SIN": np.sin(2 * np.pi * (ca_mensuel_all.iloc[-1]["MOIS"] + np.arange(1, 7)) / 12),
        "MOIS_COS": np.cos(2 * np.pi * (ca_mensuel_all.iloc[-1]["MOIS"] + np.arange(1, 7)) / 12)
    })
    previsions = model.predict(mois_futur)
    
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
    
    mae = mae_test
    mape = mape_test
    
    borne_inf = previsions - mae
    borne_sup = previsions + mae
    
    col_precision1, col_precision2, col_precision3 = st.columns(3)
    with col_precision1:
        st.metric("Précision du modèle", f"±{mae:,.0f} FCFA", help="Erreur absolue moyenne sur données de test")
    with col_precision2:
        st.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error - plus bas = meilleur")
    with col_precision3:
        if mape < 10:
            st.success("Modèle fiable")
        elif mape < 20:
            st.warning("Modèle acceptable")
        else:
            st.error("Modèle peu fiable")
    
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
            st.caption(f"Fourchette: {format_cfa(borne_inf[i])} - {format_cfa(borne_sup[i])}")
    
    historique_labels = [f"{MOIS_ABBR.get(row['MOIS'], row['MOIS'])} {row['ANNEE']}" for _, row in ca_mensuel_all.tail(12).iterrows()]
    historique_vals = ca_mensuel_all.tail(12)["MONTANT_NET"].values
    
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
    
    moyenne_previsions = previsions.mean()
    tendance = "HAUSSIERE" if moyenne_previsions > y_train.mean() else "BAISSIERE"
    
    st.markdown(f"""
    <div style='background-color: {"#d4edda" if moyenne_previsions > y_train.mean() else "#fff3cd"}; 
                padding: 15px; 
                border-radius: 10px;
                margin: 10px 0;'>
        <b>Interpretation :</b> Tendance <b style='color: {"#28a745" if moyenne_previsions > y_train.mean() else "#ffc107"}'>
        {tendance}</b><br>
        Prevision moyenne : {format_cfa(moyenne_previsions)} vs moyenne historique : {format_cfa(y_train.mean())}
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
    
    if "DATE_PRODUCTION" in df_prod.columns:
        df_prod["DATE_PROD"] = pd.to_datetime(df_prod["DATE_PRODUCTION"], errors="coerce")
        df_prod["MOIS_PROD"] = df_prod["DATE_PROD"].dt.month
        df_prod["ANNEE_PROD"] = df_prod["DATE_PROD"].dt.year
        
        prod_mensuelle = df_prod.groupby(["ANNEE_PROD", "MOIS_PROD"])["QUANTITE_TOTALE"].sum().reset_index()
        
        if "PERTES_TOTALES" in df_prod.columns:
            pertes_mensuelle = df_prod.groupby(["ANNEE_PROD", "MOIS_PROD"])["PERTES_TOTALES"].sum().reset_index()
        else:
            pertes_mensuelle = pd.DataFrame()
        
        st.markdown("### Evolution production & pertes")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
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

    if "PERTES_TOTALES" in df_prod.columns and "QUANTITE_TOTALE" in df_prod.columns:
        st.markdown("### Analyse detaillée du taux de perte")

        if "MOIS_PROD" in df_prod.columns and "ANNEE_PROD" in df_prod.columns:
            pertes_par_mois = df_prod.groupby(["ANNEE_PROD", "MOIS_PROD"]).agg({
                "PERTES_TOTALES": "sum",
                "QUANTITE_TOTALE": "sum"
            }).reset_index()

            pertes_par_mois["TAUX_PERTE"] = (pertes_par_mois["PERTES_TOTALES"] / pertes_par_mois["QUANTITE_TOTALE"] * 100).fillna(0)

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

            fig_taux.add_hline(
                y=objectif_taux_perte,
                line_dash="dash",
                line_color=COLORS['success'],
                annotation_text=f"Objectif: {objectif_taux_perte}%",
                annotation_position="top right"
            )

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

            mois_anormaux = pertes_par_mois[pertes_par_mois["TAUX_PERTE"] > seuil_alerte]
            if not mois_anormaux.empty:
                st.warning(f"{len(mois_anormaux)} mois depassent le seuil d'alerte ({seuil_alerte}%)")
                for _, row in mois_anormaux.tail(3).iterrows():
                    st.markdown(f"- {MOIS_FR.get(row['MOIS_PROD'], row['MOIS_PROD'])} {row['ANNEE_PROD']}: **{row['TAUX_PERTE']:.1f}%**")
            else:
                st.success("Tous les mois respectent le seuil d'alerte")

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
            
            type_max = cols_existants[pertes_by_type.argmax()]
            type_name = type_max.replace('PERDE_EN_', '').replace('_', ' ').title()
            st.info(f"Point d'attention : {type_name} represente la plus grande part des pertes - Prioriser l'amelioration sur ce poste")

# ==========================================================
# SECTION : OBJECTIFS CA & PRODUCTION (AJOUT)
# ==========================================================
st.markdown("## Objectifs CA & Production")
st.caption("Définition des objectifs stratégiques et simulation de production")

# Sous-onglets pour cette section
tab_obj_ca, tab_obj_prod = st.tabs(["Objectif de CA", "Simulation de Production"])

with tab_obj_ca:
    st.markdown("### Définition de l'Objectif de Chiffre d'Affaires")
    
    col_obj1, col_obj2 = st.columns(2)
    
    with col_obj1:
        st.markdown("**Saisir l'objectif**")
        
        ca_actuel = facture['MONTANT_NET'].sum() if not facture.empty else 0
        
        mode_objectif = st.radio("Mode de définition", ["Montant cible", "Taux de croissance"], horizontal=True)
        
        if mode_objectif == "Montant cible":
            ca_cible = st.number_input(
                "CA Annuel Cible (FCFA)",
                min_value=1000000, value=max(int(ca_actuel * 1.2), 10000000), step=1000000, format="%d"
            )
            croissance = ((ca_cible / ca_actuel) - 1) * 100 if ca_actuel > 0 else 0
            st.caption(f"Croissance implicite : {croissance:+.1f}%")
        else:
            croissance = st.slider("Taux de croissance souhaité (%)", 0, 200, 20, 5)
            ca_cible = ca_actuel * (1 + croissance / 100) if ca_actuel > 0 else 60000000
        
        # Paramètres de prix
        prix_vente = st.number_input("Prix de vente moyen/bouteille (FCFA)", value=1000, step=100)
        
        if st.button("Calculer l'objectif de production", type="primary", use_container_width=True):
            bouteilles_requises = ca_cible / prix_vente if prix_vente > 0 else 0
            
            st.session_state.obj_ca = ca_cible
            st.session_state.obj_bouteilles = bouteilles_requises
            st.session_state.obj_prix = prix_vente
            
            st.success(f"Objectif calculé : {bouteilles_requises:,.0f} bouteilles/an")
    
    with col_obj2:
        if 'obj_ca' in st.session_state:
            st.markdown("**Résultats**")
            
            st.metric("CA Objectif Annuel", format_cfa(st.session_state.obj_ca))
            st.metric("Bouteilles à Produire", f"{st.session_state.obj_bouteilles:,.0f}/an")
            st.metric("Bouteilles par Mois", f"{st.session_state.obj_bouteilles / 12:,.0f}/mois")
            
            # Matières premières requises
            st.markdown("---")
            st.markdown("**Matières Premières Requises**")
            
            b = st.session_state.obj_bouteilles
            mp_requises = pd.DataFrame([
                {'Matière': 'Sucre', 'Quantité': f"{b * 0.025:,.0f} kg", 'Coût Estimé': format_cfa(b * 0.025 * 450)},
                {'Matière': 'Eau traitée', 'Quantité': f"{b * 0.5:,.0f} L", 'Coût Estimé': format_cfa(b * 0.5 * 5)},
                {'Matière': 'Arômes', 'Quantité': f"{b * 0.002:,.0f} L", 'Coût Estimé': format_cfa(b * 0.002 * 5000)},
                {'Matière': 'Bouteilles PET', 'Quantité': f"{b:,.0f} u", 'Coût Estimé': format_cfa(b * 75)},
                {'Matière': 'Bouchons', 'Quantité': f"{b:,.0f} u", 'Coût Estimé': format_cfa(b * 15)},
                {'Matière': 'Étiquettes', 'Quantité': f"{b:,.0f} u", 'Coût Estimé': format_cfa(b * 25)},
            ])
            
            st.dataframe(mp_requises, use_container_width=True, hide_index=True)

with tab_obj_prod:
    st.markdown("### Simulation de Production")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("**Paramètres de production**")
        
        capa_horaire = st.number_input("Capacité horaire (bouteilles/h)", value=200, step=10)
        heures_jour = st.slider("Heures/jour", 1, 24, 8)
        jours_semaine = st.slider("Jours/semaine", 1, 7, 5)
        nb_equipes = st.selectbox("Nombre d'équipes", [1, 2, 3], index=0)
        
        st.markdown("**Facteurs de risque**")
        risque_mp = st.slider("Pénurie matières premières (%)", 0, 50, 10, 5)
        risque_pannes = st.slider("Pannes machines (%)", 0, 50, 5, 5)
        risque_absent = st.slider("Absentéisme (%)", 0, 30, 5, 5)
    
    with col_s2:
        if st.button("Lancer la simulation", type="primary", use_container_width=True):
            prod_jour = capa_horaire * heures_jour * nb_equipes
            prod_hebdo = prod_jour * jours_semaine
            prod_mensuelle = prod_hebdo * 4.33
            prod_annuelle = prod_mensuelle * 12
            
            facteur_risque = 1 - (risque_mp + risque_pannes + risque_absent) / 100
            prod_ajustee = prod_mensuelle * facteur_risque
            
            st.markdown("**Résultats**")
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric("Production journalière", f"{prod_jour:,.0f} bouteilles")
                st.metric("Production hebdomadaire", f"{prod_hebdo:,.0f} bouteilles")
            with col_r2:
                st.metric("Production mensuelle", f"{prod_mensuelle:,.0f} bouteilles")
                st.metric("Production annuelle", f"{prod_annuelle:,.0f} bouteilles")
            
            st.metric("Production ajustée (après risques)", f"{prod_ajustee:,.0f} bouteilles/mois")
            
            # Comparaison avec l'objectif
            if 'obj_bouteilles' in st.session_state:
                besoin_mensuel = st.session_state.obj_bouteilles / 12
                ecart = prod_ajustee - besoin_mensuel
                
                st.markdown("---")
                st.markdown("**Comparaison avec l'objectif**")
                
                if ecart > 0:
                    st.success(f"Capacité suffisante : +{ecart:,.0f} bouteilles/mois de marge")
                else:
                    st.error(f"Capacité insuffisante : déficit de {abs(ecart):,.0f} bouteilles/mois")
                    st.markdown(f"""
                    **Solutions :**
                    - Passer à {nb_equipes + 1} équipe(s)
                    - Augmenter à {min(heures_jour + 4, 24)} heures/jour
                    - Réduire les facteurs de risque
                    """)

# ==========================================================
# SECTION 7 : ALERTES AUTOMATIQUES
# ==========================================================
st.markdown("---")
st.markdown("## Alertes & recommandations automatiques")

alertes = []

if len(ca_mensuel_all) >= 3 and not ca_mensuel_all.empty:
    ca_recent = ca_mensuel_all.tail(3)["MONTANT_NET"].mean()
    ca_historique = ca_mensuel_all.head(-3)["MONTANT_NET"].mean() if len(ca_mensuel_all) > 6 else ca_mensuel_all["MONTANT_NET"].mean()
    if ca_recent < ca_historique * 0.8:
        alertes.append(("Ventes", f"Baisse significative detectee sur les 3 derniers mois (-{((1 - ca_recent/ca_historique)*100):.0f}%)"))

if not stock.empty and "QUANTITE" in stock.columns and len(stock) > 0:
    seuil_stock = stock["QUANTITE"].quantile(0.15)
    stocks_critiques = stock[stock["QUANTITE"] < seuil_stock]
    if not stocks_critiques.empty:
        alertes.append(("Stock", f"{len(stocks_critiques)} produits en risque de rupture"))

if not df_prod.empty and "PERTES_TOTALES" in df_prod.columns:
    if df_prod["PERTES_TOTALES"].sum() > 0:
        pertes_moyennes = df_prod["PERTES_TOTALES"].mean()
        if pertes_moyennes > 100:
            alertes.append(("Pertes", f"Niveau de pertes anormalement eleve ({pertes_moyennes:.0f} unites moyenne)"))

if not ca_mensuel_all.empty:
    moyennes_par_mois_alerte = safe_get_moyennes_par_mois(ca_mensuel_all)
    if not moyennes_par_mois_alerte.empty:
        dernier_mois_actuel = datetime.now().month
        ca_mois_actuel = ca_mensuel_all[ca_mensuel_all["MOIS"] == dernier_mois_actuel]["MONTANT_NET"].values
        if len(ca_mois_actuel) > 0:
            ca_moyen_mois = moyennes_par_mois_alerte[dernier_mois_actuel] if dernier_mois_actuel in moyennes_par_mois_alerte.index else 0
            if ca_moyen_mois > 0 and ca_mois_actuel[0] < ca_moyen_mois * 0.7:
                alertes.append(("Saisonnalite", f"Performance anormalement faible pour {MOIS_FR.get(dernier_mois_actuel, dernier_mois_actuel)}"))

if alertes:
    for alerte in alertes:
        st.error(f"{alerte[0]} : {alerte[1]}")
else:
    st.success("Aucune alerte critique - Performance stable, tous les indicateurs sont au vert")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption("Machine Learning & Analytics - Powered by Streamlit | Donnees actualisees en temps reel")
