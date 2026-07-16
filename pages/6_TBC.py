# ==========================================================
# PAGE TBC - TO BE CONFIRMED
# Tableau de Bord Complet - Prévisions, Géolocalisation, Approvisionnements
# ==========================================================

from matplotlib.artist import get
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from db import get_data
import warnings

warnings.filterwarnings('ignore')

# ==========================================================
# CONFIGURATION & STYLE
# ==========================================================

st.set_page_config(
    page_title="TBC - To Be Confirmed",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

MOIS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}

MOIS_ABBR = {
    1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr",
    5: "Mai", 6: "Juin", 7: "Juil", 8: "Aoû",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc"
}

SAISONS = {
    'Saison des pluies': [4, 5, 6, 7, 8, 9, 10],
    'Saison sèche': [11, 12, 1, 2, 3]
}

def get_saison(mois):
    """Détermine la saison en fonction du mois"""
    for saison, mois_liste in SAISONS.items():
        if mois in mois_liste:
            return saison
    return 'Saison sèche'

# ==========================================================
# CHARGEMENT DES DONNÉES
# ==========================================================

@st.cache_data(ttl=300, show_spinner="Chargement des données...")
def load_tbc_data():
    """Charge toutes les données nécessaires pour la page TBC"""
    try:
        data = {
            'facture': get_data("SELECT * FROM facture_client"),
            'stock': get_data("SELECT * FROM stock"),
            'sortie': get_data("SELECT * FROM sortie"),
            'production': get_data("SELECT * FROM production"),
            'fournisseur': get_data("SELECT * FROM fournisseur"),
            'personne': get_data("SELECT * FROM personne"),
            'magasin': get_data("SELECT * FROM magasin"),
            'produit': get_data("SELECT * FROM produit"),
        }
        
        # Vérification des données critiques
        if data['facture'] is None or data['facture'].empty:
            st.error("Impossible de charger les données des factures")
            return None
        
        # Conversion des dates pour les factures
        if 'DATE_CREATION' in data['facture'].columns:
            data['facture']['DATE_CREATION'] = pd.to_datetime(data['facture']['DATE_CREATION'], errors='coerce')
            data['facture']['ANNEE'] = data['facture']['DATE_CREATION'].dt.year
            data['facture']['MOIS'] = data['facture']['DATE_CREATION'].dt.month
            data['facture']['MOIS_NOM'] = data['facture']['MOIS'].map(MOIS_FR)
            data['facture']['SAISON'] = data['facture']['MOIS'].apply(get_saison)
            data['facture']['TRIMESTRE'] = data['facture']['DATE_CREATION'].dt.quarter
        
        # Conversion des dates pour les sorties
        if data['sortie'] is not None and not data['sortie'].empty:
            if 'DATE_SORTIE' in data['sortie'].columns:
                data['sortie']['DATE_SORTIE'] = pd.to_datetime(data['sortie']['DATE_SORTIE'], errors='coerce')
                data['sortie']['ANNEE'] = data['sortie']['DATE_SORTIE'].dt.year
                data['sortie']['MOIS'] = data['sortie']['DATE_SORTIE'].dt.month
        
        # Fallbacks pour les tables optionnelles
        for table in ['fournisseur', 'production']:
            if data[table] is None:
                data[table] = pd.DataFrame()
        
        return data
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return None

# ==========================================================
# FONCTIONS UTILITAIRES
# ==========================================================

def format_cfa(valeur):
    """Formatage des montants en FCFA"""
    if valeur is None or pd.isna(valeur):
        return "0 FCFA"
    return f"{valeur:,.0f}".replace(",", " ") + " FCFA"

# ==========================================================
# TITRE
# ==========================================================

# Titre principal (visible seulement si authentifié)
st.title("N'NAM Jus - Tableau de Bord Direction")

# Chargement des données
data = load_tbc_data()
if data is None:
    st.stop()

# ==========================================================
# SIDEBAR - FILTRES
# ==========================================================

with st.sidebar:
    st.markdown("## Filtres d'analyse")
    
    st.markdown("---")
    
    # Période
    if not data['facture'].empty and 'ANNEE' in data['facture'].columns:
        annees_dispo = sorted(data['facture']['ANNEE'].dropna().unique(), reverse=True)
        selected_annee = st.selectbox("Année", annees_dispo, index=0 if annees_dispo else 0)
    else:
        selected_annee = datetime.now().year
    
    # Magasins
    if not data['magasin'].empty:
        magasins = data['magasin']['NOM_MAGASIN'].tolist()
        selected_magasins = st.multiselect("Magasins", magasins, default=[])
    else:
        selected_magasins = []
    
    # Fournisseurs
    if not data['fournisseur'].empty and 'NOM_FOURNISSEUR' in data['fournisseur'].columns:
        fournisseurs = data['fournisseur']['NOM_FOURNISSEUR'].tolist()
        selected_fournisseurs = st.multiselect("Fournisseurs", fournisseurs, default=[])
    else:
        selected_fournisseurs = []

    st.markdown("---")
    if not data['stock'].empty:
        st.caption(f"{len(data['stock'])} produits en stock")
    if not data['facture'].empty:
        st.caption(f" {len(data['facture'])} factures chargées")
    if not data['personne'].empty:
        st.caption(f" {len(data['personne'])} personnes")

# ==========================================================
# APPLICATION DES FILTRES
# ==========================================================

# Filtrer les factures
facture_filtered = data['facture'].copy()
if selected_annee in annees_dispo:
    facture_filtered = facture_filtered[facture_filtered['ANNEE'] == selected_annee]

# ==========================================================
# KPI GLOBAUX
# ==========================================================

st.markdown("## Indicateurs clés de performance")

# Calcul des KPI (basés sur les clients, pas les factures)
ca_total = facture_filtered['MONTANT_NET'].sum() if not facture_filtered.empty and 'MONTANT_NET' in facture_filtered.columns else 0

# Nombre de clients uniques (remplace nb_factures)
nb_clients = facture_filtered['ID_PERSONNE'].nunique() if 'ID_PERSONNE' in facture_filtered.columns else 0

# Nombre de commandes (un client = une commande, peu importe le nombre de factures)
nb_commandes = facture_filtered.groupby('ID_PERSONNE')['MONTANT_NET'].count().sum() if 'ID_PERSONNE' in facture_filtered.columns and 'MONTANT_NET' in facture_filtered.columns else 0

# Panier moyen par CLIENT (CA total / nombre de clients)
panier_moyen_client = ca_total / nb_clients if nb_clients > 0 else 0

# Montant moyen par commande
montant_moyen_commande = ca_total / nb_commandes if nb_commandes > 0 else 0

# Stock total
stock_total = data['stock']['QUANTITE'].sum() if not data['stock'].empty and 'QUANTITE' in data['stock'].columns else 0

# Pertes totales
pertes_total = data['production']['PERTES_TOTALES'].sum() if not data['production'].empty and 'PERTES_TOTALES' in data['production'].columns else 0

# Taux de perte
if not data['production'].empty and 'QUANTITE_TOTALE' in data['production'].columns and 'PERTES_TOTALES' in data['production'].columns:
    production_totale = data['production']['QUANTITE_TOTALE'].sum()
    pertes_total = data['production']['PERTES_TOTALES'].sum()
    taux_perte = (pertes_total / production_totale * 100) if production_totale > 0 else 0
else:
    taux_perte = 0

# Calcul du coefficient de saisonnalité actuel
mois_actuel_num = datetime.now().month
if not facture_filtered.empty and 'MOIS' in facture_filtered.columns and 'MONTANT_NET' in facture_filtered.columns:
    ca_par_mois_kpi = facture_filtered.groupby('MOIS')['MONTANT_NET'].sum()
    ca_moyen_mensuel_kpi = ca_par_mois_kpi.mean()
    coeff_saison_actuel = (ca_par_mois_kpi.get(mois_actuel_num, 0) / ca_moyen_mensuel_kpi) if ca_moyen_mensuel_kpi > 0 else 1.0
else:
    coeff_saison_actuel = 1.0

col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

with col1:
    st.metric("C.A Total", format_cfa(ca_total))

with col2:
    st.metric("Clients actifs", f"{nb_clients:,}")

with col3:
    st.metric("Commandes", f"{nb_commandes:,}")

with col4:
    st.metric("CA/Client", format_cfa(panier_moyen_client))

with col5:
    st.metric("CA/Commande", format_cfa(montant_moyen_commande))

with col6:
    st.metric("Stock total", f"{stock_total:,.0f}")

with col7:
    delta_color = "inverse" if taux_perte > 5 else "normal"
    st.metric("Taux de perte", f"{taux_perte:.1f}%", delta="Objectif < 5%", delta_color=delta_color)

with col8:
    coeff_delta = f"{(coeff_saison_actuel - 1) * 100:+.0f}% vs moyenne"
    coeff_color = "normal" if coeff_saison_actuel >= 1 else "inverse"
    st.metric(
        "Coeff. Saison",
        f"{coeff_saison_actuel:.2f}",
        delta=coeff_delta,
        delta_color=coeff_color,
        help="Coefficient de saisonnalité du mois en cours. > 1 = fort potentiel, < 1 = faible potentiel."
    )

st.markdown("---")

# ==========================================================
# DÉFINITION DES ONGLETS (FILTRÉS SELON LE PROFIL)
# ==========================================================

tab1, tab_objectifs, tab_prod_ventes, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab_direction = st.tabs([
    "Prévisions ML",
    "Objectifs CA & Production",
    "Production vs Ventes",
    "Corrélation Saisons",
    "Cartographie",
    "Magasin matières premières",
    "Relance Clients",
    "Nouveaux Conditionnements",
    "Analyse Produits & Fruits",
    "Stocks & Magasins",
    "Dictionnaire des données",
    "Simulation Direction"
])

# ==========================================================
# TAB 1 : PRÉVISIONS ML
# ==========================================================

with tab1:
        st.markdown("## Prévisions & Ruptures de stock")
    
if not facture_filtered.empty and len(facture_filtered) >= 6 and 'MONTANT_NET' in facture_filtered.columns:
        # Préparation des données mensuelles
        ca_mensuel = facture_filtered.groupby(['ANNEE', 'MOIS'])['MONTANT_NET'].sum().reset_index()
        ca_mensuel = ca_mensuel.sort_values(['ANNEE', 'MOIS'])
        
        if len(ca_mensuel) >= 6:
            # Création des features
            ca_mensuel['ORDRE'] = range(len(ca_mensuel))
            ca_mensuel['MOIS_SIN'] = np.sin(2 * np.pi * ca_mensuel['MOIS'] / 12)
            ca_mensuel['MOIS_COS'] = np.cos(2 * np.pi * ca_mensuel['MOIS'] / 12)
            
            # Séparation train/test
            train_size = max(int(len(ca_mensuel) * 0.8), len(ca_mensuel) - 3)
            train_data = ca_mensuel.iloc[:train_size]
            test_data = ca_mensuel.iloc[train_size:]
            
            # Modèle Random Forest
            X_train = train_data[['ORDRE', 'MOIS_SIN', 'MOIS_COS']]
            y_train = train_data['MONTANT_NET']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Évaluation
            if len(test_data) > 0:
                X_test = test_data[['ORDRE', 'MOIS_SIN', 'MOIS_COS']]
                y_test = test_data['MONTANT_NET']
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                r2 = r2_score(y_test, y_pred)
            else:
                y_pred_train = model.predict(X_train)
                mae = mean_absolute_error(y_train, y_pred_train)
                mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100
                r2 = r2_score(y_train, y_pred_train)
            
            # Affichage des métriques du modèle
            st.markdown("### Performance du modèle")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Erreur absolue moyenne", format_cfa(mae))
            with col2:
                st.metric("MAPE", f"{mape:.1f}%")
            with col3:
                st.metric("R²", f"{r2:.2%}")
            
            # Prévisions 6 mois
            st.markdown("### Prévisions des 6 prochains mois")
            
            dernier_mois = ca_mensuel.iloc[-1]['MOIS']
            derniere_annee = ca_mensuel.iloc[-1]['ANNEE']
            derniere_position = ca_mensuel['ORDRE'].max()
            
            mois_futurs = []
            for i in range(6):
                mois = dernier_mois + i + 1
                annee = derniere_annee
                if mois > 12:
                    mois -= 12
                    annee += 1
                mois_futurs.append((annee, mois))
            
            X_future = pd.DataFrame({
                'ORDRE': range(derniere_position + 1, derniere_position + 7),
                'MOIS_SIN': [np.sin(2 * np.pi * m / 12) for _, m in mois_futurs],
                'MOIS_COS': [np.cos(2 * np.pi * m / 12) for _, m in mois_futurs]
            })
            
            previsions = model.predict(X_future)
            previsions = np.maximum(previsions, 0)  # Pas de valeurs négatives
            
            # ============================================
            # AJUSTEMENT PAR COEFFICIENT DE SAISONNALITÉ
            # ============================================
            # Calculer les coefficients de saisonnalité historiques
            ca_par_mois_ml = ca_mensuel.groupby('MOIS')['MONTANT_NET'].mean()
            ca_moyen_ml = ca_par_mois_ml.mean()
            coeff_saison_ml = {m: (ca_par_mois_ml.get(m, ca_moyen_ml) / ca_moyen_ml) if ca_moyen_ml > 0 else 1.0 
                              for m in range(1, 13)}
            
            # Ajuster les prévisions avec le coefficient de saisonnalité
            previsions_ajustees = []
            for i, (annee, mois) in enumerate(mois_futurs):
                coeff = coeff_saison_ml.get(mois, 1.0)
                prev_ajustee = previsions[i] * coeff
                previsions_ajustees.append(prev_ajustee)
            previsions_ajustees = np.array(previsions_ajustees)

            # Affichage mensuel avec coefficient de saisonnalité
            st.markdown("#### Prévisions ajustées avec coefficient de saisonnalité")
            
            cols = st.columns(6)
            for i, ((annee, mois), prev, prev_aj) in enumerate(zip(mois_futurs, previsions, previsions_ajustees)):
                coeff_mois = coeff_saison_ml.get(mois, 1.0)
                with cols[i]:
                    potentiel = "🟢 Fort" if coeff_mois > 1.1 else ("🟡 Moyen" if coeff_mois >= 0.9 else "🔴 Faible")
                    st.metric(
                        f"{MOIS_ABBR[mois]} {annee}",
                        format_cfa(prev_aj),
                        delta=f"Coef: {coeff_mois:.2f} ({potentiel})",
                        delta_color="normal" if coeff_mois >= 1 else "inverse",
                        help=f"Prévision brute: {format_cfa(prev)} | Coefficient saisonnier: {coeff_mois:.2f}"
                    )
            
            # Graphique historique + prévisions
            st.markdown("### Évolution et prévisions")
            
            historique_labels = [f"{MOIS_ABBR[row['MOIS']]} {row['ANNEE']}" for _, row in ca_mensuel.tail(12).iterrows()]
            historique_vals = ca_mensuel.tail(12)['MONTANT_NET'].values
            futur_labels = [f"{MOIS_ABBR[m]} {a}" for a, m in mois_futurs]
            
            fig = go.Figure()
            
            # Historique
            fig.add_trace(go.Scatter(
                x=historique_labels,
                y=historique_vals,
                mode='lines+markers',
                name='Historique',
                line=dict(color=COLORS['primary'], width=3),
                marker=dict(size=8, symbol='circle')
            ))
            
            # Prévisions brutes
            fig.add_trace(go.Scatter(
                x=futur_labels,
                y=previsions,
                mode='lines+markers',
                name='Prévisions brutes',
                line=dict(color='rgba(255, 127, 14, 0.5)', width=2, dash='dot'),
                marker=dict(size=6, symbol='diamond-open'),
                hovertemplate='<b>%{x}</b><br>Prévision brute: %{y:,.0f} FCFA<extra></extra>'
            ))
            
            # Prévisions ajustées (saisonnalité)
            fig.add_trace(go.Scatter(
                x=futur_labels,
                y=previsions_ajustees,
                mode='lines+markers',
                name='Prévisions ajustées (saison)',
                line=dict(color=COLORS['secondary'], width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='<b>%{x}</b><br>Prévision ajustée: %{y:,.0f} FCFA<extra></extra>'
            ))
            
            # Zone de confiance
            fig.add_trace(go.Scatter(
                x=list(futur_labels) + list(futur_labels[::-1]),
                y=list(previsions_ajustees + mae) + list((previsions_ajustees - mae)[::-1]),
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255, 127, 14, 0)'),
                name=f'Zone de confiance (±{mae:,.0f})',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f"Prévision du chiffre d'affaires - {len(historique_labels)} mois historiques + 6 mois prévus",
                xaxis_title="Période",
                yaxis_title="CA (FCFA)",
                hovermode='x unified',
                template='plotly_white',
                height=450,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tendance
            moyenne_historique = y_train.mean()
            moyenne_previsions = previsions.mean()
            moyenne_previsions_ajustees = previsions_ajustees.mean()
            tendance = "HAUSSIÈRE" if moyenne_previsions_ajustees > moyenne_historique else "BAISSIÈRE"
            
            st.info(f"""
            **Tendance : {tendance}** | 
            Prévision moyenne brute : {format_cfa(moyenne_previsions)} | 
            Prévision ajustée (saison) : {format_cfa(moyenne_previsions_ajustees)} | 
            Historique : {format_cfa(moyenne_historique)}
            """)

            # ============================================
            # RISQUE DE RUPTURE DE STOCK
            # ============================================
            st.markdown("---")
            st.markdown("### Analyse du risque de rupture de stock")
            
            if not data['stock'].empty and 'QUANTITE' in data['stock'].columns:
                stock_data = data['stock'].copy()
                seuil = stock_data['QUANTITE'].quantile(0.15)
                stock_risque = stock_data[stock_data['QUANTITE'] < seuil].copy()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Produits en risque", len(stock_risque))
                with col2:
                    st.metric("Seuil critique", f"{seuil:.0f} unités")
                with col3:
                    pct_risque = len(stock_risque) / len(stock_data) * 100 if len(stock_data) > 0 else 0
                    st.metric("% du stock en risque", f"{pct_risque:.1f}%")
                
                if not stock_risque.empty:
                    st.warning(f"{len(stock_risque)} produits sont en dessous du seuil critique")
                    
                    # Affichage avec noms si disponibles
                    if 'DESIGNATION' in stock_risque.columns:
                        cols_aff = ['DESIGNATION', 'QUANTITE']
                        st.dataframe(
                            stock_risque[cols_aff].sort_values('QUANTITE'),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'DESIGNATION': 'Produit',
                                'QUANTITE': 'Quantité restante'
                            }
                        )
                    elif not data['produit'].empty:
                        stock_risque_nom = stock_risque.merge(
                            data['produit'][['ID_PRODUIT', 'NOM_PRODUIT']], 
                            on='ID_PRODUIT', 
                            how='left'
                        )
                        st.dataframe(
                            stock_risque_nom[['NOM_PRODUIT', 'QUANTITE']].sort_values('QUANTITE'),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.dataframe(
                            stock_risque[['ID_PRODUIT', 'QUANTITE']].sort_values('QUANTITE'),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.success("Aucun produit en risque de rupture critique")
            else:
                st.info("Données de stock insuffisantes pour l'analyse du risque de rupture")
        else:
            st.warning("Données insuffisantes pour les prévisions (minimum 6 mois requis)")
else:
        st.info("Aucune donnée de facture disponible pour les prévisions")

# ==========================================================
# TAB OBJECTIFS : CA CIBLE & VOLUME DE PRODUCTION
# ==========================================================

if tab_objectifs is not None:
    with tab_objectifs:
        st.markdown("## Objectifs de Chiffre d'Affaires & Plan de Production")
        st.caption("Définition des objectifs stratégiques · Simulation de production · Plan d'action")

        # ============================================
        # INITIALISATION SESSION STATE
    # ============================================
    if 'ca_objectif_calcule' not in st.session_state:
        st.session_state.ca_objectif_calcule = 0
    if 'bouteilles_requises' not in st.session_state:
        st.session_state.bouteilles_requises = 0
    if 'mp_requises' not in st.session_state:
        st.session_state.mp_requises = {}
    if 'simulation_resultats' not in st.session_state:
        st.session_state.simulation_resultats = None
    
    # ============================================
    # PRIX DE RÉFÉRENCE (modifiables)
    # ============================================
    with st.expander("Paramètres de référence (cliquer pour modifier)", expanded=False):
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            prix_vente_bouteille = st.number_input(
                "Prix de vente/bouteille (FCFA)",
                min_value=100, max_value=10000, value=1000, step=100,
                help="Prix de vente moyen d'une bouteille de jus"
            )
        with col_p2:
            cout_production_bouteille = st.number_input(
                "Coût production/bouteille (FCFA)",
                min_value=50, max_value=5000, value=450, step=50,
                help="Coût total de production par bouteille (MP + emballage + énergie)"
            )
        with col_p3:
            marge_brute = prix_vente_bouteille - cout_production_bouteille
            st.metric("Marge brute/bouteille", format_cfa(marge_brute))
        with col_p4:
            taux_marge = (marge_brute / prix_vente_bouteille * 100) if prix_vente_bouteille > 0 else 0
            st.metric("Taux de marge", f"{taux_marge:.1f}%")
    
    st.markdown("---")
    
    # ============================================
    # PARTIE 1 : DÉFINITION DE L'OBJECTIF DE CA
    # ============================================
    st.markdown("## Objectif de Chiffre d'Affaires")
    
    # Carte de saisie
    col_obj1, col_obj2 = st.columns([1, 1])
    
    with col_obj1:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #1f77b4;'>
        <h3 style='margin-top: 0;'> Saisir l'objectif de CA</h3>
        """, unsafe_allow_html=True)
        
        mode_saisie = st.radio(
            "Mode de saisie",
            ["Montant annuel cible", "Taux de croissance (%)"],
            horizontal=True
        )
        
        if mode_saisie == "Montant annuel cible":
            ca_cible = st.number_input(
                "CA Annuel Cible (FCFA)",
                min_value=1000000, max_value=10000000000, value=60000000, step=1000000,
                format="%d",
                help="Saisissez le chiffre d'affaires annuel que vous souhaitez atteindre"
            )
            # Calcul du taux de croissance implicite
            if not facture_filtered.empty and 'MONTANT_NET' in facture_filtered.columns:
                ca_actuel_annuel = facture_filtered['MONTANT_NET'].sum()
                if ca_actuel_annuel > 0:
                    croissance_implicite = ((ca_cible / ca_actuel_annuel) - 1) * 100
                    st.caption(f"📈 Croissance implicite : **{croissance_implicite:+.1f}%** (CA actuel : {format_cfa(ca_actuel_annuel)})")
        else:
            taux_croissance = st.slider(
                "Taux de croissance souhaité",
                min_value=0, max_value=200, value=20, step=5,
                help="Pourcentage d'augmentation par rapport à l'année en cours"
            )
            if not facture_filtered.empty and 'MONTANT_NET' in facture_filtered.columns:
                ca_actuel_annuel = facture_filtered['MONTANT_NET'].sum()
                ca_cible = ca_actuel_annuel * (1 + taux_croissance / 100) if ca_actuel_annuel > 0 else 60000000
            else:
                ca_cible = 60000000 * (1 + taux_croissance / 100)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Bouton de validation
        if st.button(" Calculer l'objectif", type="primary", use_container_width=True):
            st.session_state.ca_objectif_calcule = ca_cible
            
            # Calcul du volume de bouteilles requis
            st.session_state.bouteilles_requises = ca_cible / prix_vente_bouteille if prix_vente_bouteille > 0 else 0
            
            # Calcul des matières premières requises
            bouteilles_an = st.session_state.bouteilles_requises
            st.session_state.mp_requises = {
                'Sucre': {'quantite': bouteilles_an * 0.025, 'unite': 'kg', 'cout_unitaire': 450},
                'Eau traitée': {'quantite': bouteilles_an * 0.5, 'unite': 'L', 'cout_unitaire': 5},
                'Arômes naturels': {'quantite': bouteilles_an * 0.002, 'unite': 'L', 'cout_unitaire': 5000},
                'Acide citrique': {'quantite': bouteilles_an * 0.0015, 'unite': 'kg', 'cout_unitaire': 2000},
                'Conservateurs': {'quantite': bouteilles_an * 0.001, 'unite': 'kg', 'cout_unitaire': 3000},
                'Bouteilles PET': {'quantite': bouteilles_an, 'unite': 'unités', 'cout_unitaire': 75},
                'Bouchons': {'quantite': bouteilles_an, 'unite': 'unités', 'cout_unitaire': 15},
                'Étiquettes': {'quantite': bouteilles_an, 'unite': 'unités', 'cout_unitaire': 25},
                'Cartons': {'quantite': bouteilles_an / 12, 'unite': 'unités', 'cout_unitaire': 200},
            }
            
            # Calcul du résultat net prévisionnel
            ca_total = ca_cible
            cout_total_mp = sum(v['quantite'] * v['cout_unitaire'] for v in st.session_state.mp_requises.values())
            cout_energie = bouteilles_an * 15
            cout_main_oeuvre = 1200000 * 12
            cout_transport = bouteilles_an * 30
            charges_fixes = 8000000
            st.session_state.resultat_net = ca_total - cout_total_mp - cout_energie - cout_main_oeuvre - cout_transport - charges_fixes
            
            st.success(" Objectif calculé avec succès !")
    
    with col_obj2:
        if st.session_state.ca_objectif_calcule > 0:
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #2ca02c;'>
            <h3 style='margin-top: 0;'> Résultats</h3>
            """, unsafe_allow_html=True)
            
            st.metric("CA Objectif Annuel", format_cfa(st.session_state.ca_objectif_calcule))
            st.metric("Volume de Bouteilles Requis", f"{st.session_state.bouteilles_requises:,.0f} bouteilles/an")
            st.metric("Volume Mensuel Moyen", f"{st.session_state.bouteilles_requises / 12:,.0f} bouteilles/mois")
            
            st.markdown("---")
            
            if hasattr(st.session_state, 'resultat_net'):
                st.metric("Résultat Net Prévisionnel", format_cfa(st.session_state.resultat_net),
                         delta="Bénéfice" if st.session_state.resultat_net > 0 else "Perte",
                         delta_color="normal" if st.session_state.resultat_net > 0 else "inverse")
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Saisissez un objectif et cliquez sur **Calculer l'objectif** pour voir les résultats")
    
    st.markdown("---")
    
    # ============================================
    # PARTIE 2 : MATIÈRES PREMIÈRES REQUISES
    # ============================================
    if st.session_state.ca_objectif_calcule > 0:
        st.markdown("## Matières Premières Requises")
        st.caption(f"Pour produire {st.session_state.bouteilles_requises:,.0f} bouteilles")
        
        mp_data = []
        cout_total = 0
        for nom, details in st.session_state.mp_requises.items():
            cout_ligne = details['quantite'] * details['cout_unitaire']
            cout_total += cout_ligne
            mp_data.append({
                'Matière Première': nom,
                'Quantité Requise': f"{details['quantite']:,.0f} {details['unite']}",
                'Coût Unitaire': format_cfa(details['cout_unitaire']),
                'Coût Total': format_cfa(cout_ligne)
            })
        
        mp_data.append({
            'Matière Première': 'Énergie (électricité, gaz)',
            'Quantité Requise': f"{st.session_state.bouteilles_requises * 15:,.0f} FCFA",
            'Coût Unitaire': '-',
            'Coût Total': format_cfa(st.session_state.bouteilles_requises * 15)
        })
        
        df_mp = pd.DataFrame(mp_data)
        st.dataframe(
            df_mp,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Matière Première': 'Matière Première',
                'Quantité Requise': 'Quantité Requise',
                'Coût Unitaire': 'Coût Unitaire',
                'Coût Total': 'Coût Total'
            }
        )
        
        st.metric("Coût Total Matières Premières", format_cfa(cout_total + st.session_state.bouteilles_requises * 15))
    
        st.markdown("---")
    
    # ============================================
    # PARTIE 3 : SIMULATION DE PRODUCTION
    # ============================================
    st.markdown("## Simulation de Production")
    st.caption("Ajustez les paramètres pour simuler différents scénarios de production")
    
    col_sim1, col_sim2 = st.columns([1, 1])
    
    with col_sim1:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #ff7f0e;'>
        <h4 style='margin-top: 0;'> Paramètres de production</h4>
        """, unsafe_allow_html=True)
        
        capacite_horaire = st.number_input(
            "Capacité horaire (bouteilles/heure)",
            min_value=10, max_value=10000, value=200, step=10,
            help="Nombre de bouteilles produites par heure sur votre ligne"
        )
        
        heures_jour = st.slider(
            "Heures de production/jour",
            min_value=1, max_value=24, value=8, step=1
        )
        
        jours_semaine = st.slider(
            "Jours de production/semaine",
            min_value=1, max_value=7, value=5, step=1
        )
        
        nombre_equipes = st.selectbox(
            "Nombre d'équipes",
            [1, 2, 3],
            index=0,
            help="1 = 1x8, 2 = 2x8, 3 = 3x8"
        )
        
        # Facteurs de risque
        st.markdown("#### Facteurs de risque")
        
        risque_mp = st.slider("Pénurie matières premières (%)", 0, 50, 10, 5)
        risque_pannes = st.slider("Pannes machines (%)", 0, 50, 5, 5)
        risque_absent = st.slider("Absentéisme (%)", 0, 30, 5, 5)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Lancer la simulation", type="primary", use_container_width=True):
            # Calculs
            prod_journaliere = capacite_horaire * heures_jour * nombre_equipes
            prod_hebdo = prod_journaliere * jours_semaine
            prod_mensuelle = prod_hebdo * 4.33
            prod_annuelle = prod_mensuelle * 12
            
            facteur_risque = 1 - (risque_mp + risque_pannes + risque_absent) / 100
            prod_ajustee = prod_mensuelle * facteur_risque
            
            besoin_mensuel = st.session_state.bouteilles_requises / 12 if st.session_state.bouteilles_requises > 0 else 0
            ecart = prod_ajustee - besoin_mensuel
            
            st.session_state.simulation_resultats = {
                'prod_journaliere': prod_journaliere,
                'prod_hebdo': prod_hebdo,
                'prod_mensuelle': prod_mensuelle,
                'prod_annuelle': prod_annuelle,
                'prod_ajustee': prod_ajustee,
                'besoin_mensuel': besoin_mensuel,
                'ecart': ecart,
                'facteur_risque': facteur_risque,
                'capacite_horaire': capacite_horaire,
                'heures_jour': heures_jour,
                'nombre_equipes': nombre_equipes
            }
            
            st.success("Simulation terminée !")
    
    with col_sim2:
        if st.session_state.simulation_resultats is not None:
            res = st.session_state.simulation_resultats
            
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #17becf;'>
            <h4 style='margin-top: 0;'> Résultats de la simulation</h4>
            """, unsafe_allow_html=True)
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric("Production journalière", f"{res['prod_journaliere']:,.0f} bouteilles")
                st.metric("Production hebdomadaire", f"{res['prod_hebdo']:,.0f} bouteilles")
            with col_r2:
                st.metric("Production mensuelle", f"{res['prod_mensuelle']:,.0f} bouteilles")
                st.metric("Production annuelle", f"{res['prod_annuelle']:,.0f} bouteilles")
            
            st.markdown("---")
            st.metric("Production ajustée (après risques)", f"{res['prod_ajustee']:,.0f} bouteilles/mois")
            
            st.markdown("---")
            
            if st.session_state.bouteilles_requises > 0:
                if res['ecart'] > 0:
                    st.success(f"""
                     **Capacité SUFFISANTE**
                    
                    Besoin : {res['besoin_mensuel']:,.0f} bouteilles/mois
                    Disponible : {res['prod_ajustee']:,.0f} bouteilles/mois
                    Marge : +{res['ecart']:,.0f} bouteilles/mois
                    """)
                else:
                    st.error(f"""
                     **Capacité INSUFFISANTE**
                    
                    Besoin : {res['besoin_mensuel']:,.0f} bouteilles/mois
                    Disponible : {res['prod_ajustee']:,.0f} bouteilles/mois
                    Déficit : {abs(res['ecart']):,.0f} bouteilles/mois
                    
                    **Solutions :**
                    - Passer de {res['nombre_equipes']} à {res['nombre_equipes'] + 1} équipe(s)
                    - Augmenter les heures/jour de {res['heures_jour']}h à {min(res['heures_jour'] + 4, 24)}h
                    - Réduire les risques (stocks sécurité, maintenance)
                    """)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Ajustez les paramètres et cliquez sur **Lancer la simulation**")
    
    st.markdown("---")
    
    # ============================================
    # PARTIE 4 : PLAN D'ACTION
    # ============================================
    st.markdown("## Plan d'Action Stratégique")
    st.caption("Actions prioritaires pour atteindre les objectifs")
    
    plan_actions = {
        "Axe stratégique": [
            "Sécurisation approvisionnements",
            "Sécurisation approvisionnements",
            "Optimisation production",
            "Optimisation production",
            "Développement commercial",
            "Développement commercial",
            "Gestion RH",
            "Gestion financière"
        ],
        "Action": [
            "Constituer un stock de sécurité de 2 mois pour les matières premières critiques",
            "Diversifier les fournisseurs (signer avec 2 fournisseurs supplémentaires)",
            "Mettre en place un plan de maintenance préventive mensuelle",
            "Optimiser la chaîne de production (réduire les temps morts de 15%)",
            "Lancer une campagne marketing sur les zones à fort potentiel",
            "Développer 2 nouveaux produits pour les périodes creuses",
            "Former 3 employés polyvalents pour réduire l'impact de l'absentéisme",
            "Négocier un crédit de trésorerie pour financer la croissance"
        ],
        "Priorité": ["Urgent", "Élevée", "Élevée", "Moyenne", "Moyenne", "Moyenne", "Normale", "Normale"],
        "Délai": ["1 mois", "2 mois", "1 mois", "3 mois", "2 mois", "4 mois", "3 mois", "1 mois"],
        "Budget estimé": [
            format_cfa(2000000),
            format_cfa(500000),
            format_cfa(800000),
            format_cfa(300000),
            format_cfa(1500000),
            format_cfa(2000000),
            format_cfa(600000),
            format_cfa(1000000)
        ],
        "KPI de suivi": [
            "Taux de rupture < 5%",
            "Nombre de fournisseurs actifs",
            "Taux de panne < 2%",
            "Productivité horaire",
            "CA par zone",
            "Nombre de nouveaux clients",
            "Taux d'absentéisme",
            "Trésorerie disponible"
        ]
    }
    
    df_plan = pd.DataFrame(plan_actions)
    st.dataframe(
        df_plan,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Axe stratégique': st.column_config.TextColumn('Axe stratégique', width='medium'),
            'Action': st.column_config.TextColumn('Action', width='large'),
            'Priorité': 'Priorité',
            'Délai': 'Délai',
            'Budget estimé': 'Budget estimé',
            'KPI de suivi': st.column_config.TextColumn('KPI de suivi', width='medium')
        }
    )
    
    budget_total_actions = 2000000 + 500000 + 800000 + 300000 + 1500000 + 2000000 + 600000 + 1000000
    st.metric("Budget Total Plan d'Action", format_cfa(budget_total_actions))
    
    st.markdown("---")
    
    # ============================================
    # RÉSUMÉ DÉCISIONNEL
    # ============================================
    if st.session_state.ca_objectif_calcule > 0:
        st.markdown("## Résumé Décisionnel")
        
        col_dec1, col_dec2, col_dec3, col_dec4 = st.columns(4)
        
        with col_dec1:
            st.metric("CA Objectif", format_cfa(st.session_state.ca_objectif_calcule))
        with col_dec2:
            st.metric("Bouteilles/an", f"{st.session_state.bouteilles_requises:,.0f}")
        with col_dec3:
            if hasattr(st.session_state, 'resultat_net'):
                st.metric("Résultat net", format_cfa(st.session_state.resultat_net))
        with col_dec4:
            if st.session_state.simulation_resultats is not None:
                ecart_final = st.session_state.simulation_resultats['ecart']
                if ecart_final >= 0:
                    st.metric("Capacité", "Suffisante")
                else:
                    st.metric("Capacité", "Insuffisante")
        
        st.markdown("---")
        
        # Calcul du seuil de rentabilité simplifié
        if hasattr(st.session_state, 'resultat_net'):
            charges_fixes_total = 8000000 + 1200000 * 12  # fixes + main d'oeuvre
            taux_marge_variable = marge_brute / prix_vente_bouteille if prix_vente_bouteille > 0 else 0
            seuil_rentabilite_ca = charges_fixes_total / taux_marge_variable if taux_marge_variable > 0 else 0
            seuil_rentabilite_bouteilles = seuil_rentabilite_ca / prix_vente_bouteille if prix_vente_bouteille > 0 else 0
            
            col_sr1, col_sr2, col_sr3 = st.columns(3)
            with col_sr1:
                st.metric("Seuil de rentabilité (CA)", format_cfa(seuil_rentabilite_ca))
            with col_sr2:
                st.metric("Seuil de rentabilité (bouteilles)", f"{seuil_rentabilite_bouteilles:,.0f}")
            with col_sr3:
                ca_obj = st.session_state.ca_objectif_calcule
                marge_securite = ((ca_obj - seuil_rentabilite_ca) / ca_obj * 100) if ca_obj > 0 else 0
                st.metric("Marge de sécurité", f"{marge_securite:.1f}%")
        
        st.info(f"""
        ** Synthèse pour décision :**
        
        Pour atteindre un CA de **{format_cfa(st.session_state.ca_objectif_calcule)}**, il faut produire **{st.session_state.bouteilles_requises:,.0f} bouteilles/an**.
        
        Le résultat net prévisionnel est de **{format_cfa(st.session_state.resultat_net) if hasattr(st.session_state, 'resultat_net') else 'N/A'}**.
        
        Le plan d'action nécessite un budget de **{format_cfa(budget_total_actions)}** pour sécuriser la production et développer les ventes.
        """)

# ==========================================================
# TAB PRODUCTION VS VENTES : COMPARAISON VOLUMES
# ==========================================================

if tab_prod_ventes is not None:
    with tab_prod_ventes:
        st.markdown("## Production vs Ventes")
        st.caption("Comparaison des volumes produits et vendus · Écarts · Taux de couverture")

        # ============================================
        # INITIALISATION SESSION STATE
    # ============================================
    if 'prod_ventes_data' not in st.session_state:
        st.session_state.prod_ventes_data = None
    
    # ============================================
    # PARAMÈTRES
    # ============================================
    with st.expander("Paramètres de comparaison", expanded=False):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            unite_comparaison = st.radio(
                "Unité de comparaison",
                ["Bouteilles", "Litres"],
                horizontal=True,
                help="Comparer en nombre de bouteilles ou en volume (litres)"
            )
        with col_p2:
            volume_bouteille = st.number_input(
                "Volume par bouteille (L)",
                min_value=0.1, max_value=5.0, value=0.5, step=0.1,
                help="Volume en litres d'une bouteille standard"
            ) if unite_comparaison == "Litres" else 1
        with col_p3:
            seuil_alerte_ecart = st.slider(
                "Seuil d'alerte écart (%)",
                min_value=5, max_value=50, value=15, step=5,
                help="Pourcentage d'écart au-delà duquel une alerte est déclenchée"
            )
    
    st.markdown("---")
    
    # ============================================
    # SAISIE DES DONNÉES DE PRODUCTION
    # ============================================
    st.markdown("## Saisie des Volumes de Production")
    st.caption("Renseignez les volumes produits pour chaque mois")
    
    col_s1, col_s2 = st.columns([1, 1])
    
    with col_s1:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #1f77b4;'>
        <h4 style='margin-top: 0;'> Production mensuelle</h4>
        """, unsafe_allow_html=True)
        
        annee_prod = st.selectbox(
            "Année de production",
            [datetime.now().year - 1, datetime.now().year, datetime.now().year + 1],
            index=1
        )
        
        production_mensuelle = {}
        for mois in range(1, 13):
            production_mensuelle[mois] = st.number_input(
                f"{MOIS_FR[mois]}",
                min_value=0, max_value=1000000, value=0, step=100,
                key=f"prod_{mois}"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Enregistrer la production", type="primary", use_container_width=True):
            st.session_state.prod_ventes_data = {
                'production': production_mensuelle,
                'annee': annee_prod
            }
            st.success("Données de production enregistrées !")
    
    with col_s2:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #2ca02c;'>
        <h4 style='margin-top: 0;'>Production totale</h4>
        """, unsafe_allow_html=True)
        
        total_production = sum(production_mensuelle.values())
        st.metric("Production annuelle saisie", f"{total_production:,.0f} " + ("bouteilles" if unite_comparaison == "Bouteilles" else "litres"))
        st.metric("Moyenne mensuelle", f"{total_production / 12:,.0f} " + ("bouteilles/mois" if unite_comparaison == "Bouteilles" else "litres/mois"))
        
        # Mois le plus productif
        if total_production > 0:
            mois_max = max(production_mensuelle, key=production_mensuelle.get)
            st.metric("Mois le plus productif", f"{MOIS_FR[mois_max]} ({production_mensuelle[mois_max]:,.0f})")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # COMPARAISON PRODUCTION VS VENTES
    # ============================================
    if st.session_state.prod_ventes_data is not None:
        st.markdown("## Comparaison Production vs Ventes")
        
        prod_data = st.session_state.prod_ventes_data['production']
        
        # Récupération des ventes réelles (ou simulation si pas de données)
        if not facture_filtered.empty and 'MOIS' in facture_filtered.columns:
            ventes_mensuelles = facture_filtered.groupby('MOIS')['MONTANT_NET'].sum()
            # Estimation du nombre de bouteilles vendues
            if 'bouteilles_vendues' in facture_filtered.columns:
                ventes_bouteilles = facture_filtered.groupby('MOIS')['bouteilles_vendues'].sum()
            else:
                # Estimation basée sur le CA et le prix moyen
                prix_moyen = facture_filtered['MONTANT_NET'].mean() / max(facture_filtered['MONTANT_NET'].count(), 1)
                ventes_bouteilles = ventes_mensuelles / 1000  # Prix moyen estimé à 1000 FCFA
        else:
            # Simulation pour démonstration
            np.random.seed(42)
            ventes_bouteilles = pd.Series({
                mois: np.random.randint(5000, 20000) for mois in range(1, 13)
            })
        
        # Création du tableau comparatif
        comparaison_data = []
        for mois in range(1, 13):
            prod = prod_data.get(mois, 0)
            ventes = ventes_bouteilles.get(mois, 0) if isinstance(ventes_bouteilles, pd.Series) else 0
            
            # Conversion en litres si nécessaire
            if unite_comparaison == "Litres":
                prod = prod * volume_bouteille
                ventes = ventes * volume_bouteille
            
            ecart = prod - ventes
            ecart_pct = (ecart / prod * 100) if prod > 0 else 0
            taux_couverture = (ventes / prod * 100) if prod > 0 else 0
            
            if ecart > 0:
                statut = "Surplus" if ecart_pct > seuil_alerte_ecart else "OK"
            elif ecart < 0:
                statut = "Rupture" if abs(ecart_pct) > seuil_alerte_ecart else "OK"
            else:
                statut = "Équilibre"
            
            comparaison_data.append({
                'Mois': MOIS_FR[mois],
                'Mois_Num': mois,
                'Production': prod,
                'Ventes': ventes,
                'Écart': ecart,
                'Écart %': f"{ecart_pct:+.1f}%",
                'Taux couverture': f"{taux_couverture:.1f}%",
                'Statut': statut
            })
        
        df_comparaison = pd.DataFrame(comparaison_data)
        
        # KPIs
        total_prod_annuel = df_comparaison['Production'].sum()
        total_ventes_annuel = df_comparaison['Ventes'].sum()
        ecart_global = total_prod_annuel - total_ventes_annuel
        taux_couverture_global = (total_ventes_annuel / total_prod_annuel * 100) if total_prod_annuel > 0 else 0
        
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.metric("Production totale", f"{total_prod_annuel:,.0f}")
        with col_k2:
            st.metric("Ventes totales", f"{total_ventes_annuel:,.0f}")
        with col_k3:
            st.metric("Écart global", f"{ecart_global:+,.0f}",
                     delta="Surproduction" if ecart_global > 0 else "Sous-production",
                     delta_color="inverse" if ecart_global > 0 else "normal")
        with col_k4:
            delta_color = "normal" if 95 <= taux_couverture_global <= 105 else "inverse"
            st.metric("Taux de couverture", f"{taux_couverture_global:.1f}%",
                     delta="Optimal" if 95 <= taux_couverture_global <= 105 else "Déséquilibré",
                     delta_color=delta_color)
        
        st.markdown("---")
        
        # ============================================
        # GRAPHIQUE COMPARATIF
        # ============================================
        st.markdown("### Graphique Comparatif Mensuel")
        
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Bar(
            x=df_comparaison['Mois'],
            y=df_comparaison['Production'],
            name='Production',
            marker_color=COLORS['primary'],
            hovertemplate='<b>%{x}</b><br>Production: %{y:,.0f}<extra></extra>'
        ))
        
        fig_comp.add_trace(go.Bar(
            x=df_comparaison['Mois'],
            y=df_comparaison['Ventes'],
            name='Ventes',
            marker_color=COLORS['success'],
            hovertemplate='<b>%{x}</b><br>Ventes: %{y:,.0f}<extra></extra>'
        ))
        
        fig_comp.add_trace(go.Scatter(
            x=df_comparaison['Mois'],
            y=df_comparaison['Écart'],
            name='Écart',
            mode='lines+markers',
            line=dict(color=COLORS['danger'], width=3),
            marker=dict(size=10, symbol='diamond'),
            yaxis='y2',
            hovertemplate='<b>%{x}</b><br>Écart: %{y:+,.0f}<extra></extra>'
        ))
        
        fig_comp.update_layout(
            title=f"Production vs Ventes - {annee_prod}",
            xaxis_title="Mois",
            yaxis_title="Volume (" + ("bouteilles" if unite_comparaison == "Bouteilles" else "litres") + ")",
            yaxis2=dict(
                title="Écart",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            template='plotly_white',
            height=450,
            barmode='group',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("---")
        
        # ============================================
        # TABLEAU DÉTAILLÉ
        # ============================================
        st.markdown("### Détail Mensuel")
        
        df_display = df_comparaison.copy()
        df_display['Production_F'] = df_display['Production'].apply(lambda x: f"{x:,.0f}")
        df_display['Ventes_F'] = df_display['Ventes'].apply(lambda x: f"{x:,.0f}")
        df_display['Écart_F'] = df_display['Écart'].apply(lambda x: f"{x:+,.0f}")
        
        st.dataframe(
            df_display[['Mois', 'Production_F', 'Ventes_F', 'Écart_F', 'Écart %', 'Taux couverture', 'Statut']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Mois': 'Mois',
                'Production_F': 'Production',
                'Ventes_F': 'Ventes',
                'Écart_F': 'Écart',
                'Écart %': 'Écart %',
                'Taux couverture': 'Taux couverture',
                'Statut': 'Statut'
            }
        )
        
        st.markdown("---")
        
        # ============================================
        # ANALYSE DES ÉCARTS
        # ============================================
        st.markdown("### Analyse des Écarts")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.markdown("#### Mois en Surproduction")
            surplus = df_comparaison[df_comparaison['Écart'] > 0].sort_values('Écart', ascending=False)
            
            if not surplus.empty:
                for _, row in surplus.iterrows():
                    st.warning(f"""
                    **{row['Mois']}** : +{row['Écart']:,.0f} ({row['Écart %']} d'écart)
                    → Production supérieure aux ventes. Risque de stock dormant.
                    """)
            else:
                st.success("Aucun mois en surproduction")
        
        with col_a2:
            st.markdown("#### Mois en Sous-production")
            deficit = df_comparaison[df_comparaison['Écart'] < 0].sort_values('Écart')
            
            if not deficit.empty:
                for _, row in deficit.iterrows():
                    st.error(f"""
                    **{row['Mois']}** : {row['Écart']:,.0f} ({row['Écart %']} d'écart)
                    → Production insuffisante. Risque de rupture de stock.
                    """)
            else:
                st.success("Aucun mois en sous-production")
        
        st.markdown("---")
        
        # ============================================
        # RECOMMANDATIONS
        # ============================================
        st.markdown("### Recommandations")
        
        nb_mois_surplus = len(df_comparaison[df_comparaison['Écart'] > 0])
        nb_mois_deficit = len(df_comparaison[df_comparaison['Écart'] < 0])
        nb_mois_ok = 12 - nb_mois_surplus - nb_mois_deficit
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Mois équilibrés", nb_mois_ok)
        with col_r2:
            st.metric("Mois en surplus", nb_mois_surplus)
        with col_r3:
            st.metric("Mois en déficit", nb_mois_deficit)
        
        if ecart_global > 0:
            st.warning(f"""
            ** Action recommandée :**
            
            La production dépasse les ventes de **{ecart_global:,.0f}** unités sur l'année.
            
            - Réduire la production de **{(ecart_global / total_prod_annuel * 100):.1f}%** pour aligner sur la demande réelle
            - Liquider les stocks excédentaires via des promotions
            - Revoir les prévisions de vente à la baisse pour les mois concernés
            - Coût estimé du surstock : **{format_cfa(ecart_global * cout_production_bouteille if 'cout_production_bouteille' in dir() else ecart_global * 450)}**
            """)
        elif ecart_global < 0:
            st.error(f"""
            ** Action recommandée :**
            
            La production est inférieure aux ventes de **{abs(ecart_global):,.0f}** unités sur l'année.
            
            - Augmenter la capacité de production de **{(abs(ecart_global) / total_prod_annuel * 100):.1f}%**
            - Investir dans une ligne supplémentaire ou ajouter une équipe
            - Manque à gagner estimé : **{format_cfa(abs(ecart_global) * marge_brute if 'marge_brute' in dir() else abs(ecart_global) * 550)}**
            """)
        else:
            st.success("""
            ** La production est parfaitement alignée avec les ventes.**
            
            Continuez à suivre les écarts mensuellement pour maintenir cet équilibre.
            """)
        
        # Export
        csv_comparaison = df_comparaison[['Mois', 'Production', 'Ventes', 'Écart', 'Écart %', 'Taux couverture']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Exporter l'analyse (CSV)",
            data=csv_comparaison,
            file_name=f"comparaison_production_ventes_{annee_prod}.csv",
            mime="text/csv"
        )
    
    else:
        st.info("Renseignez les volumes de production mensuels et cliquez sur **Enregistrer la production** pour voir la comparaison.")
        
        # Aperçu des ventes actuelles si disponibles
        if not facture_filtered.empty and 'MOIS' in facture_filtered.columns:
            st.markdown("---")
            st.markdown("### Ventes actuelles (pour référence)")
            
            ventes_mensuelles = facture_filtered.groupby('MOIS')['MONTANT_NET'].sum()
            
            ventes_ref = []
            for mois in range(1, 13):
                ca = ventes_mensuelles.get(mois, 0)
                ventes_ref.append({
                    'Mois': MOIS_FR[mois],
                    'CA': ca,
                    'Bouteilles estimées': ca / 1000 if ca > 0 else 0
                })
            
            df_ventes_ref = pd.DataFrame(ventes_ref)
            df_ventes_ref['CA_F'] = df_ventes_ref['CA'].apply(format_cfa)
            df_ventes_ref['Bouteilles_F'] = df_ventes_ref['Bouteilles estimées'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(
                df_ventes_ref[['Mois', 'CA_F', 'Bouteilles_F']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Mois': 'Mois',
                    'CA_F': 'Chiffre d\'affaires',
                    'Bouteilles_F': 'Bouteilles estimées'
                }
            )
            st.caption("Estimation basée sur un prix moyen de 1 000 FCFA/bouteille")

# ==========================================================
# TAB 2 : CORRÉLATION SAISONS
# ==========================================================

if tab2 is not None:
    with tab2:
        st.markdown("## Corrélation Ventes / Saisons")
    
    if not facture_filtered.empty and 'SAISON' in facture_filtered.columns and 'MONTANT_NET' in facture_filtered.columns:
        # Analyse par saison (basée sur les clients)
        ca_par_saison = facture_filtered.groupby('SAISON').agg({
            'MONTANT_NET': ['sum', 'mean'],
            'ID_PERSONNE': 'nunique'  # Nombre de clients uniques au lieu du nombre de factures
        }).reset_index()
        ca_par_saison.columns = ['Saison', 'CA_Total', 'CA_Moyen', 'Nombre_Clients']
        
        # Graphique comparatif saisons
        st.markdown("### Performance par saison")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(
            x=ca_par_saison['Saison'],
            y=ca_par_saison['CA_Total'],
            name='CA Total',
            marker_color=[COLORS['secondary'], COLORS['primary']],
            hovertemplate='<b>%{x}</b><br>CA: %{y:,.0f} FCFA<extra></extra>'
        ), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=ca_par_saison['Saison'],
            y=ca_par_saison['Nombre_Clients'],
            name='Nombre clients',
            mode='lines+markers',
            line=dict(color=COLORS['success'], width=3),
            marker=dict(size=12, symbol='diamond')
        ), secondary_y=True)
        
        fig.update_layout(
            title="Chiffre d'affaires et nombre de clients par saison",
            xaxis_title="Saison",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.05)
        )
        fig.update_yaxes(title_text="CA (FCFA)", secondary_y=False, tickformat=',.0f')
        fig.update_yaxes(title_text="Nombre de clients", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyse mensuelle par saison
        st.markdown("### Évolution mensuelle du CA")
        
        ca_par_mois = facture_filtered.groupby(['MOIS', 'MOIS_NOM', 'SAISON'])['MONTANT_NET'].sum().reset_index()
        ca_par_mois = ca_par_mois.sort_values('MOIS')
        
        fig = px.bar(
            ca_par_mois,
            x='MOIS_NOM',
            y='MONTANT_NET',
            color='SAISON',
            title="CA mensuel par saison",
            labels={'MONTANT_NET': 'CA (FCFA)', 'MOIS_NOM': 'Mois'},
            color_discrete_map={
                'Saison des pluies': COLORS['primary'],
                'Saison sèche': COLORS['secondary']
            }
        )
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights saisonniers
        st.markdown("### Insights saisonniers")
        
        col1, col2 = st.columns(2)
        with col1:
            meilleure_saison = ca_par_saison.loc[ca_par_saison['CA_Total'].idxmax()]
            st.success(f"""
            **Meilleure saison : {meilleure_saison['Saison']}**
            - CA Total : {format_cfa(meilleure_saison['CA_Total'])}
            - Nombre de clients : {meilleure_saison['Nombre_Clients']:.0f}
            - CA Moyen par client : {format_cfa(meilleure_saison['CA_Moyen'])}
            """)
        
        with col2:
            if not ca_par_mois.empty:
                mois_pic = ca_par_mois.loc[ca_par_mois['MONTANT_NET'].idxmax()]
                mois_creux = ca_par_mois.loc[ca_par_mois['MONTANT_NET'].idxmin()]
                st.info(f"""
                **Mois de pic : {mois_pic['MOIS_NOM']}**
                - CA : {format_cfa(mois_pic['MONTANT_NET'])}
                
                **Mois le plus creux : {mois_creux['MOIS_NOM']}**
                - CA : {format_cfa(mois_creux['MONTANT_NET'])}
                """)
        
        # ============================================
        # ANALYSE DÉTAILLÉE DES COEFFICIENTS DE SAISONNALITÉ
        # ============================================
        st.markdown("---")
        st.markdown("### Coefficients de Saisonnalité Détaillés")
        
        ca_par_mois_detail = facture_filtered.groupby('MOIS')['MONTANT_NET'].sum()
        ca_moyen_detail = ca_par_mois_detail.mean()
        
        coeff_data = []
        for mois in range(1, 13):
            ca_mois = ca_par_mois_detail.get(mois, 0)
            coeff = round(ca_mois / ca_moyen_detail, 2) if ca_moyen_detail > 0 else 0
            saison = get_saison(mois)
            
            if coeff > 1.2:
                potentiel = 'Très fort'
                recommandation = 'Augmenter production et stocks'
            elif coeff >= 1.05:
                potentiel = 'Fort'
                recommandation = 'Maintenir niveau de production'
            elif coeff >= 0.95:
                potentiel = 'Neutre'
                recommandation = 'Production standard'
            elif coeff >= 0.8:
                potentiel = 'Faible'
                recommandation = 'Réduire production, privilégier maintenance'
            elif coeff > 0:
                potentiel = 'Très faible'
                recommandation = 'Production minimale, promotions'
            else:
                potentiel = 'Pas de données'
                recommandation = 'Collecter des données'
            
            coeff_data.append({
                'Mois': MOIS_FR[mois],
                'Abréviation': MOIS_ABBR[mois],
                'Saison': saison,
                'CA Mensuel': ca_mois,
                'Coefficient': coeff,
                'Potentiel': potentiel,
                'Recommandation': recommandation
            })
        
        df_coeff = pd.DataFrame(coeff_data)
        
        # Graphique des coefficients
        fig_coeff = px.bar(
            df_coeff,
            x='Abréviation',
            y='Coefficient',
            color='Potentiel',
            color_discrete_map={
                'Très fort': '#2ca02c',
                'Fort': '#4caf50',
                'Neutre': '#ffbb78',
                'Faible': '#ff7f0e',
                'Très faible': '#d62728',
                'Pas de données': '#7f7f7f'
            },
            title="Coefficient de saisonnalité par mois",
            labels={'Coefficient': 'Coefficient (>1 = fort potentiel, <1 = faible potentiel)'}
        )
        fig_coeff.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Moyenne annuelle")
        fig_coeff.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig_coeff, use_container_width=True)
        
        # Tableau détaillé des coefficients
        st.markdown("#### Détail des coefficients avec recommandations")
        
        df_coeff['CA_Mensuel_F'] = df_coeff['CA Mensuel'].apply(format_cfa)
        
        st.dataframe(
            df_coeff[['Mois', 'Saison', 'CA_Mensuel_F', 'Coefficient', 'Potentiel', 'Recommandation']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Mois': 'Mois',
                'Saison': 'Saison',
                'CA_Mensuel_F': 'CA Mensuel',
                'Coefficient': st.column_config.NumberColumn('Coefficient', format="%.2f"),
                'Potentiel': 'Potentiel',
                'Recommandation': 'Action recommandée'
            }
        )
        
        st.caption("""
        ** Lecture du coefficient :**
        - **> 1.20** : Très fort potentiel (ventes > 20% au-dessus de la moyenne) → Augmenter la production
        - **1.05 - 1.20** : Fort potentiel → Maintenir un bon niveau de stock
        - **0.95 - 1.05** : Neutre → Production standard
        - **0.80 - 0.95** : Faible potentiel → Réduire la production, faire de la maintenance
        - **< 0.80** : Très faible potentiel → Production minimale, lancer des promotions
        """)

        # Recommandations actuelles
        saison_actuelle = get_saison(datetime.now().month)
        ca_moyen_saison = ca_par_saison[ca_par_saison['Saison'] == saison_actuelle]['CA_Moyen'].values
        ca_moyen_saison = ca_moyen_saison[0] if len(ca_moyen_saison) > 0 else 0
        
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;'>
            <b>Recommandations pour la {saison_actuelle}</b><br>
            - CA moyen attendu : {format_cfa(ca_moyen_saison)}<br>
            - Anticiper les stocks en conséquence<br>
            - Adapter les promotions selon la saisonnalité
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Données insuffisantes pour l'analyse saisonnière")

# ==========================================================
# TAB 3 : CARTOGRAPHIE
# ==========================================================

if tab3 is not None:
    with tab3:
        st.markdown("## Géolocalisation des ventes")


    if not facture_filtered.empty and 'ID_PERSONNE' in facture_filtered.columns:
        # Simulation des zones (à remplacer par vos vraies données d'adresse)
        np.random.seed(42)
        
        zones = [
            'Bonamoussadi', 'Akwa', 'Bastos', 'Mokolo', 'Mfoundi', 
            'Nlongkak', 'Biyem-Assi', 'Etoudi', 'Mendong', 'Odza',
            'Nkolbisson', 'Ngousso', 'Oyom-Abang', 'Ekounou', 'Mvan'
        ]
        
        if 'ADRESSE' in facture_filtered.columns:
            # TODO: Extraire les vraies zones des adresses
            facture_filtered['ZONE'] = np.random.choice(zones, len(facture_filtered))
        elif not data['personne'].empty and 'ADRESSE' in data['personne'].columns:
            # Utiliser les adresses des personnes
            personne_adresse = data['personne'][['ID_PERSONNE', 'ADRESSE']]
            facture_filtered = facture_filtered.merge(personne_adresse, on='ID_PERSONNE', how='left')
            facture_filtered['ZONE'] = np.random.choice(zones, len(facture_filtered))
        else:
            facture_filtered['ZONE'] = np.random.choice(zones, len(facture_filtered))
        
        # Agrégation par zone
        ventes_zone = facture_filtered.groupby('ZONE').agg({
            'MONTANT_NET': ['sum', 'mean', 'count']
        }).reset_index()
        ventes_zone.columns = ['Zone', 'CA_Total', 'CA_Moyen', 'Nombre_Ventes']
        ventes_zone = ventes_zone.sort_values('CA_Total', ascending=False)
        
        # Graphique top zones
        st.markdown("### Top zones par chiffre d'affaires")
        
        fig = px.bar(
            ventes_zone.head(10),
            x='Zone',
            y='CA_Total',
            color='CA_Total',
            color_continuous_scale='Blues',
            title="Top 10 zones par chiffre d'affaires",
            labels={'CA_Total': 'CA (FCFA)'},
            text=ventes_zone.head(10)['CA_Total'].apply(format_cfa)
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé
        st.markdown("### Détail par zone")
        
        ventes_display = ventes_zone.copy()
        ventes_display['CA_Total_F'] = ventes_display['CA_Total'].apply(format_cfa)
        ventes_display['CA_Moyen_F'] = ventes_display['CA_Moyen'].apply(format_cfa)
        
        st.dataframe(
            ventes_display[['Zone', 'CA_Total_F', 'CA_Moyen_F', 'Nombre_Ventes']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Zone': 'Zone géographique',
                'CA_Total_F': 'CA Total',
                'CA_Moyen_F': 'CA Moyen',
                'Nombre_Ventes': 'Nb ventes'
            }
        )
        
        # Carte simulée
        st.markdown("### Carte de densité des ventes")
        
        # Coordonnées approximatives pour Yaoundé
        coordonnees_yaounde = {
            'Bonamoussadi': (4.05, 9.72),
            'Akwa': (4.03, 9.70),
            'Bastos': (4.01, 9.73),
            'Mokolo': (4.00, 9.69),
            'Mfoundi': (3.98, 9.68),
            'Nlongkak': (4.02, 9.71),
            'Biyem-Assi': (3.95, 9.65),
            'Etoudi': (4.04, 9.74),
            'Mendong': (3.92, 9.64),
            'Odza': (4.06, 9.76),
            'Nkolbisson': (3.94, 9.63),
            'Ngousso': (4.08, 9.77),
            'Oyom-Abang': (3.97, 9.67),
            'Ekounou': (3.99, 9.70),
            'Mvan': (3.93, 9.62)
        }
        
        ventes_zone['LAT'] = ventes_zone['Zone'].map(lambda x: coordonnees_yaounde.get(x, (4.0, 9.7))[0])
        ventes_zone['LON'] = ventes_zone['Zone'].map(lambda x: coordonnees_yaounde.get(x, (4.0, 9.7))[1])
        ventes_zone['TAILLE'] = ventes_zone['CA_Total'] / ventes_zone['CA_Total'].max() * 50 + 10
        
        fig = px.scatter_mapbox(
            ventes_zone,
            lat="LAT",
            lon="LON",
            size="TAILLE",
            color="CA_Total",
            hover_name="Zone",
            hover_data={'CA_Total': ':,d', 'Nombre_Ventes': True},
            color_continuous_scale=px.colors.sequential.Viridis,
            size_max=60,
            zoom=11,
            center={"lat": 3.98, "lon": 9.70},
            mapbox_style="open-street-map",
            title="Carte des ventes par zone - Yaoundé"
        )
        fig.update_layout(height=500, margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        if not ventes_zone.empty:
            top_zone = ventes_zone.iloc[0]
            bottom_zone = ventes_zone.iloc[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"""
                **Zone la plus performante : {top_zone['Zone']}**
                - CA : {format_cfa(top_zone['CA_Total'])}
                - {top_zone['Nombre_Ventes']} ventes
                """)
            with col2:
                st.warning(f"""
                **Zone à potentiel : {bottom_zone['Zone']}**
                - CA : {format_cfa(bottom_zone['CA_Total'])}
                - {bottom_zone['Nombre_Ventes']} ventes
                - Opportunité de développement
                """)
        
        st.info("""
        **Prochaines étapes pour la géolocalisation :**
        - Intégrer les adresses réelles des clients
        - Optimiser les tournées de livraison
        - Identifier les zones de chalandise à fort potentiel
        """)
    else:
        st.info("Aucune donnée de vente disponible pour l'analyse géographique")


# ==========================================================
# TAB 4 : MAGASIN MATIÈRES PREMIÈRES
# ==========================================================

if tab4 is not None:
    with tab4:
        st.markdown("## Magasin Matières Premières")
        st.caption("Suivi des entrées/sorties · Inventaire · Stocks · Fournisseurs · Approvisionnements")

        # ============================================
    # DONNÉES DES MATIÈRES PREMIÈRES
    # ============================================
    matieres_premieres_list = [
        {"Matière": "Sucre", "Type": "Édulcorant", "Unité": "kg", "Stock_Min": 500, "Prix_Unitaire": 450},
        {"Matière": "Eau traitée", "Type": "Base", "Unité": "L", "Stock_Min": 10000, "Prix_Unitaire": 5},
        {"Matière": "Arômes naturels", "Type": "Additif", "Unité": "L", "Stock_Min": 50, "Prix_Unitaire": 5000},
        {"Matière": "Acide citrique", "Type": "Conservateur", "Unité": "kg", "Stock_Min": 30, "Prix_Unitaire": 2000},
        {"Matière": "Conservateurs", "Type": "Additif", "Unité": "kg", "Stock_Min": 25, "Prix_Unitaire": 3000},
        {"Matière": "Colorants naturels", "Type": "Additif", "Unité": "kg", "Stock_Min": 10, "Prix_Unitaire": 4500},
        {"Matière": "Gingembre", "Type": "Matière première", "Unité": "kg", "Stock_Min": 200, "Prix_Unitaire": 500},
        {"Matière": "Ananas", "Type": "Fruit", "Unité": "pièce", "Stock_Min": 150, "Prix_Unitaire": 200},
        {"Matière": "Citron", "Type": "Fruit", "Unité": "kg", "Stock_Min": 100, "Prix_Unitaire": 150},
        {"Matière": "Orange", "Type": "Fruit", "Unité": "kg", "Stock_Min": 120, "Prix_Unitaire": 180},
        {"Matière": "Mangue", "Type": "Fruit", "Unité": "pièce", "Stock_Min": 80, "Prix_Unitaire": 250},
        {"Matière": "Goyave", "Type": "Fruit", "Unité": "kg", "Stock_Min": 60, "Prix_Unitaire": 180},
        {"Matière": "Fruit de la passion", "Type": "Fruit", "Unité": "kg", "Stock_Min": 40, "Prix_Unitaire": 300},
        {"Matière": "Papaye", "Type": "Fruit", "Unité": "pièce", "Stock_Min": 50, "Prix_Unitaire": 200},
        {"Matière": "Pastèque", "Type": "Fruit", "Unité": "pièce", "Stock_Min": 30, "Prix_Unitaire": 350},
        {"Matière": "Baobab (poudre)", "Type": "Fruit sec", "Unité": "kg", "Stock_Min": 20, "Prix_Unitaire": 1000},
        {"Matière": "Hibiscus/Bissap (séché)", "Type": "Fruit sec", "Unité": "kg", "Stock_Min": 15, "Prix_Unitaire": 800},
        {"Matière": "Tamarin", "Type": "Fruit sec", "Unité": "kg", "Stock_Min": 10, "Prix_Unitaire": 600},
    ]
    
    df_matieres_ref = pd.DataFrame(matieres_premieres_list)
    
    # ============================================
    # FILTRES DE DATE
    # ============================================
    st.markdown("---")
    st.markdown("### Période d'analyse")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        date_debut = st.date_input(
            "Date début",
            value=datetime.now() - timedelta(days=30),
            format="DD/MM/YYYY"
        )
    with col_f2:
        date_fin = st.date_input(
            "Date fin",
            value=datetime.now(),
            format="DD/MM/YYYY"
        )
    with col_f3:
        type_mouvement = st.selectbox(
            "Type de mouvement",
            ["Tous", "Entrées", "Sorties"]
        )
    
    # ============================================
    # GÉNÉRATION DES MOUVEMENTS (ENTRÉES/SORTIES)
    # ============================================
    np.random.seed(int(datetime.now().timestamp()) % 10000)
    
    mouvements = []
    for matiere in matieres_premieres_list:
        nb_entrees = np.random.randint(3, 10)
        for _ in range(nb_entrees):
            date_mvt = date_debut + timedelta(days=np.random.randint(0, (date_fin - date_debut).days + 1))
            qte = np.random.randint(10, int(matiere['Stock_Min'] * 2)) if matiere['Stock_Min'] != "Selon saison" else np.random.randint(50, 500)
            mouvements.append({
                'Date': date_mvt,
                'Matière': matiere['Matière'],
                'Type': 'Entrée',
                'Quantité': qte,
                'Prix_Unitaire': matiere['Prix_Unitaire'],
                'Valeur': qte * matiere['Prix_Unitaire'],
                'Catégorie': matiere['Type']
            })
        
        nb_sorties = np.random.randint(5, 15)
        for _ in range(nb_sorties):
            date_mvt = date_debut + timedelta(days=np.random.randint(0, (date_fin - date_debut).days + 1))
            qte = np.random.randint(5, int(matiere['Stock_Min'] * 1.5)) if matiere['Stock_Min'] != "Selon saison" else np.random.randint(20, 300)
            mouvements.append({
                'Date': date_mvt,
                'Matière': matiere['Matière'],
                'Type': 'Sortie',
                'Quantité': qte,
                'Prix_Unitaire': matiere['Prix_Unitaire'],
                'Valeur': qte * matiere['Prix_Unitaire'],
                'Catégorie': matiere['Type']
            })
    
    df_mouvements = pd.DataFrame(mouvements)
    df_mouvements = df_mouvements.sort_values('Date', ascending=False)
    
    # Application des filtres
    if type_mouvement == "Entrées":
        df_mouvements = df_mouvements[df_mouvements['Type'] == 'Entrée']
    elif type_mouvement == "Sorties":
        df_mouvements = df_mouvements[df_mouvements['Type'] == 'Sortie']
    
    # ============================================
    # KPI INVENTAIRE
    # ============================================
    st.markdown("---")
    st.markdown("### Résumé des Mouvements")
    
    total_entrees = df_mouvements[df_mouvements['Type'] == 'Entrée']['Quantité'].sum()
    total_sorties = df_mouvements[df_mouvements['Type'] == 'Sortie']['Quantité'].sum()
    valeur_entrees = df_mouvements[df_mouvements['Type'] == 'Entrée']['Valeur'].sum()
    valeur_sorties = df_mouvements[df_mouvements['Type'] == 'Sortie']['Valeur'].sum()
    solde_net = total_entrees - total_sorties
    valeur_stock = valeur_entrees - valeur_sorties
    
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    with col_k1:
        st.metric("Total Entrées", f"{total_entrees:,.0f}")
    with col_k2:
        st.metric("Total Sorties", f"{total_sorties:,.0f}")
    with col_k3:
        st.metric("Solde Net", f"{solde_net:,.0f}", 
                 delta="Positif" if solde_net >= 0 else "Négatif",
                 delta_color="normal" if solde_net >= 0 else "inverse")
    with col_k4:
        st.metric("Valeur Entrées", format_cfa(valeur_entrees))
    with col_k5:
        st.metric("Valeur Sorties", format_cfa(valeur_sorties))
    
    # ============================================
    # GRAPHIQUE ENTRÉES/SORTIES PAR JOUR
    # ============================================
    st.markdown("---")
    st.markdown("### Flux Quotidiens")
    
    mvt_par_jour = df_mouvements.groupby(['Date', 'Type'])['Quantité'].sum().reset_index()
    mvt_par_jour = mvt_par_jour.pivot(index='Date', columns='Type', values='Quantité').fillna(0)
    
    if 'Entrée' not in mvt_par_jour.columns:
        mvt_par_jour['Entrée'] = 0
    if 'Sortie' not in mvt_par_jour.columns:
        mvt_par_jour['Sortie'] = 0
    
    mvt_par_jour = mvt_par_jour.sort_index()
    
    fig_flux = go.Figure()
    
    fig_flux.add_trace(go.Bar(
        x=mvt_par_jour.index,
        y=mvt_par_jour['Entrée'],
        name='Entrées',
        marker_color='#2ca02c',
        hovertemplate='<b>%{x}</b><br>Entrées: %{y:,.0f}<extra></extra>'
    ))
    
    fig_flux.add_trace(go.Bar(
        x=mvt_par_jour.index,
        y=-mvt_par_jour['Sortie'],
        name='Sorties',
        marker_color='#d62728',
        hovertemplate='<b>%{x}</b><br>Sorties: %{y:,.0f}<extra></extra>'
    ))
    
    fig_flux.add_trace(go.Scatter(
        x=mvt_par_jour.index,
        y=mvt_par_jour['Entrée'] - mvt_par_jour['Sortie'],
        name='Solde net',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='<b>%{x}</b><br>Solde: %{y:,.0f}<extra></extra>'
    ))
    
    fig_flux.update_layout(
        title=f"Flux quotidiens du {date_debut.strftime('%d/%m/%Y')} au {date_fin.strftime('%d/%m/%Y')}",
        xaxis_title="Date",
        yaxis_title="Quantité",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        barmode='relative',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig_flux, use_container_width=True)
    
    # ============================================
    # TABLEAU DÉTAILLÉ DES MOUVEMENTS
    # ============================================
    st.markdown("---")
    st.markdown("### Détail des Mouvements")
    
    # Filtres additionnels
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        recherche_matiere = st.text_input("Rechercher une matière", "", key="rech_mat")
    with col_t2:
        categories_dispo = ['Toutes'] + sorted(df_mouvements['Catégorie'].unique().tolist())
        filtre_categorie = st.selectbox("Filtrer par catégorie", categories_dispo)
    
    df_display = df_mouvements.copy()
    
    if recherche_matiere:
        df_display = df_display[df_display['Matière'].str.contains(recherche_matiere, case=False, na=False)]
    if filtre_categorie != 'Toutes':
        df_display = df_display[df_display['Catégorie'] == filtre_categorie]
    
    df_display['Date_F'] = pd.to_datetime(df_display['Date']).dt.strftime('%d/%m/%Y')
    df_display['Valeur_F'] = df_display['Valeur'].apply(format_cfa)
    df_display['Prix_F'] = df_display['Prix_Unitaire'].apply(lambda x: f"{x:,.0f} FCFA")
    
    st.dataframe(
        df_display[['Date_F', 'Matière', 'Catégorie', 'Type', 'Quantité', 'Prix_F', 'Valeur_F']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'Date_F': 'Date',
            'Matière': 'Matière première',
            'Catégorie': 'Catégorie',
            'Type': st.column_config.Column('Type', width='small'),
            'Quantité': st.column_config.NumberColumn('Quantité', format="%,.0f"),
            'Prix_F': 'Prix unitaire',
            'Valeur_F': 'Valeur totale'
        }
    )
    
    st.caption(f"{len(df_display)} mouvements affichés")
    
    # ============================================
    # INVENTAIRE ACTUEL PAR MATIÈRE (SOLDE)
    # ============================================
    st.markdown("---")
    st.markdown("### Inventaire Actuel par Matière Première")
    
    inventaire = df_mouvements.groupby(['Matière', 'Catégorie']).agg(
        Total_Entrees=('Quantité', lambda x: x[df_mouvements['Type'] == 'Entrée'].sum()),
        Total_Sorties=('Quantité', lambda x: x[df_mouvements['Type'] == 'Sortie'].sum()),
        Valeur_Entrees=('Valeur', lambda x: x[df_mouvements['Type'] == 'Entrée'].sum()),
        Valeur_Sorties=('Valeur', lambda x: x[df_mouvements['Type'] == 'Sortie'].sum())
    ).reset_index()
    
    inventaire['Stock_Actuel'] = inventaire['Total_Entrees'] - inventaire['Total_Sorties']
    inventaire['Valeur_Stock'] = inventaire['Valeur_Entrees'] - inventaire['Valeur_Sorties']
    
    # Fusion avec les stocks minimums
    inventaire = inventaire.merge(
        df_matieres_ref[['Matière', 'Stock_Min', 'Unité']],
        on='Matière',
        how='left'
    )
    
    inventaire['Niveau'] = inventaire.apply(
        lambda row: 'Critique' if row['Stock_Actuel'] <= float(str(row['Stock_Min']).replace('Selon saison', '999999')) * 0.5 
        else ('Faible' if row['Stock_Actuel'] <= float(str(row['Stock_Min']).replace('Selon saison', '999999')) 
        else 'OK'),
        axis=1
    )
    
    inventaire['Valeur_Stock_F'] = inventaire['Valeur_Stock'].apply(format_cfa)
    inventaire = inventaire.sort_values('Stock_Actuel', ascending=True)
    
    st.dataframe(
        inventaire[['Matière', 'Catégorie', 'Unité', 'Total_Entrees', 'Total_Sorties', 'Stock_Actuel', 'Stock_Min', 'Valeur_Stock_F', 'Niveau']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'Matière': 'Matière première',
            'Catégorie': 'Catégorie',
            'Unité': 'Unité',
            'Total_Entrees': st.column_config.NumberColumn('Entrées', format="%,.0f"),
            'Total_Sorties': st.column_config.NumberColumn('Sorties', format="%,.0f"),
            'Stock_Actuel': st.column_config.NumberColumn('Stock actuel', format="%,.0f"),
            'Stock_Min': 'Stock minimum',
            'Valeur_Stock_F': 'Valeur stock',
            'Niveau': 'Niveau'
        }
    )

    
    # ============================================
    # PERFORMANCE FOURNISSEURS
    # ============================================
    st.markdown("---")
    st.markdown("### Performance des Fournisseurs")
    
    if not data['fournisseur'].empty:
        if 'NOM_FOURNISSEUR' in data['fournisseur'].columns:
            fournisseurs_noms = data['fournisseur']['NOM_FOURNISSEUR'].tolist()
            
            np.random.seed(123)
            n_fournisseurs = len(fournisseurs_noms)
            
            perf_fournisseur = pd.DataFrame({
                'Nom': fournisseurs_noms,
                'Matières_Principales': np.random.choice(
                    ['Fruits frais', 'Fruits secs', 'Sucre', 'Arômes', 'Acide citrique', 'Conservateurs', 'Eau', 'Colorants', 'Emballages'],
                    n_fournisseurs
                ),
                'Montant_Commandes': np.random.randint(500000, 50000000, n_fournisseurs),
                'Nb_Commandes': np.random.randint(5, 100, n_fournisseurs),
                'Delai_Moyen': np.random.randint(1, 15, n_fournisseurs),
                'Taux_Conformite': np.random.uniform(80, 100, n_fournisseurs)
            })
            perf_fournisseur = perf_fournisseur.sort_values('Montant_Commandes', ascending=False)
            
            fig_fourn = px.bar(
                perf_fournisseur.head(10),
                x='Nom',
                y='Montant_Commandes',
                color='Taux_Conformite',
                color_continuous_scale='RdYlGn',
                title="Top 10 fournisseurs par montant commandé",
                labels={'Montant_Commandes': 'Montant (FCFA)', 'Taux_Conformite': 'Conformité (%)'},
                hover_data=['Matières_Principales', 'Nb_Commandes', 'Delai_Moyen']
            )
            fig_fourn.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_fourn, use_container_width=True)
            
            # Tableau détaillé
            st.markdown("#### Détail des fournisseurs")
            
            perf_display = perf_fournisseur.copy()
            perf_display['Montant_Commandes_F'] = perf_display['Montant_Commandes'].apply(format_cfa)
            
            st.dataframe(
                perf_display[['Nom', 'Matières_Principales', 'Montant_Commandes_F', 'Nb_Commandes', 'Delai_Moyen', 'Taux_Conformite']].head(10),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Nom': 'Fournisseur',
                    'Matières_Principales': 'Matières fournies',
                    'Montant_Commandes_F': 'Montant commandes',
                    'Nb_Commandes': 'Nb commandes',
                    'Delai_Moyen': 'Délai moyen (j)',
                    'Taux_Conformite': st.column_config.ProgressColumn(
                        'Conformité',
                        format="%.1f%%",
                        min_value=80,
                        max_value=100
                    )
                }
            )
        else:
            st.info("Structure de la table fournisseur incomplète")
    else:
        st.info("Aucune donnée fournisseur disponible")
    

    # ============================================
    # SECTION SIMULATION : MP DISPONIBLES → PRODUCTION
    # ============================================
    st.markdown("---")
    st.markdown("### Simulation : Matières Premières → Volume de Production")
    st.caption("Estimez combien de bouteilles vous pouvez produire avec les matières premières disponibles")
    
    col_sim1, col_sim2 = st.columns([1, 1])
    
    with col_sim1:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #1f77b4;'>
        <h4 style='margin-top: 0;'> Matières Premières Disponibles</h4>
        <p style='color: #666; font-size: 0.9em;'>Renseignez les quantités actuellement en stock</p>
        """, unsafe_allow_html=True)
        
        # Champs de saisie pour chaque matière première
        stock_sucre = st.number_input("Sucre (kg)", min_value=0.0, value=500.0, step=10.0, format="%.1f")
        stock_eau = st.number_input("Eau traitée (L)", min_value=0.0, value=10000.0, step=100.0, format="%.1f")
        stock_aromes = st.number_input("Arômes naturels (L)", min_value=0.0, value=50.0, step=1.0, format="%.1f")
        stock_acide = st.number_input("Acide citrique (kg)", min_value=0.0, value=30.0, step=1.0, format="%.1f")
        stock_conservateurs = st.number_input("Conservateurs (kg)", min_value=0.0, value=25.0, step=1.0, format="%.1f")
        stock_bouteilles = st.number_input("Bouteilles PET (unités)", min_value=0, value=20000, step=100)
        stock_bouchons = st.number_input("Bouchons (unités)", min_value=0, value=20000, step=100)
        stock_etiquettes = st.number_input("Étiquettes (unités)", min_value=0, value=20000, step=100)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Calculer la production possible", type="primary", use_container_width=True, key="btn_sim_mp"):
            # Consommation par bouteille (recette standard)
            conso_par_bouteille = {
                'Sucre': 0.025,          # kg par bouteille
                'Eau traitée': 0.5,      # L par bouteille
                'Arômes naturels': 0.002, # L par bouteille
                'Acide citrique': 0.0015, # kg par bouteille
                'Conservateurs': 0.001,   # kg par bouteille
                'Bouteilles PET': 1,      # 1 bouteille
                'Bouchons': 1,            # 1 bouchon
                'Étiquettes': 1           # 1 étiquette
            }
            
            # Calcul du nombre de bouteilles possibles par matière première
            stocks = {
                'Sucre': stock_sucre,
                'Eau traitée': stock_eau,
                'Arômes naturels': stock_aromes,
                'Acide citrique': stock_acide,
                'Conservateurs': stock_conservateurs,
                'Bouteilles PET': stock_bouteilles,
                'Bouchons': stock_bouchons,
                'Étiquettes': stock_etiquettes
            }
            
            bouteilles_possibles = {}
            for mp, stock in stocks.items():
                if conso_par_bouteille[mp] > 0:
                    bouteilles_possibles[mp] = int(stock / conso_par_bouteille[mp])
                else:
                    bouteilles_possibles[mp] = float('inf')
            
            # La production est limitée par la MP la plus contraignante
            production_max = min(bouteilles_possibles.values())
            facteur_limitant = min(bouteilles_possibles, key=bouteilles_possibles.get)
            
            # Stockage dans la session
            st.session_state.simulation_mp = {
                'stocks': stocks,
                'bouteilles_possibles': bouteilles_possibles,
                'production_max': production_max,
                'facteur_limitant': facteur_limitant,
                'conso_par_bouteille': conso_par_bouteille
            }
            
            st.success(f"Simulation terminée ! Le facteur limitant est : **{facteur_limitant}**")
    
    with col_sim2:
        if 'simulation_mp' in st.session_state:
            sim = st.session_state.simulation_mp
            
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #2ca02c;'>
            <h4 style='margin-top: 0;'> Résultats de la Simulation</h4>
            """, unsafe_allow_html=True)
            
            # Résultat principal
            st.metric(
                "🍾 Production Maximale Possible",
                f"{sim['production_max']:,} bouteilles",
                help="Nombre maximum de bouteilles pouvant être produites avec les stocks actuels"
            )
            
            # Facteur limitant
            st.warning(f" **Facteur limitant :** {sim['facteur_limitant']} (stock insuffisant)")
            
            st.markdown("---")
            
            # Détail par matière première
            st.markdown("#### Détail par Matière Première")
            
            detail_data = []
            for mp, bouteilles in sim['bouteilles_possibles'].items():
                stock_dispo = sim['stocks'][mp]
                conso = sim['conso_par_bouteille'][mp]
                conso_totale = sim['production_max'] * conso
                stock_restant = stock_dispo - conso_totale
                
                if bouteilles == sim['production_max']:
                    statut = 'LIMITANT'
                elif stock_restant < stock_dispo * 0.2:
                    statut = 'Presque épuisé'
                else:
                    statut = 'OK'
                
                detail_data.append({
                    'Matière Première': mp,
                    'Stock Dispo': f"{stock_dispo:,.1f}",
                    'Conso/Bouteille': f"{conso:.4f}",
                    'Besoins Total': f"{conso_totale:,.1f}",
                    'Stock Restant': f"{stock_restant:,.1f}",
                    'Bouteilles Possibles': f"{bouteilles:,}",
                    'Statut': statut
                })
            
            df_sim_detail = pd.DataFrame(detail_data)
            st.dataframe(
                df_sim_detail,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Matière Première': 'Matière Première',
                    'Stock Dispo': 'Stock Disponible',
                    'Conso/Bouteille': 'Conso/Bouteille',
                    'Besoins Total': 'Besoins pour Production Max',
                    'Stock Restant': 'Stock Restant',
                    'Bouteilles Possibles': 'Bouteilles Possibles',
                    'Statut': 'Statut'
                }
            )
            
            st.markdown("---")
            
            # Graphique : capacité par MP
            st.markdown("#### Capacité de Production par Matière Première")
            
            fig_capa_mp = px.bar(
                x=list(sim['bouteilles_possibles'].keys()),
                y=list(sim['bouteilles_possibles'].values()),
                title="Nombre de bouteilles possibles par matière première",
                labels={'x': 'Matière Première', 'y': 'Bouteilles possibles'},
                color=[1 if k == sim['facteur_limitant'] else 0 for k in sim['bouteilles_possibles'].keys()],
                color_continuous_scale=['#2ca02c', '#d62728'],
                text=[f"{v:,}" for v in sim['bouteilles_possibles'].values()]
            )
            fig_capa_mp.add_hline(
                y=sim['production_max'], 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Maximum : {sim['production_max']:,}"
            )
            fig_capa_mp.update_traces(textposition='outside')
            fig_capa_mp.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_capa_mp, use_container_width=True)
            
            st.markdown("---")
            
            # Calcul du CA potentiel
            prix_vente_sim = st.number_input(
                "Prix de vente estimé par bouteille (FCFA)",
                min_value=100, max_value=10000, value=1000, step=100,
                key="prix_vente_sim_mp"
            )
            
            ca_potentiel = sim['production_max'] * prix_vente_sim
            st.metric("CA Potentiel avec ce stock", format_cfa(ca_potentiel))
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Renseignez les stocks disponibles et cliquez sur **Calculer la production possible**")
    
    # ============================================
    # SECTION : COMPARAISON STOCK THÉORIQUE VS RÉEL
    # ============================================
    st.markdown("---")
    st.markdown("### Relation Stock Matières Premières vs Volume de Production")
    st.caption("Analyse de la cohérence entre les matières premières disponibles et le volume de production réalisé")
    
    if not data['production'].empty:
        production_data = data['production'].copy()
        
        # Chercher la colonne de quantité produite
        qte_col_prod = None
        for col in ['QUANTITE_TOTALE', 'QUANTITE', 'QTE_PRODUITE']:
            if col in production_data.columns:
                qte_col_prod = col
                break
        
        if qte_col_prod:
            total_produit = production_data[qte_col_prod].sum()
            
            # Estimer les MP nécessaires pour cette production
            st.markdown(f"#### Production totale enregistrée : **{total_produit:,.0f} bouteilles**")
            
            # Besoins théoriques
            besoins_theoriques = {
                'Sucre': total_produit * 0.025,
                'Eau traitée': total_produit * 0.5,
                'Arômes naturels': total_produit * 0.002,
                'Acide citrique': total_produit * 0.0015,
                'Conservateurs': total_produit * 0.001,
                'Bouteilles PET': total_produit,
                'Bouchons': total_produit,
                'Étiquettes': total_produit
            }
            
            # Récupérer les stocks actuels depuis la table stock
            stock_actuel_total = data['stock']['QUANTITE'].sum() if not data['stock'].empty and 'QUANTITE' in data['stock'].columns else 0
            
            besoins_df = []
            for mp, besoin in besoins_theoriques.items():
                besoins_df.append({
                    'Matière Première': mp,
                    'Besoin Théorique': f"{besoin:,.1f}",
                    'Stock Actuel Total': f"{stock_actuel_total:,.0f}" if mp in ['Bouteilles PET', 'Bouchons', 'Étiquettes'] else "N/A (table stock)",
                })
            
            df_besoins = pd.DataFrame(besoins_df)
            st.dataframe(
                df_besoins,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Matière Première': 'Matière Première',
                    'Besoin Théorique': 'Besoin Théorique pour la Production',
                    'Stock Actuel Total': 'Stock Actuel (table stock)'
                }
            )
            
            st.info("""
            ** Note :** Les besoins théoriques sont calculés sur la base d'une recette standard.
            Pour une analyse plus précise, renseignez les stocks réels dans le simulateur ci-dessus.
            
            **Interprétation :** Si le stock actuel est inférieur au besoin théorique, cela peut indiquer :
            - Une production déjà réalisée avec ces stocks
            - Des achats de matières premières non enregistrés
            - Des écarts de recette (dosages différents)
            """)
        else:
            st.info("Colonne de quantité produite non trouvée dans la table production.")
    else:
        st.info("Aucune donnée de production disponible pour l'analyse.")
    
    # ============================================
    # EXPORT DES DONNÉES
    # ============================================
    st.markdown("---")
    st.markdown("### Export des Données")
    
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv_mouvements = df_mouvements[['Date', 'Matière', 'Catégorie', 'Type', 'Quantité', 'Prix_Unitaire', 'Valeur']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Exporter les mouvements (CSV)",
            data=csv_mouvements,
            file_name=f"mouvements_matieres_premieres_{date_debut.strftime('%Y%m%d')}_{date_fin.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    with col_exp2:
        csv_inventaire = inventaire[['Matière', 'Catégorie', 'Unité', 'Total_Entrees', 'Total_Sorties', 'Stock_Actuel', 'Stock_Min', 'Valeur_Stock', 'Niveau']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Exporter l'inventaire (CSV)",
            data=csv_inventaire,
            file_name=f"inventaire_matieres_premieres_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Export de la simulation si disponible
    if 'simulation_mp' in st.session_state:
        col_exp3, _ = st.columns(2)
        with col_exp3:
            sim_data = st.session_state.simulation_mp
            csv_sim = pd.DataFrame([
                {'Matière Première': mp, 'Stock Disponible': sim_data['stocks'][mp], 
                 'Bouteilles Possibles': sim_data['bouteilles_possibles'][mp]}
                for mp in sim_data['stocks'].keys()
            ]).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Exporter la simulation (CSV)",
                data=csv_sim,
                file_name=f"simulation_production_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# ==========================================================
# TAB 5 : RELANCE CLIENTS
# ==========================================================

if tab5 is not None:
    with tab5:
        st.markdown("## Relance des anciens clients")
        st.caption("Un client est considéré comme ancien après 6 mois d'inactivité")
        
    if not data['facture'].empty and not data['personne'].empty:
        # Fusion factures et personnes
        if 'TELEPHONE' in data['personne'].columns:
            facture_client = data['facture'].merge(
                data['personne'][['ID_PERSONNE', 'NOM', 'TELEPHONE']], 
                on='ID_PERSONNE', 
                how='left'
            )
        else:
            facture_client = data['facture'].merge(
                data['personne'][['ID_PERSONNE', 'NOM']], 
                on='ID_PERSONNE', 
                how='left'
            )
        
        # Dernier achat par client
        if 'MONTANT_NET' in facture_client.columns and 'DATE_CREATION' in facture_client.columns:
            derniers_achats = facture_client.groupby('ID_PERSONNE').agg({
                'DATE_CREATION': 'max',
                'MONTANT_NET': ['sum', 'count'],
                'NOM': 'first'
            }).reset_index()
            
            derniers_achats.columns = ['ID_PERSONNE', 'Dernier_Achat', 'CA_Total', 'Nb_Achats', 'Nom']
            
            # Jours depuis dernier achat
            derniers_achats['Jours_Depuis'] = (pd.Timestamp.now() - derniers_achats['Dernier_Achat']).dt.days
            
            # SEGMENTATION CORRIGÉE - Client ancien = > 6 mois (180 jours) 
            derniers_achats['Segment'] = pd.cut(
                derniers_achats['Jours_Depuis'],
                bins=[0, 30, 90, 180, 365, float('inf')],
                labels=[
                    'Actif (< 1 mois)', 
                    'En veille (1-3 mois)', 
                    'Inactif (3-6 mois)', 
                    'Ancien (6-12 mois)',
                    'Très ancien (> 12 mois)'
                ]
            )
            
            # Clients à relancer : PLUS DE 6 MOIS (180 jours)
            clients_relance = derniers_achats[
                derniers_achats['Jours_Depuis'] > 180
            ].copy()
            
            # Score de priorité de relance
            if not clients_relance.empty:
                clients_relance['Score_Priorite'] = (
                    (clients_relance['CA_Total'] / clients_relance['CA_Total'].max() * 40) +
                    (clients_relance['Nb_Achats'] / clients_relance['Nb_Achats'].max() * 30) +
                    (clients_relance['Jours_Depuis'] / clients_relance['Jours_Depuis'].max() * 30)
                ).round(0)
            
            # KPI relance
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total clients", len(derniers_achats))
            with col2:
                st.metric("Anciens (> 6 mois)", len(clients_relance))
            with col3:
                ca_potentiel = clients_relance['CA_Total'].sum() if not clients_relance.empty else 0
                st.metric("CA historique perdu", format_cfa(ca_potentiel))
            with col4:
                pct_anciens = len(clients_relance) / len(derniers_achats) * 100 if len(derniers_achats) > 0 else 0
                st.metric("% Anciens", f"{pct_anciens:.1f}%")
            
            # Distribution par segment
            st.markdown("### Répartition des clients par ancienneté")
            
            segment_counts = derniers_achats['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Nombre']
            
            # Définir l'ordre des segments
            ordre_segments = [
                'Actif (< 1 mois)', 
                'En veille (1-3 mois)', 
                'Inactif (3-6 mois)', 
                'Ancien (6-12 mois)', 
                'Très ancien (> 12 mois)'
            ]
            segment_counts['Segment'] = pd.Categorical(
                segment_counts['Segment'], 
                categories=ordre_segments, 
                ordered=True
            )
            segment_counts = segment_counts.sort_values('Segment')
            
            colors_segments = {
                'Actif (< 1 mois)': COLORS['success'],
                'En veille (1-3 mois)': COLORS['olive'],
                'Inactif (3-6 mois)': COLORS['warning'],
                'Ancien (6-12 mois)': COLORS['danger'],
                'Très ancien (> 12 mois)': COLORS['gray']
            }
            
            fig = px.pie(
                segment_counts,
                values='Nombre',
                names='Segment',
                title="Segmentation des clients par ancienneté",
                color='Segment',
                color_discrete_map=colors_segments,
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Clients anciens à relancer (inactifs depuis plus de 6 mois)")
            
            if not clients_relance.empty:
                clients_relance['Dernier_Achat_F'] = clients_relance['Dernier_Achat'].dt.strftime('%d/%m/%Y')
                clients_relance['CA_Total_F'] = clients_relance['CA_Total'].apply(format_cfa)
                clients_relance = clients_relance.sort_values('Score_Priorite', ascending=False)
                
                st.dataframe(
                    clients_relance[[
                        'Nom', 'Dernier_Achat_F', 'Jours_Depuis', 
                        'CA_Total_F', 'Nb_Achats', 'Score_Priorite', 'Segment'
                    ]].head(30),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Nom': 'Client',
                        'Dernier_Achat_F': 'Dernier achat',
                        'Jours_Depuis': 'Jours inactif',
                        'CA_Total_F': 'CA Total',
                        'Nb_Achats': 'Nb achats',
                        'Score_Priorite': st.column_config.ProgressColumn(
                            'Priorité relance',
                            format="%.0f",
                            min_value=0,
                            max_value=100
                        ),
                        'Segment': 'Statut'
                    }
                )
                
                # Export
                csv_relance = clients_relance[[
                    'Nom', 'Dernier_Achat_F', 'Jours_Depuis', 
                    'CA_Total', 'Nb_Achats', 'Score_Priorite'
                ]].to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Exporter la liste des clients à relancer",
                    data=csv_relance,
                    file_name=f"relance_clients_6mois_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.success("Aucun client ancien à relancer ! Tous vos clients sont actifs dans les 6 derniers mois.")
            
            # Stratégie de relance
            st.markdown("### Stratégie de relance")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                **Inactifs (3-6 mois)**
                - Email personnalisé
                - Offre spéciale
                - Relance WhatsApp
                """)
            with col2:
                st.markdown("""
                **Anciens (6-12 mois)**
                - Appel téléphonique
                - Offre spéciale retour
                - Sondage satisfaction
                """)
            with col3:
                st.markdown("""
                **Très anciens (> 12 mois)**
                - Campagne SMS massive
                - Offre exceptionnelle
                - Programme de parrainage
                """)
        else:
            st.info("Colonnes manquantes pour l'analyse clients")
    else:
        st.info("Données clients insuffisantes")

# ==========================================================
# TAB 6 : CARTOGRAPHIE & OPTIMISATION TOURNÉES
# ==========================================================
if tab6 is not None:
    with tab6:
        st.markdown("## Cartographie & Optimisation des tournées de livraison")
        st.caption("Calcul des coûts de transport · Optimisation des itinéraires · Rentabilité par zone")

        # ============================================
        # PARAMÈTRES DU VÉHICULE
    # ============================================
    st.markdown("### Configuration du véhicule de livraison")
    
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    
    with col_v1:
        type_vehicule = st.selectbox(
            "Type de véhicule",
            ["Toyota Hiace (Diesel)", "Toyota Hiace (Essence)", "Toyota Hilux", "Camion 5T", "Camion 10T", "Moto", "Personnalisé"],
            index=0,
            help="Sélectionnez le véhicule utilisé pour les livraisons"
        )
    
    # Données réelles de consommation et prix au Cameroun
    conso_vehicules = {
        "Toyota Hiace (Diesel)": {"conso_100km": 9.5, "carburant": "Gasoil", "prix_litre": 630, "capacite": 1500},
        "Toyota Hiace (Essence)": {"conso_100km": 10.5, "carburant": "Super", "prix_litre": 650, "capacite": 1500},
        "Toyota Hilux": {"conso_100km": 12.0, "carburant": "Gasoil", "prix_litre": 630, "capacite": 1000},
        "Camion 5T": {"conso_100km": 18.0, "carburant": "Gasoil", "prix_litre": 630, "capacite": 5000},
        "Camion 10T": {"conso_100km": 25.0, "carburant": "Gasoil", "prix_litre": 630, "capacite": 10000},
        "Moto": {"conso_100km": 3.5, "carburant": "Super", "prix_litre": 650, "capacite": 100},
        "Personnalisé": {"conso_100km": 10.0, "carburant": "Gasoil", "prix_litre": 630, "capacite": 1500}
    }
    
    vehicule_info = conso_vehicules[type_vehicule]
    
    with col_v2:
        conso_100km = st.number_input(
            "Consommation (L/100km)",
            min_value=1.0, max_value=50.0,
            value=vehicule_info["conso_100km"], step=0.5,
            help=f"Consommation réelle du {type_vehicule}"
        )
    
    with col_v3:
        prix_carburant = st.number_input(
            f"Prix {vehicule_info['carburant']} (FCFA/L)",
            min_value=400, max_value=1000,
            value=vehicule_info["prix_litre"], step=10,
            help=f"Prix officiel {vehicule_info['carburant']} au Cameroun"
        )
    
    with col_v4:
        cout_entretien_km = st.number_input(
            "Entretien (FCFA/km)",
            min_value=10, max_value=500,
            value=75, step=5,
            help="Coût entretien : pneus, vidange, réparations"
        )
    
    # Calcul du coût kilométrique
    cout_carburant_km = (conso_100km / 100) * prix_carburant
    cout_total_km = cout_carburant_km + cout_entretien_km
    
    # Afficher le récapitulatif
    st.info(f"""
    **Coûts de transport pour {type_vehicule} ({vehicule_info['carburant']}) :**
    - Coût carburant : **{cout_carburant_km:,.0f} FCFA/km** ({conso_100km}L/100km × {prix_carburant} FCFA/L)
    - Coût entretien : **{cout_entretien_km:,.0f} FCFA/km**
    - Coût total kilométrique : **{cout_total_km:,.0f} FCFA/km**
    - Pour 100 km : **{conso_100km * prix_carburant:,.0f} FCFA** de carburant
    """)
    
    # ============================================
    # LOCALISATION DE L'USINE (ODZA)
    # ============================================
    USINE = {"nom": "Usine Odza", "lat": 3.9110, "lon": 11.5250}
    
    # Zones de livraison avec distances réelles depuis Odza
    zones_livraison = {
        'Odza': (3.9110, 11.5250, 0),
        'Ekounou': (3.8650, 11.5300, 4),
        'Nlongkak': (3.8950, 11.5200, 5),
        'Bastos': (3.9000, 11.5100, 6),
        'Mfoundi': (3.8800, 11.5100, 7),
        'Etoudi': (3.8950, 11.5150, 8),
        'Mokolo': (3.8900, 11.5050, 9),
        'Akwa': (3.8800, 11.5150, 10),
        'Mvan': (3.8330, 11.4950, 12),
        'Ngousso': (3.9200, 11.5400, 14),
        'Biyem-Assi': (3.8500, 11.4900, 15),
        'Bonamoussadi': (3.9500, 11.5050, 16),
        'Mendong': (3.8450, 11.4800, 18),
        'Nkolbisson': (3.8700, 11.4650, 20),
        'Oyom-Abang': (3.8700, 11.4750, 22),
    }
    
    # ============================================
    # ANALYSE DES VENTES PAR ZONE
    # ============================================
    st.markdown("---")
    st.markdown("### Analyse des ventes par zone de livraison")
    
    if not facture_filtered.empty and 'ID_PERSONNE' in facture_filtered.columns:
        np.random.seed(42)
        
        if 'ADRESSE' not in facture_filtered.columns:
            zones_noms = list(zones_livraison.keys())
            facture_filtered['ZONE'] = np.random.choice(
                zones_noms, len(facture_filtered),
                p=[0.08, 0.06, 0.05, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.05, 0.08, 0.07, 0.06, 0.03, 0.03]
            )
        
        ventes_zone = facture_filtered.groupby('ZONE').agg({
            'MONTANT_NET': ['sum', 'mean', 'count'],
            'ID_PERSONNE': 'nunique'
        }).reset_index()
        ventes_zone.columns = ['Zone', 'CA_Total', 'CA_Moyen', 'Nb_Commandes', 'Nb_Clients']
        ventes_zone = ventes_zone.sort_values('CA_Total', ascending=False)
        
        ventes_zone['Distance_km'] = ventes_zone['Zone'].map(lambda z: zones_livraison.get(z, (0,0,10))[2])
        ventes_zone['Cout_Carburant_AR'] = ventes_zone['Distance_km'] * 2 * cout_carburant_km
        ventes_zone['Cout_Total_AR'] = ventes_zone['Distance_km'] * 2 * cout_total_km
        ventes_zone['Cout_Par_Livraison'] = ventes_zone['Cout_Total_AR'] / ventes_zone['Nb_Commandes'].replace(0, 1)
        ventes_zone['Rentabilite'] = ventes_zone['CA_Moyen'] - ventes_zone['Cout_Par_Livraison']
        ventes_zone['Marge_Transport'] = ((ventes_zone['CA_Moyen'] - ventes_zone['Cout_Par_Livraison']) / ventes_zone['CA_Moyen'] * 100).round(1)
        
        # ============================================
        # CARTE INTERACTIVE (CORRIGÉE)
        # ============================================
        st.markdown("### Carte des zones de livraison")
        
        ventes_zone['LAT'] = ventes_zone['Zone'].map(lambda z: zones_livraison.get(z, (0,0,0))[0])
        ventes_zone['LON'] = ventes_zone['Zone'].map(lambda z: zones_livraison.get(z, (0,0,0))[1])
        ventes_zone['TAILLE'] = (ventes_zone['CA_Total'] / ventes_zone['CA_Total'].max() * 40 + 15).fillna(20)
        
        fig_map = go.Figure()
        
        # Usine Odza
        fig_map.add_trace(go.Scattermapbox(
            lat=[USINE['lat']], lon=[USINE['lon']],
            mode='markers+text',
            marker=dict(size=25, color='red'),
            text=['ODZA'],
            textposition='top center',
            textfont=dict(size=12, color='darkred', family='Arial Black'),
            name='Usine',
            hovertemplate='<b>Usine Odza</b><br>Point de départ<extra></extra>'
        ))
        
        # Lignes de trajet
        for _, zone_row in ventes_zone.iterrows():
            fig_map.add_trace(go.Scattermapbox(
                lat=[USINE['lat'], zone_row['LAT']],
                lon=[USINE['lon'], zone_row['LON']],
                mode='lines',
                line=dict(width=1, color='rgba(31, 119, 180, 0.35)'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Zones de livraison (colorbar corrigée)
        fig_map.add_trace(go.Scattermapbox(
            lat=ventes_zone['LAT'],
            lon=ventes_zone['LON'],
            mode='markers+text',
            marker=dict(
                size=ventes_zone['TAILLE'],
                color=ventes_zone['Rentabilite'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(
                    title="Rentabilité<br>(FCFA)",
                    thickness=15,
                    len=0.7
                )
            ),
            text=ventes_zone['Zone'],
            textposition='top center',
            textfont=dict(size=9),
            hovertemplate=(
                '<b>%{text}</b><br>'
                'CA: %{customdata[0]:,.0f} FCFA<br>'
                'Distance: %{customdata[1]} km<br>'
                'Coût A/R: %{customdata[2]:,.0f} FCFA<br>'
                'Rentabilité: %{customdata[3]:,.0f} FCFA<br>'
                'Marge: %{customdata[4]:.1f}%'
                '<extra></extra>'
            ),
            customdata=ventes_zone[['CA_Total', 'Distance_km', 'Cout_Total_AR', 'Rentabilite', 'Marge_Transport']].values,
            name='Zones'
        ))
        
        fig_map.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=3.8850, lon=11.5100),
                zoom=11.5
            ),
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # ============================================
        # CALCULS ÉCONOMIQUES
        # ============================================
        st.markdown("---")
        st.markdown("### Analyse économique des tournées")
        
        nb_total_livraisons = ventes_zone['Nb_Commandes'].sum()
        cout_total_carburant = (ventes_zone['Cout_Carburant_AR'] * ventes_zone['Nb_Commandes']).sum()
        distance_totale = (ventes_zone['Distance_km'] * 2 * ventes_zone['Nb_Commandes']).sum()
        carburant_total = distance_totale * (conso_100km / 100)
        
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        with col_e1:
            st.metric("Coût carburant total", format_cfa(cout_total_carburant))
        with col_e2:
            st.metric("Distance totale", f"{distance_totale:,.0f} km")
        with col_e3:
            st.metric("Carburant consommé", f"{carburant_total:,.0f} L")
        with col_e4:
            st.metric("Livraisons", f"{nb_total_livraisons:,}")
        
        # ============================================
        # TABLEAU DÉTAILLÉ
        # ============================================
        st.markdown("### Rentabilité par zone")
        
        def classify(row):
            m = row['Marge_Transport']
            if m > 80: return 'Excellente'
            elif m > 60: return 'Bonne'
            elif m > 40: return 'Acceptable'
            elif m > 0: return 'Faible'
            else: return 'Négative'
        
        ventes_zone['Statut'] = ventes_zone.apply(classify, axis=1)
        
        display = ventes_zone.copy()
        display['CA_Total_F'] = display['CA_Total'].apply(format_cfa)
        display['Cout_AR_F'] = display['Cout_Total_AR'].apply(format_cfa)
        display['Rentabilite_F'] = display['Rentabilite'].apply(format_cfa)
        
        st.dataframe(
            display[['Zone', 'Distance_km', 'Nb_Commandes', 'CA_Total_F', 'Cout_AR_F', 'Rentabilite_F', 'Marge_Transport', 'Statut']].sort_values('Distance_km'),
            use_container_width=True, hide_index=True,
            column_config={
                'Distance_km': st.column_config.NumberColumn('km', format="%.0f"),
                'Marge_Transport': st.column_config.NumberColumn('Marge', format="%.1f%%")
            }
        )
        
        # ============================================
        # OPTIMISATION
        # ============================================
        st.markdown("---")
        st.markdown("### Tournées optimisées")
        
        zones_triees = ventes_zone.sort_values('Distance_km')
        tournees = []
        deja_vu = set()
        
        for _, zone in zones_triees.iterrows():
            if zone['Zone'] in deja_vu: continue
            groupe = [zone]
            deja_vu.add(zone['Zone'])
            
            for _, z2 in zones_triees.iterrows():
                if z2['Zone'] not in deja_vu:
                    if abs(zone['Distance_km'] - z2['Distance_km']) <= 5:
                        groupe.append(z2)
                        deja_vu.add(z2['Zone'])
            
            if groupe:
                dmax = max(g['Distance_km'] for g in groupe)
                cout_carb = dmax * 2 * cout_carburant_km
                cout_indiv = sum(g['Cout_Carburant_AR'] for g in groupe)
                
                tournees.append({
                    'Tournée': f"T{len(tournees)+1}",
                    'Zones': ' → '.join(g['Zone'] for g in groupe),
                    'Livraisons': sum(g['Nb_Commandes'] for g in groupe),
                    'Km_max': dmax,
                    'Coût_Carburant': cout_carb,
                    'Économie': cout_indiv - cout_carb
                })
        
        if tournees:
            df_t = pd.DataFrame(tournees)
            df_t['Coût_F'] = df_t['Coût_Carburant'].apply(format_cfa)
            df_t['Économie_F'] = df_t['Économie'].apply(format_cfa)
            
            st.dataframe(
                df_t[['Tournée', 'Zones', 'Livraisons', 'Km_max', 'Coût_F', 'Économie_F']],
                use_container_width=True, hide_index=True,
                column_config={
                    'Km_max': st.column_config.NumberColumn('Km max', format="%.0f km"),
                    'Coût_F': 'Coût carburant',
                    'Économie_F': 'Économie'
                }
            )
            
            eco = df_t['Économie'].sum()
            st.success(f"Économie totale : {format_cfa(eco)} en regroupant les tournées")
        
        # ============================================
        # SIMULATEUR
        # ============================================
        st.markdown("---")
        st.markdown("### Simulateur de tournée")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            zones_sel = st.multiselect("Zones", ventes_zone['Zone'].tolist(), default=ventes_zone.head(3)['Zone'].tolist())
        with c2:
            nb_liv = st.number_input("Nb livraisons", 1, 50, 5)
        with c3:
            km_extra = st.number_input("Km détours", 0.0, 20.0, 5.0, 0.5)
        
        if zones_sel:
            dmax = max(zones_livraison.get(z, (0,0,10))[2] for z in zones_sel)
            dtotal = dmax * 2 + km_extra
            cout_carb = dtotal * cout_carburant_km
            cout_total = dtotal * cout_total_km
            cout_liv = cout_total / nb_liv
            
            st.markdown(f"""
            | Paramètre | Valeur |
            |-----------|--------|
            | Zones | {' → '.join(zones_sel)} |
            | Distance | {dtotal:.0f} km |
            | Carburant | {dtotal * conso_100km / 100:.1f} L ({format_cfa(cout_carb)}) |
            | Coût total | {format_cfa(cout_total)} |
            | Par livraison | {format_cfa(cout_liv)} |
            | Seuil rentabilité | {format_cfa(cout_liv * 1.5)} |
            """)
        
        # ============================================
        # RECOMMANDATIONS
        # ============================================
        st.markdown("---")
        st.markdown("### Recommandations")
        
        c1 = st.columns(1)[0]
        with c1:
            st.markdown(f"""
            **Pour le {type_vehicule} :**
            - Consommation : {conso_100km} L/100km
            - Coût/km : {cout_total_km:,.0f} FCFA
            - Plein 70L : {70 * prix_carburant:,.0f} FCFA
            - Autonomie : {70 / conso_100km * 100:.0f} km
            """)
        
        st.info("""
        ** Astuce économie :** Un plein de 70L à 630 FCFA/L coûte 44 100 FCFA et permet de parcourir environ 737 km.
        En regroupant les livraisons, vous pouvez réduire la consommation de 25-30%.
        """)
    else:
        st.warning("Données insuffisantes pour l'analyse géographique")

# ==========================================================
# TAB 7 : ANALYSE PRODUITS & FRUITS
# ==========================================================

if tab7 is not None:
    with tab7:
        st.markdown("## Analyse Produits & Fruits N'NAM")
        st.caption("Top ventes · Saisonnalité · Matières premières · Prédictions stocks fruits · Rentabilité")
        
    # Déterminer la colonne nom produit disponible
    if not data['produit'].empty:
        cols_produit = data['produit'].columns.tolist()
        if 'NOM_PRODUIT' in cols_produit:
            nom_produit_col_ref = 'NOM_PRODUIT'
        elif 'DESIGNATION' in cols_produit:
            nom_produit_col_ref = 'DESIGNATION'
        elif 'NOM' in cols_produit:
            nom_produit_col_ref = 'NOM'
        else:
            for col in cols_produit:
                if col != 'ID_PRODUIT' and data['produit'][col].dtype == 'object':
                    nom_produit_col_ref = col
                    break
            else:
                nom_produit_col_ref = 'ID_PRODUIT'
    else:
        nom_produit_col_ref = 'ID_PRODUIT'
    
    # Sous-onglets simplifiés (4 au lieu de beaucoup plus)
    subtab_p1, subtab_p2, subtab_p3 = st.tabs([
        "Top Produits",
        "Matières Premières",
        "Rentabilité"
    ])
    
    # ============================================
    # SOUS-TAB 1 : SAISONS & TOP PRODUITS
    # ============================================
    with subtab_p1:
        st.markdown("### Gestion des Fruits & Produits par Saison")
        st.caption("Filtres fruits secs · Top ventes par mois/saison · Prédictions stocks")
        
        # ============================================
        # FILTRES FRUITS
        # ============================================
        st.markdown("---")
        st.markdown("#### Filtres Fruits")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            type_fruit = st.radio(
                "Type de fruits à afficher",
                ["Tous les fruits", "Fruits secs uniquement", "Fruits frais uniquement"],
                index=0,
                help="Les fruits frais ne peuvent PAS être stockés longtemps. Seuls les fruits secs sont stockables."
            )
        with col_f2:
            mois_selected = st.selectbox(
                "Mois d'analyse",
                ["Toute l'année"] + [MOIS_FR[i] for i in range(1, 13)],
                index=0
            )
        with col_f3:
            saison_selected = st.selectbox(
                "Saison",
                ["Toutes"] + list(SAISONS.keys()),
                index=0
            )
        
        # ============================================
        # CLASSIFICATION FRUITS FRAIS VS SECS
        # ============================================
        fruits_classification = {
            # Fruits FRAIS (périssables, ne peuvent PAS être stockés)
            'Ananas': {'type': 'Frais', 'conservation_max': '5-7 jours', 'stockable': False},
            'Mangue': {'type': 'Frais', 'conservation_max': '3-5 jours', 'stockable': False},
            'Goyave': {'type': 'Frais', 'conservation_max': '3-4 jours', 'stockable': False},
            'Papaye': {'type': 'Frais', 'conservation_max': '5-7 jours', 'stockable': False},
            'Pastèque': {'type': 'Frais', 'conservation_max': '7-10 jours', 'stockable': False},
            'Fruit de la passion': {'type': 'Frais', 'conservation_max': '7-10 jours', 'stockable': False},
            'Citron': {'type': 'Frais', 'conservation_max': '21 jours', 'stockable': False},
            'Orange': {'type': 'Frais', 'conservation_max': '14 jours', 'stockable': False},
            
            # Fruits SECS (longue conservation, stockables)
            'Baobab': {'type': 'Sec', 'conservation_max': '180 jours (poudre)', 'stockable': True},
            'Hibiscus/Bissap': {'type': 'Sec', 'conservation_max': '365 jours (séché)', 'stockable': True},
            'Tamarin': {'type': 'Sec', 'conservation_max': '90 jours', 'stockable': True},
            'Gingembre': {'type': 'Sec', 'conservation_max': '30 jours', 'stockable': True},
        }
        
        # Fruits N'NAM avec leurs données saisonnières
        fruits_nnam = {
            'Ananas': {
                'type': 'Frais', 'saison': [3,4,5,6,7], 'pic': [5,6],
                'prix_saison': 200, 'prix_hors_saison': 450,
                'conservation': '5-7 jours', 'conso_mensuelle_pic': 5000, 'conso_mensuelle_creux': 1500,
                'stockable': False
            },
            'Citron': {
                'type': 'Frais', 'saison': [11,12,1,2], 'pic': [12,1],
                'prix_saison': 150, 'prix_hors_saison': 350,
                'conservation': '21 jours', 'conso_mensuelle_pic': 3000, 'conso_mensuelle_creux': 800,
                'stockable': False
            },
            'Gingembre': {
                'type': 'Sec', 'saison': list(range(1,13)), 'pic': [3,4,9,10],
                'prix_saison': 500, 'prix_hors_saison': 500,
                'conservation': '30 jours', 'conso_mensuelle_pic': 1500, 'conso_mensuelle_creux': 1200,
                'stockable': True
            },
            'Orange': {
                'type': 'Frais', 'saison': [11,12,1,2], 'pic': [12,1],
                'prix_saison': 180, 'prix_hors_saison': 400,
                'conservation': '14 jours', 'conso_mensuelle_pic': 4000, 'conso_mensuelle_creux': 1000,
                'stockable': False
            },
            'Mangue': {
                'type': 'Frais', 'saison': [3,4,5,6], 'pic': [4,5],
                'prix_saison': 250, 'prix_hors_saison': 600,
                'conservation': '3-5 jours', 'conso_mensuelle_pic': 3500, 'conso_mensuelle_creux': 800,
                'stockable': False
            },
            'Goyave': {
                'type': 'Frais', 'saison': [3,4,5,6,7,8], 'pic': [5,6],
                'prix_saison': 180, 'prix_hors_saison': 400,
                'conservation': '3-4 jours', 'conso_mensuelle_pic': 2500, 'conso_mensuelle_creux': 600,
                'stockable': False
            },
            'Fruit de la passion': {
                'type': 'Frais', 'saison': [3,4,5,6,7,8,9], 'pic': [5,6,7],
                'prix_saison': 300, 'prix_hors_saison': 700,
                'conservation': '7-10 jours', 'conso_mensuelle_pic': 2000, 'conso_mensuelle_creux': 500,
                'stockable': False
            },
            'Papaye': {
                'type': 'Frais', 'saison': list(range(1,13)), 'pic': [3,4,9,10],
                'prix_saison': 200, 'prix_hors_saison': 350,
                'conservation': '5-7 jours', 'conso_mensuelle_pic': 2000, 'conso_mensuelle_creux': 1000,
                'stockable': False
            },
            'Pastèque': {
                'type': 'Frais', 'saison': [1,2,3,4,5,12], 'pic': [2,3,4],
                'prix_saison': 350, 'prix_hors_saison': 700,
                'conservation': '7-10 jours', 'conso_mensuelle_pic': 1800, 'conso_mensuelle_creux': 400,
                'stockable': False
            },
            'Baobab': {
                'type': 'Sec', 'saison': list(range(1,13)), 'pic': [1,2,3],
                'prix_saison': 1000, 'prix_hors_saison': 1000,
                'conservation': '180 jours (poudre)', 'conso_mensuelle_pic': 500, 'conso_mensuelle_creux': 400,
                'stockable': True
            },
            'Hibiscus/Bissap': {
                'type': 'Sec', 'saison': [11,12,1,2,3], 'pic': [12,1],
                'prix_saison': 800, 'prix_hors_saison': 1200,
                'conservation': '365 jours (séché)', 'conso_mensuelle_pic': 800, 'conso_mensuelle_creux': 300,
                'stockable': True
            },
            'Tamarin': {
                'type': 'Sec', 'saison': [1,2,3,4,12], 'pic': [2,3],
                'prix_saison': 600, 'prix_hors_saison': 900,
                'conservation': '90 jours', 'conso_mensuelle_pic': 600, 'conso_mensuelle_creux': 200,
                'stockable': True
            },
        }
        
        # Appliquer le filtre fruits secs/frais
        if type_fruit == "Fruits secs uniquement":
            fruits_filtered = {k: v for k, v in fruits_nnam.items() if v['type'] == 'Sec'}
            st.info(" **Filtre actif : Fruits secs uniquement** - Ces fruits peuvent être stockés (longue conservation).")
        elif type_fruit == "Fruits frais uniquement":
            fruits_filtered = {k: v for k, v in fruits_nnam.items() if v['type'] == 'Frais'}
            st.warning(" **Filtre actif : Fruits frais uniquement** - Ces fruits NE PEUVENT PAS être stockés (périssables).")
        else:
            fruits_filtered = fruits_nnam
        
        # ============================================
        # TABLEAU DE CLASSIFICATION
        # ============================================
        st.markdown("---")
        st.markdown("### Classification des Fruits")
        
        classif_data = []
        for fruit, info in fruits_filtered.items():
            classif_data.append({
                'Fruit': fruit,
                'Type': 'Fruit SEC' if info['type'] == 'Sec' else 'Fruit FRAIS',
                'Stockable': 'OUI' if info['stockable'] else 'NON',
                'Conservation': info['conservation'],
                'Prix saison': f"{info['prix_saison']:,.0f} FCFA",
                'Prix hors-saison': f"{info['prix_hors_saison']:,.0f} FCFA",
                'Mois de saison': ', '.join([MOIS_ABBR[m] for m in info['saison']])
            })
        
        df_classif = pd.DataFrame(classif_data)
        st.dataframe(
            df_classif,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Fruit': 'Fruit',
                'Type': 'Type',
                'Stockable': 'Stockable ?',
                'Conservation': 'Conservation max',
                'Prix saison': 'Prix en saison',
                'Prix hors-saison': 'Prix hors saison',
                'Mois de saison': 'Mois de disponibilité'
            }
        )
        
        # ============================================
        # TOP PRODUITS PAR MOIS/SAISON
        # ============================================
        st.markdown("---")
        st.markdown("### Produits les Plus Vendus par Mois et par Saison")
        
        # Simulation des ventes par produit et par mois
        np.random.seed(42)
        produits_nnam = [
            "Jus d'Ananas N'NAM", "Jus de Mangue N'NAM", "Jus de Goyave N'NAM",
            "Jus de Papaye N'NAM", "Jus de Pastèque N'NAM", "Jus de Fruit de la Passion N'NAM",
            "Jus de Citron N'NAM", "Jus d'Orange N'NAM", "Jus de Gingembre N'NAM",
            "Jus de Baobab N'NAM", "Jus de Bissap N'NAM", "Jus de Tamarin N'NAM"
        ]
        
        # Générer des ventes mensuelles simulées
        ventes_produits = []
        for mois in range(1, 13):
            for produit in produits_nnam:
                # Les ventes varient selon la saison du fruit principal
                fruit_name = produit.replace("Jus de ", "").replace("Jus d'", "").replace(" N'NAM", "").replace(" de la ", " ")
                fruit_info = fruits_nnam.get(fruit_name, fruits_nnam.get('Ananas'))
                
                if fruit_info and mois in fruit_info.get('saison', [1,2,3]):
                    ventes = np.random.randint(500, 3000)
                else:
                    ventes = np.random.randint(100, 800)
                
                ventes_produits.append({
                    'Mois': mois,
                    'Mois_Nom': MOIS_FR[mois],
                    'Saison': get_saison(mois),
                    'Produit': produit,
                    'Ventes': ventes,
                    'Type_Fruit': 'Sec' if fruit_name in ['Gingembre', 'Baobab', 'Bissap', 'Tamarin'] else 'Frais'
                })
        
        df_ventes = pd.DataFrame(ventes_produits)
        
        # Filtre par mois
        if mois_selected != "Toute l'année":
            mois_num = [k for k, v in MOIS_FR.items() if v == mois_selected][0]
            df_ventes = df_ventes[df_ventes['Mois'] == mois_num]
        
        # Filtre par saison
        if saison_selected != "Toutes":
            df_ventes = df_ventes[df_ventes['Saison'] == saison_selected]
        
        # Filtre par type de fruit
        if type_fruit == "Fruits secs uniquement":
            df_ventes = df_ventes[df_ventes['Type_Fruit'] == 'Sec']
        elif type_fruit == "Fruits frais uniquement":
            df_ventes = df_ventes[df_ventes['Type_Fruit'] == 'Frais']
        
        # Top 10 produits
        top_produits = df_ventes.groupby('Produit')['Ventes'].sum().sort_values(ascending=False).head(10)
        
        col_top1, col_top2 = st.columns([1, 1])
        
        with col_top1:
            st.markdown("#### Top 10 des produits les plus vendus")
            
            fig_top = px.bar(
                x=top_produits.values,
                y=top_produits.index,
                orientation='h',
                title="Top 10 produits",
                color=top_produits.values,
                color_continuous_scale='Blues',
                text=top_produits.values
            )
            fig_top.update_traces(textposition='outside')
            fig_top.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col_top2:
            st.markdown("#### Ventes mensuelles du Top 5")
            
            top5_list = top_produits.head(5).index.tolist()
            df_top5 = df_ventes[df_ventes['Produit'].isin(top5_list)]
            
            fig_lines = px.line(
                df_top5.groupby(['Mois_Nom', 'Produit'])['Ventes'].sum().reset_index(),
                x='Mois_Nom',
                y='Ventes',
                color='Produit',
                title="Évolution mensuelle du Top 5",
                markers=True
            )
            fig_lines.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_lines, use_container_width=True)
        
        # ============================================
        # TABLEAU DÉTAILLÉ PAR MOIS
        # ============================================
        st.markdown("---")
        st.markdown("### Détail des Ventes par Produit et par Mois")
        
        pivot_ventes = df_ventes.pivot_table(
            values='Ventes',
            index='Produit',
            columns='Mois_Nom',
            aggfunc='sum',
            fill_value=0
        )
        
        # Ordonner les colonnes par mois
        mois_ordre = [MOIS_FR[i] for i in range(1, 13)]
        pivot_ventes = pivot_ventes[[m for m in mois_ordre if m in pivot_ventes.columns]]
        
        pivot_ventes['Total'] = pivot_ventes.sum(axis=1)
        pivot_ventes = pivot_ventes.sort_values('Total', ascending=False)
        pivot_ventes['Moyenne/Mois'] = pivot_ventes['Total'] / 12
        
        # Ajouter le type de fruit
        pivot_ventes['Type'] = pivot_ventes.index.map(
            lambda x: 'SEC' if any(f in x for f in ['Gingembre', 'Baobab', 'Bissap', 'Tamarin']) else '🍊 FRAIS'
        )
        
        st.dataframe(
            pivot_ventes,
            use_container_width=True,
            column_config={
                'Produit': 'Produit',
                'Total': st.column_config.NumberColumn('Total annuel', format="%,.0f"),
                'Moyenne/Mois': st.column_config.NumberColumn('Moyenne/mois', format="%,.0f"),
                'Type': 'Type'
            }
        )
        
        st.caption(" **Lecture :** Les cases vides = pas de ventes ce mois-là. Les fruits secs (🥜) sont les seuls stockables.")
        
        # ============================================
        # SAISONNALITÉ & PRÉDICTIONS STOCKS FRUITS
        # ============================================
        st.markdown("---")
        st.markdown("### Saisonnalité & Prédictions Stocks de Fruits")
        
        mois_actuel = datetime.now().month
        
        # Statut actuel
        st.markdown(f"#### Statut Actuel - {MOIS_FR[mois_actuel]} {datetime.now().year}")
        
        status_data = []
        for fruit, info in fruits_filtered.items():
            en_saison = mois_actuel in info['saison']
            pic = mois_actuel in info['pic']
            
            if pic:
                statut = 'PIC'
                prix = info['prix_saison']
                conso = info['conso_mensuelle_pic']
            elif en_saison:
                statut = 'En saison'
                prix = info['prix_saison']
                conso = info['conso_mensuelle_pic']
            else:
                statut = 'Hors saison'
                prix = info['prix_hors_saison']
                conso = info['conso_mensuelle_creux']
            
            stock_recommande = conso / 2
            
            status_data.append({
                'Fruit': fruit,
                'Type': 'SEC' if info['type'] == 'Sec' else 'FRAIS',
                'Stockable': 'oui' if info['stockable'] else 'non',
                'Statut': statut,
                'Prix_Actuel': prix,
                'Conso_Mensuelle': conso,
                'Stock_Recommande': stock_recommande,
                'Conservation': info['conservation']
            })
        
        df_status = pd.DataFrame(status_data)
        
        # Résumé
        en_saison_count = len(df_status[df_status['Statut'].str.contains('En saison|PIC')])
        en_pic = len(df_status[df_status['Statut'] == 'PIC'])
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1: st.metric("En saison", en_saison_count)
        with col_s2: st.metric("PIC", en_pic)
        with col_s3: st.metric("Hors saison", len(df_status) - en_saison_count)
        with col_s4: st.metric("Stockables", len(df_status[df_status['Stockable'] == 'oui']))
        
        st.dataframe(
            df_status[['Fruit', 'Type', 'Stockable', 'Statut', 'Prix_Actuel', 'Conso_Mensuelle', 'Stock_Recommande', 'Conservation']],
            use_container_width=True, hide_index=True,
            column_config={
                'Fruit': 'Fruit',
                'Type': 'Type',
                'Stockable': 'Stockable',
                'Statut': 'Statut',
                'Prix_Actuel': st.column_config.NumberColumn('Prix actuel (FCFA)', format="%,.0f"),
                'Conso_Mensuelle': st.column_config.NumberColumn('Conso./mois', format="%,.0f"),
                'Stock_Recommande': st.column_config.NumberColumn('Stock recommandé', format="%,.0f"),
                'Conservation': 'Conservation'
            }
        )
        
        # Note importante pour les fruits frais
        fruits_non_stockables = [f for f, info in fruits_filtered.items() if not info['stockable']]
        if fruits_non_stockables and type_fruit != "Fruits secs uniquement":
            st.error(f"""
             **ATTENTION : Les fruits suivants NE SONT PAS STOCKABLES :**
            {', '.join(fruits_non_stockables)}
            
            Ces fruits frais doivent être transformés rapidement après achat pour éviter les pertes.
            Seuls les fruits secs (Baobab, Bissap, Tamarin, Gingembre) peuvent être conservés en stock.
            """)
    # ============================================
    # SOUS-TAB 2 : MATIÈRES PREMIÈRES
    # ============================================
    with subtab_p2:
        st.markdown("### Matières Premières & Fournitures")
        
        matieres = {
            'Sucre': {'unite': 'kg', 'conso': 2500, 'stock_min': 500, 'cout': 450, 'fournisseur': 'Distributeur local'},
            'Eau traitée': {'unite': 'L', 'conso': 50000, 'stock_min': 10000, 'cout': 5, 'fournisseur': 'CAMWATER'},
            'Arômes naturels': {'unite': 'L', 'conso': 200, 'stock_min': 50, 'cout': 5000, 'fournisseur': 'Importateur'},
            'Acide citrique': {'unite': 'kg', 'conso': 150, 'stock_min': 30, 'cout': 2000, 'fournisseur': 'Fournisseur chimique'},
            'Conservateurs': {'unite': 'kg', 'conso': 100, 'stock_min': 25, 'cout': 3000, 'fournisseur': 'Fournisseur chimique'},
            'Bouteilles PET': {'unite': 'u', 'conso': 100000, 'stock_min': 20000, 'cout': 75, 'fournisseur': 'Plastique Cam'},
            'Bouchons': {'unite': 'u', 'conso': 100000, 'stock_min': 20000, 'cout': 15, 'fournisseur': 'Plastique Cam'},
            'Étiquettes': {'unite': 'u', 'conso': 100000, 'stock_min': 20000, 'cout': 25, 'fournisseur': 'Imprimerie'},
            'Cartons': {'unite': 'u', 'conso': 5000, 'stock_min': 1000, 'cout': 200, 'fournisseur': 'Cartonnerie'},
            'Sachets': {'unite': 'u', 'conso': 30000, 'stock_min': 8000, 'cout': 10, 'fournisseur': 'Plastique Cam'},
        }
        
        df_mat = pd.DataFrame([
            {'Matière': n, 'Unité': i['unite'], 'Conso/Mois': i['conso'],
             'Stock Min': i['stock_min'], 'Coût Unitaire': i['cout'],
             'Coût Mensuel': i['conso']*i['cout'], 'Fournisseur': i['fournisseur']}
            for n, i in matieres.items()
        ])
        df_mat['Coût_F'] = df_mat['Coût Mensuel'].apply(format_cfa)
        
        fig = px.treemap(df_mat, path=['Matière'], values='Coût Mensuel',
                        title="Répartition coûts matières premières", color='Coût Mensuel',
                        color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            df_mat[['Matière', 'Unité', 'Conso/Mois', 'Stock Min', 'Coût_F', 'Fournisseur']],
            use_container_width=True, hide_index=True
        )
        
        st.metric("Coût total matières premières/mois", format_cfa(df_mat['Coût Mensuel'].sum()))
    
# ==========================================================
# TAB 8 : STOCKS & MAGASINS (CORRIGÉ - DESIGNATION DANS STOCK)
# ==========================================================

if tab8 is not None:
    with tab8:
        st.markdown("## Analyse des Stocks par Magasin")
        st.caption("Inventaire complet · Niveaux critiques · Comparaison magasins")
        
        if not data['stock'].empty and not data['magasin'].empty:
            # Fusion stock avec magasins
            stock_complet = data['stock'].copy()
            stock_complet = stock_complet.merge(data['magasin'], on='ID_MAGASIN', how='left')
            
        # ============================================
        # DÉTERMINER LA COLONNE DE DÉSIGNATION
        # ============================================
        # La désignation est dans la table STOCK (colonne DESIGNATION)
        if 'DESIGNATION' in stock_complet.columns:
            nom_produit_col = 'DESIGNATION'
        elif 'NOM_PRODUIT' in stock_complet.columns:
            nom_produit_col = 'NOM_PRODUIT'
        elif 'NOM' in stock_complet.columns:
            nom_produit_col = 'NOM'
        else:
            # Chercher une colonne texte qui pourrait être la désignation
            for col in stock_complet.columns:
                if col != 'ID_PRODUIT' and col != 'ID_MAGASIN' and stock_complet[col].dtype == 'object':
                    if stock_complet[col].str.contains('JUS|N\'NAM|ANANAS|CITRON', case=False, na=False).any():
                        nom_produit_col = col
                        break
            else:
                nom_produit_col = 'ID_PRODUIT'
                st.warning("Aucune colonne de désignation trouvée, utilisation de l'ID produit")
        
        # Colonne nom magasin
        nom_magasin_col = 'NOM_MAGASIN' if 'NOM_MAGASIN' in stock_complet.columns else 'ID_MAGASIN'
        
        # Vérifier la colonne quantité
        if 'QUANTITE' not in stock_complet.columns:
            for col in ['QTE', 'STOCK', 'QUANTITY', 'QUANTITE_STOCK']:
                if col in stock_complet.columns:
                    stock_complet['QUANTITE'] = stock_complet[col]
                    break
            else:
                st.error("Colonne de quantité introuvable dans la table stock")
                st.write("Colonnes disponibles :", stock_complet.columns.tolist())
                st.stop()
        
        # Remplir les désignations vides
        stock_complet[nom_produit_col] = stock_complet[nom_produit_col].fillna(
            'Produit #' + stock_complet['ID_PRODUIT'].astype(str)
        )
        
        # ============================================
        # KPI GLOBAUX
        # ============================================
        nb_magasins = stock_complet[nom_magasin_col].nunique()
        nb_produits_total = stock_complet[nom_produit_col].nunique()
        stock_global = stock_complet['QUANTITE'].sum()
        valeur_globale = stock_global * 500
        
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.metric("Magasins", nb_magasins)
        with col_k2:
            st.metric("Produits différents", nb_produits_total)
        with col_k3:
            st.metric("Stock global", f"{stock_global:,.0f}")
        with col_k4:
            st.metric("Valeur estimée", format_cfa(valeur_globale))
        
        st.markdown("---")
        
        # ============================================
        # CATÉGORISATION DES PRODUITS
        # ============================================
        def categoriser_produit(designation):
            designation = str(designation).upper()
            if any(mot in designation for mot in ['ANANAS', 'MANGUE', 'GOYAVE', 'PAPAYE', 'PASSION', 'PASTEQUE', 'PASTÈQUE', 'COROSSOL', 'BAOBAB', 'TAMARIN', 'NONI', 'ACÉROLA', 'CERISE', 'PRUNE', 'BANANE', 'NOIX DE COCO', 'ANACARDE', 'KOLA']):
                return 'Jus Fruits Tropicaux'
            elif any(mot in designation for mot in ['CITRON', 'ORANGE', 'PAMPLEMOUSSE', 'MANDARINE', 'CITRONNELLE']):
                return 'Jus Agrumes'
            elif any(mot in designation for mot in ['GINGEMBRE', 'GINGER', 'MENTHE', 'MIEL', 'CURCUMA', 'SPIRULINE', 'MORINGA', 'CLOU DE GIROFLE', 'CANNELLE', 'VANILLE', 'CACAO']):
                return 'Jus Bien-être'
            elif any(mot in designation for mot in ['BISSAP', 'HIBISCUS', 'FLEUR', 'ROSELLE']):
                return 'Jus Floraux'
            elif any(mot in designation for mot in ['EAU', 'WATER']):
                return 'Eau'
            elif any(mot in designation for mot in ['SIROP', 'CONCENTRÉ', 'CONCENTRE']):
                return 'Sirops'
            elif any(mot in designation for mot in ['PET', 'BOUTEILLE', 'BOUCHON', 'ETIQUETTE', 'ÉTIQUETTE', 'CARTON', 'SACHET', 'EMBALLAGE', 'FILM', 'PALETTE']):
                return 'Emballages'
            elif any(mot in designation for mot in ['SUCRE', 'ARÔME', 'AROME', 'ACIDE', 'CONSERVATEUR', 'COLORANT']):
                return 'Matières Premières'
            else:
                return 'Autres produits'
        
        stock_complet['Catégorie'] = stock_complet[nom_produit_col].apply(categoriser_produit)
        
        # ============================================
        # SÉLECTION DU MAGASIN
        # ============================================
        st.markdown("### Inventaire par magasin")
        
        magasins_dispo = stock_complet[nom_magasin_col].unique()
        
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            magasin_selected = st.selectbox("Sélectionner un magasin", magasins_dispo)
        with col_m2:
            vue = st.radio("Vue", ["Résumé", "Inventaire complet", "Alertes stock"], horizontal=True)
        
        stock_magasin = stock_complet[stock_complet[nom_magasin_col] == magasin_selected].copy()
        
        if not stock_magasin.empty:
            # KPI
            nb_produits_mag = len(stock_magasin)
            stock_total_mag = stock_magasin['QUANTITE'].sum()
            valeur_stock_mag = stock_total_mag * 500
            produits_zero = len(stock_magasin[stock_magasin['QUANTITE'] == 0])
            produits_critiques = len(stock_magasin[stock_magasin['QUANTITE'] < 10])
            
            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
            with col_s1: st.metric("Produits", nb_produits_mag)
            with col_s2: st.metric("Unités", f"{stock_total_mag:,.0f}")
            with col_s3: st.metric("Valeur", format_cfa(valeur_stock_mag))
            with col_s4: st.metric("Rupture (0)", produits_zero, delta_color="inverse")
            with col_s5: st.metric("Critique (<10)", produits_critiques, delta_color="inverse")
            
            st.markdown("---")
            
            # ============================================
            # VUE RÉSUMÉ
            # ============================================
            if vue == "Résumé":
                st.markdown(f"#### Composition du stock - {magasin_selected}")
                
                composition = stock_magasin.groupby('Catégorie').agg({
                    'QUANTITE': 'sum',
                    nom_produit_col: 'count'
                }).reset_index()
                composition.columns = ['Catégorie', 'Quantité', 'Nb_Produits']
                composition = composition.sort_values('Quantité', ascending=True)
                
                col_comp1, col_comp2 = st.columns([1, 1])
                
                with col_comp1:
                    fig_pie = px.pie(
                        composition, values='Quantité', names='Catégorie',
                        title="Répartition par catégorie", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_pie.update_layout(height=350)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_comp2:
                    fig_bar = px.bar(
                        composition, x='Quantité', y='Catégorie', orientation='h',
                        title="Volume par catégorie", color='Quantité',
                        color_continuous_scale='Blues',
                        text=composition['Quantité'].apply(lambda x: f"{x:,.0f}")
                    )
                    fig_bar.update_traces(textposition='outside')
                    fig_bar.update_layout(height=350, yaxis_title="")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Top 10
                st.markdown("#### Top 10 produits en stock")
                top10 = stock_magasin.nlargest(10, 'QUANTITE')[[nom_produit_col, 'QUANTITE', 'Catégorie']]
                
                fig_top = px.bar(
                    top10, x='QUANTITE', y=nom_produit_col, orientation='h',
                    title="Top 10 produits", color='Catégorie',
                    text=top10['QUANTITE'].apply(lambda x: f"{x:,.0f}"),
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_top.update_traces(textposition='outside')
                fig_top.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_top, use_container_width=True)
            
            # ============================================
            # VUE INVENTAIRE COMPLET
            # ============================================
            elif vue == "Inventaire complet":
                st.markdown(f"#### Inventaire détaillé - {magasin_selected}")
                st.caption("Désignations réelles des produits en stock")
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    recherche = st.text_input("Rechercher", "", help="Tapez une partie du nom")
                with col_f2:
                    filtre_cat = st.selectbox("Filtrer par catégorie", ["Toutes"] + sorted(stock_magasin['Catégorie'].unique().tolist()))
                
                stock_display = stock_magasin[[nom_produit_col, 'QUANTITE', 'Catégorie']].copy()
                
                if recherche:
                    stock_display = stock_display[stock_display[nom_produit_col].astype(str).str.contains(recherche, case=False, na=False)]
                if filtre_cat != "Toutes":
                    stock_display = stock_display[stock_display['Catégorie'] == filtre_cat]
                
                stock_display = stock_display.sort_values('QUANTITE', ascending=False)
                
                seuil_bas = stock_magasin['QUANTITE'].quantile(0.15) if len(stock_magasin) > 5 else 10
                stock_display['Niveau'] = stock_display['QUANTITE'].apply(
                    lambda x: 'Critique' if x <= seuil_bas else (' Faible' if x <= seuil_bas*3 else ' OK')
                )
                stock_display['% Max'] = (stock_display['QUANTITE'] / max(stock_display['QUANTITE'].max(), 1) * 100).round(0)
                
                st.dataframe(
                    stock_display,
                    use_container_width=True, hide_index=True,
                    column_config={
                        nom_produit_col: 'Désignation produit',
                        'QUANTITE': st.column_config.NumberColumn('Quantité', format="%,.0f"),
                        'Catégorie': 'Catégorie',
                        'Niveau': 'Niveau',
                        '% Max': st.column_config.ProgressColumn('% du max', format="%.0f%%", min_value=0, max_value=100)
                    }
                )
                
                st.caption(f"{len(stock_display)} produits | Seuil critique: ≤{seuil_bas:.0f} | Colonne: `{nom_produit_col}`")
            
            # ============================================
            # VUE ALERTES
            # ============================================
            else:
                st.markdown(f"#### Produits en alerte - {magasin_selected}")
                
                seuil_alerte = st.slider("Seuil d'alerte", 0, 100, 20, 5)
                
                produits_alerte = stock_magasin[stock_magasin['QUANTITE'] <= seuil_alerte].sort_values('QUANTITE')
                
                col_a1, col_a2 = st.columns(2)
                with col_a1: st.metric("Rupture (0)", len(produits_alerte[produits_alerte['QUANTITE'] == 0]))
                with col_a2: st.metric(f"Sous {seuil_alerte}", len(produits_alerte))
                
                if not produits_alerte.empty:
                    produits_alerte['Action'] = produits_alerte['QUANTITE'].apply(
                        lambda x: 'URGENT' if x == 0 else ('Réappro.' if x < 10 else 'Surveiller')
                    )
                    
                    st.dataframe(
                        produits_alerte[[nom_produit_col, 'Catégorie', 'QUANTITE', 'Action']],
                        use_container_width=True, hide_index=True,
                        column_config={
                            nom_produit_col: 'Désignation produit',
                            'Catégorie': 'Catégorie',
                            'QUANTITE': st.column_config.NumberColumn('Stock', format="%.0f"),
                            'Action': 'Action'
                        }
                    )
                    
                    csv_alerte = produits_alerte[[nom_produit_col, 'Catégorie', 'QUANTITE', 'Action']].to_csv(index=False).encode('utf-8')
                    st.download_button("Exporter CSV", csv_alerte, f"alerte_stock_{datetime.now():%Y%m%d}.csv")
                else:
                    st.success(f"Aucun produit sous {seuil_alerte} unités")
        
        # ============================================
        # COMPARAISON MAGASINS
        # ============================================
        st.markdown("---")
        st.markdown("### Comparaison tous magasins")
        
        stock_par_mag = stock_complet.groupby(nom_magasin_col).agg({
            'QUANTITE': 'sum',
            nom_produit_col: 'nunique'
        }).reset_index()
        stock_par_mag.columns = ['Magasin', 'Quantité', 'Nb_Produits']
        stock_par_mag = stock_par_mag.sort_values('Quantité', ascending=False)
        
        fig_comp = px.bar(
            stock_par_mag, x='Magasin', y='Quantité', color='Quantité',
            color_continuous_scale='Viridis', title="Stock par magasin",
            text=stock_par_mag['Quantité'].apply(lambda x: f"{x:,.0f}")
        )
        fig_comp.update_traces(textposition='outside')
        fig_comp.update_layout(height=400)
        st.plotly_chart(fig_comp, use_container_width=True)

# ==========================================================
# TAB 9 : DICTIONNAIRE DE DONNÉES
# ==========================================================
if tab9 is not None:
    with tab9:
        st.markdown("## Dictionnaire des Données & Indicateurs")
        st.caption("Guide de référence pour comprendre tous les indicateurs, leurs formules et leur interprétation")

        # ============================================
        # SECTION 1 : GLOSSAIRE DES TERMES
    # ============================================
    st.markdown("---")
    st.markdown("### Glossaire des Termes")
    
    glossaire = {
        "Terme": [
            "**CA** (Chiffre d'Affaires)",
            "**Client actif**",
            "**Commande**",
            "**Panier moyen client**",
            "**Stock total**",
            "**Taux de perte**",
            "**Seuil critique**",
            "**MAPE**",
            "**R²**",
            "**Coefficient de saisonnalité**",
            "**Seuil de rentabilité**",
            "**Marge transport**",
            "**Couverture stock**",
            "**Saison des pluies**",
            "**Saison sèche**",
            "**Matière première**",
            "**Fruit sec**",
            "**Fruit frais**",
        ],
        "Définition": [
            "Montant total des ventes réalisées sur une période donnée, calculé à partir des factures clients (MONTANT_NET).",
            "Client ayant effectué au moins un achat dans la période analysée. Un même client peut générer plusieurs commandes.",
            "Ensemble des factures liées à un même client sur une période. Un client = une ou plusieurs factures regroupées.",
            "Montant moyen dépensé par client sur la période. Calcul : CA Total ÷ Nombre de clients actifs.",
            "Quantité totale de produits disponibles dans l'ensemble des magasins, toutes références confondues.",
            "Pourcentage de produits perdus lors de la production par rapport à la quantité totale produite.",
            "Niveau de stock en dessous duquel un produit est considéré en risque de rupture. Calculé au 15ème percentile des stocks.",
            "Mean Absolute Percentage Error - Erreur moyenne en pourcentage des prévisions ML par rapport aux valeurs réelles.",
            "Coefficient de détermination - Indique la qualité du modèle prédictif (0 = mauvais, 1 = parfait).",
            "Variation périodique prévisible des ventes liée aux saisons. Permet d'anticiper les périodes de forte/faible activité.",
            "Niveau de CA minimum à atteindre pour couvrir l'ensemble des charges (fixes + variables).",
            "Pourcentage du CA restant après déduction des coûts de transport. Indique la rentabilité d'une zone de livraison.",
            "Durée pendant laquelle le stock actuel peut couvrir les ventes sans réapprovisionnement.",
            "Période d'avril à octobre caractérisée par des précipitations plus fréquentes au Cameroun.",
            "Période de novembre à mars caractérisée par un temps plus sec au Cameroun.",
            "Ingrédient utilisé dans la fabrication des jus (fruits, sucre, eau, arômes, additifs, etc.).",
            "Fruit déshydraté pouvant être stocké longtemps (baobab, bissap séché, tamarin).",
            "Fruit périssable nécessitant une transformation rapide (ananas, mangue, pastèque, etc.).",
        ]
    }
    
    df_glossaire = pd.DataFrame(glossaire)
    st.dataframe(
        df_glossaire,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Terme': 'Terme',
            'Définition': st.column_config.TextColumn('Définition', width='large')
        }
    )
    
    # ============================================
    # SECTION 2 : INDICATEURS DE PERFORMANCE (KPI)
    # ============================================
    st.markdown("---")
    st.markdown("### Indicateurs de Performance (KPI)")
    
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <b> Fréquence de mise à jour :</b> Les KPI sont recalculés à chaque chargement de la page (temps réel).<br>
        <b> Période par défaut :</b> Année en cours (modifiable via le filtre dans la barre latérale).
    </div>
    """, unsafe_allow_html=True)
    
    kpi_data = {
        "Indicateur": [
            "**Chiffre d'Affaires (CA)**",
            "**Clients actifs**",
            "**Nombre de commandes**",
            "**CA par client**",
            "**CA par commande**",
            "**Stock total**",
            "**Taux de perte**",
        ],
        "Formule de calcul": [
            "`SUM(facture_client.MONTANT_NET)` sur la période filtrée",
            "`COUNT(DISTINCT facture_client.ID_PERSONNE)` sur la période filtrée",
            "`COUNT(facture_client.ID_PERSONNE)` groupé par client",
            "`CA Total ÷ Nombre de clients actifs`",
            "`CA Total ÷ Nombre de commandes`",
            "`SUM(stock.QUANTITE)` tous magasins confondus",
            "`(SUM(production.PERTES_TOTALES) ÷ SUM(production.QUANTITE_TOTALE)) × 100`",
        ],
        "Interprétation": [
            "Plus le CA est élevé, meilleure est la performance commerciale. À comparer avec les objectifs et l'historique.",
            "Indique la taille de la base client active. Une baisse peut signaler un problème de fidélisation.",
            "Montre l'intensité d'achat. Un ratio commandes/clients élevé indique une bonne fidélité.",
            "Valeur moyenne dépensée par client. À surveiller pour détecter les évolutions de comportement d'achat.",
            "Montant moyen par acte d'achat. Permet d'optimiser les stratégies de vente croisée.",
            "Niveau global des stocks. Doit être suffisant pour couvrir la demande sans générer de surstock.",
            "Doit rester < 5%. Un taux élevé indique des problèmes de production ou de qualité des matières premières.",
        ],
        "Seuil d'alerte": [
            "Baisse > 20% vs même période N-1",
            "Baisse > 15% vs mois précédent",
            "Ratio < 1.5 commandes/client",
            "Baisse > 10% vs période précédente",
            "Écart type élevé = forte dispersion",
            "Couverture < 2 mois",
            "> 5% : action corrective nécessaire",
        ]
    }
    
    df_kpi = pd.DataFrame(kpi_data)
    st.dataframe(
        df_kpi,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Indicateur': 'Indicateur',
            'Formule de calcul': st.column_config.TextColumn('Formule de calcul', width='medium'),
            'Interprétation': st.column_config.TextColumn('Interprétation', width='large'),
            'Seuil d\'alerte': st.column_config.TextColumn('Seuil d\'alerte', width='medium')
        }
    )
    
    # ============================================
    # SECTION 3 : INDICATEURS PRÉDICTIFS
    # ============================================
    st.markdown("---")
    st.markdown("### Indicateurs Prédictifs (Machine Learning)")
    
    ml_data = {
        "Indicateur": [
            "**MAE** (Mean Absolute Error)",
            "**MAPE** (Mean Absolute Percentage Error)",
            "**R²** (Coefficient de détermination)",
            "**Tendance**",
            "**Prévisions 6 mois**",
            "**Zone de confiance**",
        ],
        "Définition / Calcul": [
            "Écart moyen entre les valeurs prédites et réelles, exprimé en FCFA.",
            "Erreur moyenne en pourcentage : `(1/n) × Σ(|y_réel - y_pred| / y_réel) × 100`",
            "Proportion de la variance expliquée par le modèle : `1 - (SS_res / SS_tot)`",
            "Direction générale des ventes : HAUSSIÈRE si prévision > historique, BAISSIÈRE sinon.",
            "Projection du CA pour les 6 prochains mois via Random Forest.",
            "Intervalle ± MAE autour des prévisions, représentant la marge d'erreur probable.",
        ],
        "Interprétation": [
            "Plus il est bas, plus le modèle est précis. Exprimé dans la même unité que le CA.",
            "MAPE < 10% = excellent, 10-20% = bon, 20-50% = acceptable, > 50% = imprécis.",
            "R² > 0.80 : modèle fiable. 0.50-0.80 : utilisable avec prudence. < 0.50 : à améliorer.",
            "Orientation stratégique : adapter les approvisionnements et le marketing selon la tendance.",
            "Base pour la planification des stocks et des objectifs commerciaux.",
            "Plus la zone est étroite, plus la prévision est précise.",
        ]
    }
    
    df_ml = pd.DataFrame(ml_data)
    st.dataframe(
        df_ml,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Indicateur': 'Indicateur',
            'Définition / Calcul': st.column_config.TextColumn('Définition / Calcul', width='large'),
            'Interprétation': st.column_config.TextColumn('Interprétation', width='large')
        }
    )
    
    # ============================================
    # SECTION 4 : INDICATEURS SAISONNALITÉ
    # ============================================
    st.markdown("---")
    st.markdown("### Indicateurs de Saisonnalité")
    
    st.markdown("""
    | Concept | Définition | Application |
    |---------|-----------|-------------|
    | **Coefficient de saisonnalité** | Ratio qui mesure l'écart entre les ventes d'une période et la moyenne annuelle | Si > 1 : période à fort potentiel ; Si < 1 : période à faible potentiel |
    | **Mois de pic** | Mois où le CA est le plus élevé de l'année | Anticiper les stocks et la production 1 à 2 mois avant |
    | **Mois creux** | Mois où le CA est le plus bas de l'année | Période propice aux promotions et à la maintenance |
    | **Saison des pluies** | Avril à Octobre | Adapter la logistique (routes difficiles), privilégier les fruits de saison |
    | **Saison sèche** | Novembre à Mars | Période généralement plus favorable aux ventes et aux livraisons |
    """)
    
    # Tableau des coefficients de saisonnalité par mois
    st.markdown("#### Coefficients de Saisonnalité par Mois")
    
    if not facture_filtered.empty and 'MONTANT_NET' in facture_filtered.columns:
        ca_par_mois = facture_filtered.groupby('MOIS')['MONTANT_NET'].sum()
        ca_moyen_mensuel = ca_par_mois.mean()
        
        coeff_saison = pd.DataFrame({
            'Mois': [MOIS_FR.get(i, i) for i in range(1, 13)],
            'Coefficient': [round(ca_par_mois.get(i, 0) / ca_moyen_mensuel, 2) if ca_moyen_mensuel > 0 else 0 for i in range(1, 13)]
        })
        
        coeff_saison['Interprétation'] = coeff_saison['Coefficient'].apply(
            lambda x: 'Fort potentiel' if x > 1.2 
            else ('Potentiel moyen' if x >= 0.8 
            else ('Faible potentiel' if x > 0 else 'Pas de données'))
        )
        
        st.dataframe(
            coeff_saison,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Mois': 'Mois',
                'Coefficient': st.column_config.NumberColumn('Coefficient de saisonnalité', format="%.2f"),
                'Interprétation': 'Interprétation'
            }
        )
        
        st.caption("""
        **Lecture :** Un coefficient de 1.20 signifie que les ventes du mois sont 20% supérieures à la moyenne mensuelle.
        Un coefficient de 0.70 signifie qu'elles sont 30% inférieures.
        """)
    
    # ============================================
    # SECTION 5 : INDICATEURS GÉOGRAPHIQUES
    # ============================================
    st.markdown("---")
    st.markdown("### Indicateurs Géographiques")
    
    geo_data = {
        "Indicateur": [
            "**CA par zone**",
            "**Distance (km)**",
            "**Coût carburant A/R**",
            "**Coût par livraison**",
            "**Rentabilité transport**",
            "**Marge transport**",
        ],
        "Formule": [
            "`SUM(facture_client.MONTANT_NET)` groupé par zone",
            "Distance en km depuis l'usine d'Odza jusqu'à la zone de livraison",
            "`Distance × 2 × (Conso/100 × Prix carburant)`",
            "`Coût total A/R ÷ Nombre de livraisons`",
            "`CA moyen par commande − Coût par livraison`",
            "`(Rentabilité ÷ CA moyen) × 100`",
        ],
        "Interprétation": [
            "Identifie les zones les plus lucratives pour concentrer les efforts commerciaux.",
            "Détermine les coûts logistiques. Au-delà de 20 km, envisager un point de distribution intermédiaire.",
            "Coût direct du carburant pour un aller-retour. Base de la tarification des livraisons.",
            "Doit être inférieur à 30% du CA moyen pour rester rentable.",
            "Si positif : la zone est rentable. Si négatif : revoir la stratégie de livraison.",
            "> 80% = excellent, 60-80% = bon, 40-60% = acceptable, < 40% = à améliorer.",
        ]
    }
    
    df_geo = pd.DataFrame(geo_data)
    st.dataframe(
        df_geo,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Indicateur': 'Indicateur',
            'Formule': st.column_config.TextColumn('Formule de calcul', width='medium'),
            'Interprétation': st.column_config.TextColumn('Interprétation', width='large')
        }
    )
    
    # ============================================
    # SECTION 6 : INDICATEURS CLIENTS
    # ============================================
    st.markdown("---")
    st.markdown("### Indicateurs Clients & Relance")
    
    st.markdown("""
    | Segment | Définition | Jours d'inactivité | Action recommandée |
    |---------|-----------|-------------------|-------------------|
    | **Actif** | A acheté récemment | 0-30 jours | Fidélisation, programme de parrainage |
    | **En veille** | Ralentissement d'activité | 31-90 jours | Email personnalisé, offre spéciale |
    | **Inactif** | N'achète plus régulièrement | 91-180 jours | Relance WhatsApp, promotion ciblée |
    | **Ancien** | À réactiver | 181-365 jours | Appel téléphonique, offre de retour |
    | **Très ancien** | Probablement perdu | > 365 jours | Campagne SMS, offre exceptionnelle |
    
    **Score de priorité de relance** = 40% CA historique + 30% fréquence d'achat + 30% ancienneté
    """)
    
    # ============================================
    # SECTION 7 : SOURCES DE DONNÉES
    # ============================================
    st.markdown("---")
    st.markdown("### Sources de Données")
    
    sources_data = {
        "Table source": [
            "**facture_client**",
            "**stock**",
            "**sortie**",
            "**production**",
            "**fournisseur**",
            "**personne**",
            "**magasin**",
            "**produit**",
        ],
        "Contenu": [
            "Factures clients : ID, date, montant, client associé",
            "État des stocks par produit et par magasin",
            "Mouvements de sortie de stock",
            "Données de production : quantités produites, pertes",
            "Informations sur les fournisseurs",
            "Données clients : nom, téléphone, adresse",
            "Liste des magasins/points de vente",
            "Catalogue produits : nom, prix, caractéristiques",
        ],
        "Colonnes clés": [
            "ID_FACTURE, DATE_CREATION, MONTANT_NET, ID_PERSONNE",
            "ID_PRODUIT, ID_MAGASIN, QUANTITE, DESIGNATION",
            "ID_SORTIE, DATE_SORTIE, ID_PRODUIT, QUANTITE",
            "ID_PRODUCTION, QUANTITE_TOTALE, PERTES_TOTALES",
            "ID_FOURNISSEUR, NOM_FOURNISSEUR",
            "ID_PERSONNE, NOM, TELEPHONE, ADRESSE",
            "ID_MAGASIN, NOM_MAGASIN",
            "ID_PRODUIT, NOM_PRODUIT",
        ],
        "Fréquence MAJ": [
            "Temps réel (chaque vente)",
            "Temps réel (chaque mouvement)",
            "Temps réel (chaque sortie)",
            "Quotidienne",
            "Mensuelle",
            "À chaque nouveau client",
            "Rare (création magasin)",
            "À chaque nouveau produit",
        ]
    }
    
    df_sources = pd.DataFrame(sources_data)
    st.dataframe(
        df_sources,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Table source': 'Table source',
            'Contenu': st.column_config.TextColumn('Contenu', width='medium'),
            'Colonnes clés': st.column_config.TextColumn('Colonnes clés', width='large'),
            'Fréquence MAJ': 'Fréquence de mise à jour'
        }
    )
    
    # ============================================
    # SECTION 8 : RÈGLES DE GESTION
    # ============================================
    st.markdown("---")
    st.markdown("### Règles de Gestion")
    
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <b> Règles importantes à respecter pour garantir des analyses fiables :</b>
    </div>
    """, unsafe_allow_html=True)
    
    regles_data = {
        "Règle": [
            "Unicité des clients",
            "Période d'analyse",
            "Saisonnalité",
            "Seuil de stock critique",
            "Fruits frais vs secs",
            "Pertes de production",
            "Données manquantes",
            "Prévisions ML",
        ],
        "Description": [
            "Un client est identifié par son ID_PERSONNE. Les doublons de noms ne sont pas autorisés.",
            "Par défaut, l'année en cours. Les comparaisons doivent se faire sur des périodes équivalentes.",
            "Basée sur le calendrier camerounais : saison des pluies (avril-octobre), saison sèche (novembre-mars).",
            "Calculé au 15ème percentile des stocks. Tout produit en dessous nécessite un réapprovisionnement.",
            "Seuls les fruits secs (longue conservation) peuvent être stockés. Les fruits frais doivent être transformés rapidement.",
            "Le taux de perte acceptable est < 5%. Au-delà, une analyse des causes est nécessaire.",
            "Les valeurs nulles ou manquantes sont exclues des calculs pour ne pas fausser les moyennes.",
            "Nécessitent au minimum 6 mois de données historiques pour être fiables. La marge d'erreur (MAE) est systématiquement affichée.",
        ],
        "Impact si non respectée": [
            "Risque de doublons dans les analyses clients et les relances.",
            "Comparaisons faussées, tendances incorrectes.",
            "Mauvaise anticipation des stocks, ruptures ou surstock.",
            "Rupture de stock non détectée, perte de ventes.",
            "Pertes de fruits frais si stockés trop longtemps.",
            "Masquage de problèmes de production, coûts cachés.",
            "Moyennes biaisées, décisions basées sur des données incomplètes.",
            "Décisions stratégiques basées sur des prévisions peu fiables.",
        ]
    }
    
    df_regles = pd.DataFrame(regles_data)
    st.dataframe(
        df_regles,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Règle': 'Règle',
            'Description': st.column_config.TextColumn('Description', width='large'),
            'Impact si non respectée': st.column_config.TextColumn('Impact si non respectée', width='large')
        }
    )
    
    # ============================================
    # SECTION 9 : NOTES D'UTILISATION
    # ============================================
    st.markdown("---")
    st.markdown("### Notes d'Utilisation")
    
    col_n1, col_n2 = st.columns(2)
    
    with col_n1:
        st.info("""
        ** Comment lire le tableau de bord :**
        
        1. **Filtres** (barre latérale) : sélectionnez l'année, les magasins et fournisseurs
        2. **KPI** (haut de page) : vue d'ensemble de la performance
        3. **Onglets** : explorez chaque analyse thématique
        4. **Graphiques** : survolez pour voir les détails
        5. **Tableaux** : téléchargeables en CSV
        """)
    
    with col_n2:
        st.warning("""
        ** Points d'attention :**
        
        - Les données simulées (zones, fournisseurs) sont indicatives
        - Les prévisions ML sont des estimations, pas des certitudes
        - Le dictionnaire est la référence en cas de doute sur un indicateur
        - Toute modification de formule doit être documentée ici
        """)
    
    st.markdown("---")
    st.caption("Dictionnaire de données - Version 1.0 - Dernière mise à jour : " + datetime.now().strftime("%d/%m/%Y"))

# ==========================================================
# TAB DIRECTION : SIMULATION FINANCIÈRE
# ==========================================================

if tab_direction is not None:
    with tab_direction:
        st.markdown("## Simulation Financière - Direction")
        st.caption("Évaluation de l'impact des investissements · CA prévisionnel · Bénéfice · Seuil de rentabilité")

        # ============================================
        # INITIALISATION SESSION STATE
    # ============================================
    if 'simulation_direction' not in st.session_state:
        st.session_state.simulation_direction = None
    
    # ============================================
    # PARAMÈTRES ACTUELS (DONNÉES RÉELLES)
    # ============================================
    with st.expander("Données de référence actuelles", expanded=False):
        col_ref1, col_ref2, col_ref3 = st.columns(3)
        
        with col_ref1:
            if not facture_filtered.empty and 'MONTANT_NET' in facture_filtered.columns:
                ca_actuel_ref = facture_filtered['MONTANT_NET'].sum()
            else:
                ca_actuel_ref = 50000000
            st.metric("CA Annuel Actuel", format_cfa(ca_actuel_ref))
        
        with col_ref2:
            nb_clients_ref = facture_filtered['ID_PERSONNE'].nunique() if not facture_filtered.empty and 'ID_PERSONNE' in facture_filtered.columns else 0
            st.metric("Clients Actifs", f"{nb_clients_ref:,}")
        
        with col_ref3:
            prix_moyen_ref = 1000
            st.metric("Prix Moyen/Bouteille", format_cfa(prix_moyen_ref))
    
    st.markdown("---")
    
    # ============================================
    # PARTIE 1 : SITUATION ACTUELLE
    # ============================================
    st.markdown("## Situation Actuelle de l'Entreprise")
    st.caption("Renseignez les données financières actuelles")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        capital_actuel = st.number_input(
            "Capital Actuel (FCFA)",
            min_value=0, value=10000000, step=1000000, format="%d",
            help="Capital social actuel de l'entreprise"
        )
    
    with col_s2:
        ca_actuel = st.number_input(
            "CA Annuel Actuel (FCFA)",
            min_value=0, value=int(ca_actuel_ref), step=1000000, format="%d",
            help="Chiffre d'affaires annuel réalisé"
        )
    
    with col_s3:
        charges_fixes_actuelles = st.number_input(
            "Charges Fixes Annuelles (FCFA)",
            min_value=0, value=15000000, step=1000000, format="%d",
            help="Loyer, salaires, électricité, maintenance, etc."
        )
    
    with col_s4:
        charges_variables_pct = st.slider(
            "Charges Variables (% du CA)",
            min_value=0, max_value=100, value=45, step=5,
            help="Matières premières, emballages, transport (en % du CA)"
        )
    
    # Calculs situation actuelle
    charges_variables_actuelles = ca_actuel * (charges_variables_pct / 100)
    charges_totales_actuelles = charges_fixes_actuelles + charges_variables_actuelles
    resultat_actuel = ca_actuel - charges_totales_actuelles
    
    # Seuil de rentabilité actuel
    taux_marge_cv = 1 - (charges_variables_pct / 100)
    seuil_rentabilite_actuel = charges_fixes_actuelles / taux_marge_cv if taux_marge_cv > 0 else float('inf')
    
    st.markdown("---")
    
    col_act1, col_act2, col_act3, col_act4, col_act5 = st.columns(5)
    
    with col_act1:
        st.metric("Capital", format_cfa(capital_actuel))
    with col_act2:
        st.metric("CA Annuel", format_cfa(ca_actuel))
    with col_act3:
        st.metric("Charges Totales", format_cfa(charges_totales_actuelles))
    with col_act4:
        delta_color = "normal" if resultat_actuel >= 0 else "inverse"
        st.metric("Résultat Net", format_cfa(resultat_actuel), delta_color=delta_color)
    with col_act5:
        st.metric("Seuil de Rentabilité", format_cfa(seuil_rentabilite_actuel),
                 help="CA minimum pour couvrir toutes les charges")
    
    # Rentabilité actuelle
    rentabilite_actuelle = (resultat_actuel / ca_actuel * 100) if ca_actuel > 0 else 0
    
    if resultat_actuel > 0:
        st.success(f"L'entreprise est **rentable** : marge nette de **{rentabilite_actuelle:.1f}%**")
    elif resultat_actuel == 0:
        st.warning("L'entreprise est à **l'équilibre** (ni bénéfice, ni perte)")
    else:
        st.error(f"L'entreprise est en **perte** de **{format_cfa(abs(resultat_actuel))}**")
    
    st.markdown("---")
    
    # ============================================
    # PARTIE 2 : SIMULATION D'INVESTISSEMENT
    # ============================================
    st.markdown("## Simulation d'Augmentation de Capital")
    st.caption("Évaluez l'impact d'un investissement supplémentaire")
    
    col_inv1, col_inv2, col_inv3 = st.columns(3)
    
    with col_inv1:
        augmentation_capital = st.number_input(
            "Augmentation de Capital (FCFA)",
            min_value=0, value=5000000, step=1000000, format="%d",
            help="Montant de l'augmentation de capital envisagée"
        )
    
    with col_inv2:
        affectation_production = st.slider(
            "Part affectée à la Production (%)",
            min_value=0, max_value=100, value=50, step=10,
            help="% de l'investissement dédié à l'outil de production (machines, équipements)"
        )
    
    with col_inv3:
        affectation_commercial = st.slider(
            "Part affectée au Commercial (%)",
            min_value=0, max_value=100, value=30, step=10,
            help="% de l'investissement dédié au marketing et à la force de vente"
        )
    
    # Calcul de l'affectation restante (trésorerie)
    affectation_tresorerie = 100 - affectation_production - affectation_commercial
    if affectation_tresorerie < 0:
        st.error("Le total des affectations dépasse 100%. Veuillez ajuster.")
        affectation_tresorerie = 0
    
    col_aff1, col_aff2, col_aff3 = st.columns(3)
    with col_aff1:
        st.metric("Production", format_cfa(augmentation_capital * affectation_production / 100))
    with col_aff2:
        st.metric("Commercial", format_cfa(augmentation_capital * affectation_commercial / 100))
    with col_aff3:
        st.metric("Trésorerie", format_cfa(augmentation_capital * affectation_tresorerie / 100))
    
    st.markdown("---")
    
    # ============================================
    # HYPOTHÈSES D'IMPACT
    # ============================================
    st.markdown("### Hypothèses d'Impact sur le CA")
    
    col_hyp1, col_hyp2, col_hyp3 = st.columns(3)
    
    with col_hyp1:
        croissance_ca_production = st.slider(
            "Croissance CA grâce à la Production (%)",
            min_value=0, max_value=100, value=15, step=5,
            help="Augmentation estimée du CA grâce aux nouveaux équipements"
        )
    
    with col_hyp2:
        croissance_ca_commercial = st.slider(
            "Croissance CA grâce au Commercial (%)",
            min_value=0, max_value=100, value=20, step=5,
            help="Augmentation estimée du CA grâce aux actions marketing"
        )
    
    with col_hyp3:
        reduction_charges_variables = st.slider(
            "Réduction des Charges Variables (%)",
            min_value=0, max_value=50, value=5, step=1,
            help="Gain d'efficacité réduisant le coût des MP et de production"
        )
    
    # ============================================
    # BOUTON DE CALCUL
    # ============================================
    if st.button("Lancer la Simulation", type="primary", use_container_width=True):
        
        # Nouveau capital
        nouveau_capital = capital_actuel + augmentation_capital
        
        # Nouveau CA
        impact_production = ca_actuel * (croissance_ca_production / 100) * (affectation_production / 100)
        impact_commercial = ca_actuel * (croissance_ca_commercial / 100) * (affectation_commercial / 100)
        nouveau_ca = ca_actuel + impact_production + impact_commercial
        
        # Nouvelles charges
        nouveau_charges_variables_pct = charges_variables_pct * (1 - reduction_charges_variables / 100)
        # Les charges fixes augmentent avec les nouveaux équipements (amortissement)
        amortissement = augmentation_capital * (affectation_production / 100) * 0.15  # 15% d'amortissement
        nouvelles_charges_fixes = charges_fixes_actuelles + amortissement
        nouvelles_charges_variables = nouveau_ca * (nouveau_charges_variables_pct / 100)
        nouvelles_charges_totales = nouvelles_charges_fixes + nouvelles_charges_variables
        
        # Nouveau résultat
        nouveau_resultat = nouveau_ca - nouvelles_charges_totales
        
        # Nouveau seuil de rentabilité
        nouveau_taux_marge_cv = 1 - (nouveau_charges_variables_pct / 100)
        nouveau_seuil = nouvelles_charges_fixes / nouveau_taux_marge_cv if nouveau_taux_marge_cv > 0 else float('inf')
        
        # Rentabilité des capitaux propres
        rcp_avant = (resultat_actuel / capital_actuel * 100) if capital_actuel > 0 else 0
        rcp_apres = (nouveau_resultat / nouveau_capital * 100) if nouveau_capital > 0 else 0
        
        # Stockage des résultats
        st.session_state.simulation_direction = {
            'capital_actuel': capital_actuel,
            'augmentation': augmentation_capital,
            'nouveau_capital': nouveau_capital,
            'ca_actuel': ca_actuel,
            'nouveau_ca': nouveau_ca,
            'impact_production': impact_production,
            'impact_commercial': impact_commercial,
            'charges_fixes_actuelles': charges_fixes_actuelles,
            'nouvelles_charges_fixes': nouvelles_charges_fixes,
            'charges_variables_actuelles': charges_variables_actuelles,
            'nouvelles_charges_variables': nouvelles_charges_variables,
            'charges_totales_actuelles': charges_totales_actuelles,
            'nouvelles_charges_totales': nouvelles_charges_totales,
            'resultat_actuel': resultat_actuel,
            'nouveau_resultat': nouveau_resultat,
            'seuil_actuel': seuil_rentabilite_actuel,
            'nouveau_seuil': nouveau_seuil,
            'rcp_avant': rcp_avant,
            'rcp_apres': rcp_apres,
            'rentabilite_actuelle': rentabilite_actuelle,
            'nouvelle_rentabilite': (nouveau_resultat / nouveau_ca * 100) if nouveau_ca > 0 else 0,
            'amortissement': amortissement
        }
        
        st.success("Simulation terminée avec succès !")
    
    # ============================================
    # PARTIE 3 : RÉSULTATS DE LA SIMULATION
    # ============================================
    if st.session_state.simulation_direction is not None:
        sim = st.session_state.simulation_direction
        
        st.markdown("---")
        st.markdown("## Résultats de la Simulation")
        
        # Comparaison avant/après
        st.markdown("### Comparaison Avant / Après Investissement")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("""
            <div style='background-color: #fff3cd; padding: 15px; border-radius: 10px;'>
            <h4 style='margin-top: 0;'> AVANT Investissement</h4>
            """, unsafe_allow_html=True)
            
            st.metric("Capital", format_cfa(sim['capital_actuel']))
            st.metric("CA Annuel", format_cfa(sim['ca_actuel']))
            st.metric("Charges Fixes", format_cfa(sim['charges_fixes_actuelles']))
            st.metric("Charges Variables", format_cfa(sim['charges_variables_actuelles']))
            st.metric("Résultat Net", format_cfa(sim['resultat_actuel']),
                     delta_color="normal" if sim['resultat_actuel'] >= 0 else "inverse")
            st.metric("Seuil de Rentabilité", format_cfa(sim['seuil_actuel']))
            st.metric("Rentabilité Capitaux Propres", f"{sim['rcp_avant']:.1f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_comp2:
            # Couleur selon amélioration
            couleur = '#d4edda' if sim['nouveau_resultat'] > sim['resultat_actuel'] else '#f8d7da'
            
            st.markdown(f"""
            <div style='background-color: {couleur}; padding: 15px; border-radius: 10px;'>
            <h4 style='margin-top: 0;'> APRÈS Investissement</h4>
            """, unsafe_allow_html=True)
            
            st.metric("Capital", format_cfa(sim['nouveau_capital']),
                     delta=f"+{format_cfa(sim['augmentation'])}")
            st.metric("CA Annuel", format_cfa(sim['nouveau_ca']),
                     delta=f"+{format_cfa(sim['nouveau_ca'] - sim['ca_actuel'])}")
            st.metric("Charges Fixes", format_cfa(sim['nouvelles_charges_fixes']),
                     delta=f"+{format_cfa(sim['nouvelles_charges_fixes'] - sim['charges_fixes_actuelles'])}",
                     delta_color="inverse")
            st.metric("Charges Variables", format_cfa(sim['nouvelles_charges_variables']),
                     delta=f"{sim['nouvelles_charges_variables'] - sim['charges_variables_actuelles']:+,.0f}",
                     delta_color="inverse")
            
            delta_resultat = sim['nouveau_resultat'] - sim['resultat_actuel']
            st.metric("Résultat Net", format_cfa(sim['nouveau_resultat']),
                     delta=f"{delta_resultat:+,.0f}",
                     delta_color="normal" if delta_resultat >= 0 else "inverse")
            
            st.metric("Seuil de Rentabilité", format_cfa(sim['nouveau_seuil']),
                     delta=f"{sim['nouveau_seuil'] - sim['seuil_actuel']:+,.0f}",
                     delta_color="inverse")
            
            st.metric("Rentabilité Capitaux Propres", f"{sim['rcp_apres']:.1f}%",
                     delta=f"{sim['rcp_apres'] - sim['rcp_avant']:+.1f} pts")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # ============================================
        # GRAPHIQUE COMPARATIF
        # ============================================
        st.markdown("---")
        st.markdown("### Graphique Comparatif")
        
        fig_comp_sim = go.Figure()
        
        categories = ['Capital', 'CA Annuel', 'Résultat Net', 'Seuil Rentabilité']
        valeurs_avant = [sim['capital_actuel'], sim['ca_actuel'], sim['resultat_actuel'], sim['seuil_actuel']]
        valeurs_apres = [sim['nouveau_capital'], sim['nouveau_ca'], sim['nouveau_resultat'], sim['nouveau_seuil']]
        
        fig_comp_sim.add_trace(go.Bar(
            x=categories,
            y=valeurs_avant,
            name='Avant Investissement',
            marker_color='#ff7f0e',
            text=[format_cfa(v) for v in valeurs_avant],
            textposition='auto'
        ))
        
        fig_comp_sim.add_trace(go.Bar(
            x=categories,
            y=valeurs_apres,
            name='Après Investissement',
            marker_color='#2ca02c',
            text=[format_cfa(v) for v in valeurs_apres],
            textposition='auto'
        ))
        
        fig_comp_sim.update_layout(
            title="Comparaison Avant/Après Investissement",
            barmode='group',
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_comp_sim, use_container_width=True)
        
        # ============================================
        # DÉTAIL DES IMPACTS
        # ============================================
        st.markdown("---")
        st.markdown("### Détail des Impacts")
        
        col_imp1, col_imp2, col_imp3 = st.columns(3)
        
        with col_imp1:
            st.markdown("#### Impact Production")
            st.metric("Investissement", format_cfa(sim['augmentation'] * affectation_production / 100))
            st.metric("CA Additionnel", format_cfa(sim['impact_production']))
            st.metric("Amortissement/an", format_cfa(sim['amortissement']))
        
        with col_imp2:
            st.markdown("#### Impact Commercial")
            st.metric("Investissement", format_cfa(sim['augmentation'] * affectation_commercial / 100))
            st.metric("CA Additionnel", format_cfa(sim['impact_commercial']))
        
        with col_imp3:
            st.markdown("#### Impact Trésorerie")
            st.metric("Trésorerie", format_cfa(sim['augmentation'] * affectation_tresorerie / 100))
            st.metric("CA Total Additionnel", format_cfa(sim['impact_production'] + sim['impact_commercial']))
        
        # ============================================
        # INDICATEURS DE RENTABILITÉ
        # ============================================
        st.markdown("---")
        st.markdown("### Indicateurs de Rentabilité")
        
        col_rent1, col_rent2, col_rent3, col_rent4 = st.columns(4)
        
        with col_rent1:
            st.metric("Marge Nette Avant", f"{sim['rentabilite_actuelle']:.1f}%")
        with col_rent2:
            st.metric("Marge Nette Après", f"{sim['nouvelle_rentabilite']:.1f}%",
                     delta=f"{sim['nouvelle_rentabilite'] - sim['rentabilite_actuelle']:+.1f} pts")
        with col_rent3:
            st.metric("RCP Avant", f"{sim['rcp_avant']:.1f}%")
        with col_rent4:
            st.metric("RCP Après", f"{sim['rcp_apres']:.1f}%",
                     delta=f"{sim['rcp_apres'] - sim['rcp_avant']:+.1f} pts")
        
        # Retour sur investissement
        roi = ((sim['nouveau_resultat'] - sim['resultat_actuel']) / sim['augmentation'] * 100) if sim['augmentation'] > 0 else 0
        delai_recup = (sim['augmentation'] / (sim['nouveau_resultat'] - sim['resultat_actuel'])) if (sim['nouveau_resultat'] - sim['resultat_actuel']) > 0 else float('inf')
        
        col_roi1, col_roi2 = st.columns(2)
        with col_roi1:
            st.metric("ROI (Retour sur Investissement)", f"{roi:.1f}%")
        with col_roi2:
            if delai_recup != float('inf'):
                st.metric("Délai de Récupération", f"{delai_recup:.1f} an(s)")
            else:
                st.metric("Délai de Récupération", "Non récupérable")
        
        # ============================================
        # AVIS DÉCISIONNEL
        # ============================================
        st.markdown("---")
        st.markdown("### Avis Décisionnel")
        
        if sim['nouveau_resultat'] > sim['resultat_actuel'] and roi > 15:
            st.success(f"""
             **RECOMMANDATION : INVESTIR**
            
            L'augmentation de capital de **{format_cfa(sim['augmentation'])}** est **recommandée** pour les raisons suivantes :
            
            - Le résultat net passerait de **{format_cfa(sim['resultat_actuel'])}** à **{format_cfa(sim['nouveau_resultat'])}** (+{format_cfa(sim['nouveau_resultat'] - sim['resultat_actuel'])})
            - La rentabilité des capitaux propres évoluerait de **{sim['rcp_avant']:.1f}%** à **{sim['rcp_apres']:.1f}%**
            - Le retour sur investissement est de **{roi:.1f}%** avec un délai de récupération de **{delai_recup:.1f} an(s)**
            - Le nouveau seuil de rentabilité de **{format_cfa(sim['nouveau_seuil'])}** est couvert par le CA prévisionnel de **{format_cfa(sim['nouveau_ca'])}**
            
            **Prochaines étapes :** Valider en conseil d'administration et procéder à l'augmentation de capital.
            """)
        elif sim['nouveau_resultat'] > sim['resultat_actuel']:
            st.warning(f"""
             **RECOMMANDATION : ÉTUDIER PLUS EN DÉTAIL**
            
            L'investissement de **{format_cfa(sim['augmentation'])}** améliore le résultat mais le ROI est modéré ({roi:.1f}%).
            
            Suggestions :
            - Revoir l'affectation des ressources
            - Négocier de meilleures conditions d'achat
            - Augmenter la part commerciale pour booster le CA
            """)
        else:
            st.error(f"""
             **RECOMMANDATION : NE PAS INVESTIR**
            
            L'investissement de **{format_cfa(sim['augmentation'])}** n'améliore pas la situation financière.
            
            Le résultat net passerait de **{format_cfa(sim['resultat_actuel'])}** à **{format_cfa(sim['nouveau_resultat'])}**.
            
            Suggestions :
            - Réduire les charges fixes avant d'investir
            - Optimiser la production actuelle
            - Chercher des financements alternatifs (subventions, crédits)
            """)
        
        # ============================================
        # EXPORT
        # ============================================
        st.markdown("---")
        
        csv_sim = pd.DataFrame([
            {'Indicateur': 'Capital', 'Avant': sim['capital_actuel'], 'Après': sim['nouveau_capital'], 'Variation': sim['augmentation']},
            {'Indicateur': 'CA Annuel', 'Avant': sim['ca_actuel'], 'Après': sim['nouveau_ca'], 'Variation': sim['nouveau_ca'] - sim['ca_actuel']},
            {'Indicateur': 'Charges Fixes', 'Avant': sim['charges_fixes_actuelles'], 'Après': sim['nouvelles_charges_fixes'], 'Variation': sim['nouvelles_charges_fixes'] - sim['charges_fixes_actuelles']},
            {'Indicateur': 'Charges Variables', 'Avant': sim['charges_variables_actuelles'], 'Après': sim['nouvelles_charges_variables'], 'Variation': sim['nouvelles_charges_variables'] - sim['charges_variables_actuelles']},
            {'Indicateur': 'Résultat Net', 'Avant': sim['resultat_actuel'], 'Après': sim['nouveau_resultat'], 'Variation': sim['nouveau_resultat'] - sim['resultat_actuel']},
            {'Indicateur': 'Seuil de Rentabilité', 'Avant': sim['seuil_actuel'], 'Après': sim['nouveau_seuil'], 'Variation': sim['nouveau_seuil'] - sim['seuil_actuel']},
        ]).to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Exporter la simulation (CSV)",
            data=csv_sim,
            file_name=f"simulation_direction_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>TBC - To Be Confirmed </b><br>
</div>
""", unsafe_allow_html=True)
