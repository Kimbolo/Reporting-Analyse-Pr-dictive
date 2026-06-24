# ==========================================================
# PAGE TBC - TO BE CONFIRMED
# Tableau de Bord Complet - Prévisions, Géolocalisation, Approvisionnements
# ==========================================================

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
from fpdf import FPDF
import tempfile 
import os
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

st.title("TBC - To Be Confirmed")
st.caption("Prévisions ML · Corrélation Saisons · Cartographie · Approvisionnements · Relance Clients · Nouveaux Conditionnements")

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

# Calcul des KPI
ca_total = facture_filtered['MONTANT_NET'].sum() if not facture_filtered.empty and 'MONTANT_NET' in facture_filtered.columns else 0
nb_factures = len(facture_filtered)
nb_clients = facture_filtered['ID_PERSONNE'].nunique() if 'ID_PERSONNE' in facture_filtered.columns else 0
panier_moyen = ca_total / nb_factures if nb_factures > 0 else 0

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

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("C.A", format_cfa(ca_total))

with col2:
    st.metric("Factures", f"{nb_factures:,}")

with col3:
    st.metric("Clients actifs", f"{nb_clients:,}")

with col4:
    st.metric("Panier moyen", format_cfa(panier_moyen))

with col5:
    st.metric("Stock total", f"{stock_total:,.0f}")

with col6:
    delta_color = "inverse" if taux_perte > 5 else "normal"
    st.metric("Taux de perte", f"{taux_perte:.1f}%", delta="Objectif < 5%", delta_color=delta_color)

st.markdown("---")

# ==========================================================
# ONGLETS PRINCIPAUX
# ==========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Prévisions ML",
    "Corrélation Saisons",
    "Cartographie",
    "Approvisionnements",
    "Relance Clients",
    "Nouveaux Conditionnements",
    "Analyse Produits & Fruits",
    "Stocks & Magasins"
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
            
            # Affichage mensuel
            cols = st.columns(6)
            for i, ((annee, mois), prev) in enumerate(zip(mois_futurs, previsions)):
                with cols[i]:
                    st.metric(
                        f"{MOIS_ABBR[mois]} {annee}",
                        format_cfa(prev),
                        delta=f"±{mae:,.0f} FCFA",
                        delta_color="off"
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
            
            # Prévisions
            fig.add_trace(go.Scatter(
                x=futur_labels,
                y=previsions,
                mode='lines+markers',
                name='Prévisions',
                line=dict(color=COLORS['secondary'], width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            # Zone de confiance
            fig.add_trace(go.Scatter(
                x=list(futur_labels) + list(futur_labels[::-1]),
                y=list(previsions + mae) + list((previsions - mae)[::-1]),
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
            tendance = "HAUSSIÈRE" if moyenne_previsions > moyenne_historique else "BAISSIÈRE"
            
            st.info(f"**Tendance : {tendance}** | Prévision moyenne : {format_cfa(moyenne_previsions)} vs Historique : {format_cfa(moyenne_historique)}")
            
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
# TAB 2 : CORRÉLATION SAISONS
# ==========================================================

with tab2:
    st.markdown("## Corrélation Ventes / Saisons")
    
    if not facture_filtered.empty and 'SAISON' in facture_filtered.columns and 'MONTANT_NET' in facture_filtered.columns:
        # Analyse par saison
        ca_par_saison = facture_filtered.groupby('SAISON')['MONTANT_NET'].agg(['sum', 'mean', 'count']).reset_index()
        ca_par_saison.columns = ['Saison', 'CA_Total', 'CA_Moyen', 'Nombre_Factures']
        
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
            y=ca_par_saison['Nombre_Factures'],
            name='Nombre factures',
            mode='lines+markers',
            line=dict(color=COLORS['success'], width=3),
            marker=dict(size=12, symbol='diamond')
        ), secondary_y=True)
        
        fig.update_layout(
            title="Chiffre d'affaires et nombre de factures par saison",
            xaxis_title="Saison",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.05)
        )
        fig.update_yaxes(title_text="CA (FCFA)", secondary_y=False, tickformat=',.0f')
        fig.update_yaxes(title_text="Nombre de factures", secondary_y=True)
        
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
            - Nombre de factures : {meilleure_saison['Nombre_Factures']:.0f}
            - CA Moyen par facture : {format_cfa(meilleure_saison['CA_Moyen'])}
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
# TAB 4 : APPROVISIONNEMENTS
# ==========================================================

with tab4:
    st.markdown("## Indicateurs Fournisseurs & Approvisionnements")
    
    # Performance fournisseurs
    st.markdown("### Performance des fournisseurs")
    
    if not data['fournisseur'].empty:
        # Simulation de commandes fournisseurs (à remplacer par vraies données)
        if 'NOM_FOURNISSEUR' in data['fournisseur'].columns:
            fournisseurs_noms = data['fournisseur']['NOM_FOURNISSEUR'].tolist()
            
            # Données simulées pour démonstration
            np.random.seed(123)
            n_fournisseurs = len(fournisseurs_noms)
            
            perf_fournisseur = pd.DataFrame({
                'Nom': fournisseurs_noms,
                'Montant_Commandes': np.random.randint(500000, 50000000, n_fournisseurs),
                'Nb_Commandes': np.random.randint(5, 100, n_fournisseurs),
                'Delai_Moyen': np.random.randint(1, 15, n_fournisseurs),
                'Taux_Conformite': np.random.uniform(80, 100, n_fournisseurs)
            })
            perf_fournisseur = perf_fournisseur.sort_values('Montant_Commandes', ascending=False)
            
            # Graphique
            fig = px.bar(
                perf_fournisseur.head(10),
                x='Nom',
                y='Montant_Commandes',
                color='Taux_Conformite',
                color_continuous_scale='RdYlGn',
                title="Top fournisseurs par montant commandé",
                labels={'Montant_Commandes': 'Montant (FCFA)', 'Taux_Conformite': 'Conformité (%)'}
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé
            st.markdown("### Détail des fournisseurs")
            
            perf_display = perf_fournisseur.copy()
            perf_display['Montant_Commandes_F'] = perf_display['Montant_Commandes'].apply(format_cfa)
            
            st.dataframe(
                perf_display[['Nom', 'Montant_Commandes_F', 'Nb_Commandes', 'Delai_Moyen', 'Taux_Conformite']].head(10),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Nom': 'Fournisseur',
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
    
    # Niveaux de stock
    st.markdown("---")
    st.markdown("### Niveaux de stock par magasin")
    
    if not data['stock'].empty and not data['magasin'].empty:
        stock_mag = data['stock'].merge(data['magasin'], on='ID_MAGASIN', how='left')
        
        if 'NOM_MAGASIN' in stock_mag.columns:
            stock_par_magasin = stock_mag.groupby('NOM_MAGASIN')['QUANTITE'].sum().reset_index()
            stock_par_magasin = stock_par_magasin.sort_values('QUANTITE', ascending=False)
        else:
            stock_par_magasin = stock_mag.groupby('ID_MAGASIN')['QUANTITE'].sum().reset_index()
            stock_par_magasin.columns = ['NOM_MAGASIN', 'QUANTITE']
        
        fig = px.bar(
            stock_par_magasin,
            x='NOM_MAGASIN',
            y='QUANTITE',
            color='QUANTITE',
            color_continuous_scale='Reds',
            title="Stock par magasin",
            labels={'QUANTITE': 'Quantité en stock', 'NOM_MAGASIN': 'Magasin'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données de stock ou magasins indisponibles")
    
    # Prévision des besoins
    st.markdown("---")
    st.markdown("### Prévision des besoins d'approvisionnement")
    
    if not facture_filtered.empty and not data['stock'].empty:
        # Estimation des ventes mensuelles
        if 'MONTANT_NET' in facture_filtered.columns and 'DATE_CREATION' in facture_filtered.columns:
            ventes_mensuelles = facture_filtered.groupby(facture_filtered['DATE_CREATION'].dt.to_period("M"))['MONTANT_NET'].sum()
            
            if len(ventes_mensuelles) > 0:
                vente_moyenne = ventes_mensuelles.mean()
                stock_actuel = data['stock']['QUANTITE'].sum() if 'QUANTITE' in data['stock'].columns else 0
                
                # Estimation mois de couverture
                valeur_stock = stock_actuel * (vente_moyenne / max(stock_actuel, 1))
                mois_couverture = stock_actuel / max(vente_moyenne / 30, 1) if vente_moyenne > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Vente moyenne mensuelle", format_cfa(vente_moyenne))
                with col2:
                    st.metric("Stock actuel", f"{stock_actuel:,.0f} unités")
                with col3:
                    st.metric("Couverture estimée", f"{mois_couverture:.1f} mois")
                
                if mois_couverture < 2:
                    st.error("**URGENT** - Stock critique ! Approvisionnement nécessaire ")
                elif mois_couverture < 3:
                    st.warning("Stock à surveiller - Prévoir approvisionnement")
                else:
                    st.success("Niveau de stock satisfaisant")
        else:
            st.info("Colonnes manquantes pour le calcul des besoins")
    else:
        st.info("Données insuffisantes pour la prévision des besoins")

# ==========================================================
# TAB 5 : RELANCE CLIENTS
# ==========================================================

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
            
            # ⚡ SEGMENTATION CORRIGÉE - Client ancien = > 6 mois (180 jours) ⚡
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
            if m > 80: return '🟢 Excellente'
            elif m > 60: return '🟡 Bonne'
            elif m > 40: return '🟠 Acceptable'
            elif m > 0: return '🔴 Faible'
            else: return '⛔ Négative'
        
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
    # SOUS-TAB 1 : SAISONS & PRÉDICTIONS STOCKS FRUITS
    # ============================================
    with subtab_p1:
        st.markdown("### Saisonnalité & Prédictions Stocks de Fruits")
        st.caption("Analyse prédictive des besoins en fruits selon les saisons")
        
        # Fruits N'NAM avec saisons + prix saisonniers
        fruits_nnam = {
            'Ananas': {
                'saison': [3,4,5,6,7], 'pic': [5,6],
                'prix_saison': 200, 'prix_hors_saison': 450,
                'conservation': '5-7 jours', 'conso_mensuelle_pic': 5000, 'conso_mensuelle_creux': 1500
            },
            'Citron': {
                'saison': [11,12,1,2], 'pic': [12,1],
                'prix_saison': 150, 'prix_hors_saison': 350,
                'conservation': '21 jours', 'conso_mensuelle_pic': 3000, 'conso_mensuelle_creux': 800
            },
            'Gingembre': {
                'saison': list(range(1,13)), 'pic': [3,4,9,10],
                'prix_saison': 500, 'prix_hors_saison': 500,
                'conservation': '30 jours', 'conso_mensuelle_pic': 1500, 'conso_mensuelle_creux': 1200
            },
            'Orange': {
                'saison': [11,12,1,2], 'pic': [12,1],
                'prix_saison': 180, 'prix_hors_saison': 400,
                'conservation': '14 jours', 'conso_mensuelle_pic': 4000, 'conso_mensuelle_creux': 1000
            },
            'Mangue': {
                'saison': [3,4,5,6], 'pic': [4,5],
                'prix_saison': 250, 'prix_hors_saison': 600,
                'conservation': '3-5 jours', 'conso_mensuelle_pic': 3500, 'conso_mensuelle_creux': 800
            },
            'Goyave': {
                'saison': [3,4,5,6,7,8], 'pic': [5,6],
                'prix_saison': 180, 'prix_hors_saison': 400,
                'conservation': '3-4 jours', 'conso_mensuelle_pic': 2500, 'conso_mensuelle_creux': 600
            },
            'Fruit de la passion': {
                'saison': [3,4,5,6,7,8,9], 'pic': [5,6,7],
                'prix_saison': 300, 'prix_hors_saison': 700,
                'conservation': '7-10 jours', 'conso_mensuelle_pic': 2000, 'conso_mensuelle_creux': 500
            },
            'Papaye': {
                'saison': list(range(1,13)), 'pic': [3,4,9,10],
                'prix_saison': 200, 'prix_hors_saison': 350,
                'conservation': '5-7 jours', 'conso_mensuelle_pic': 2000, 'conso_mensuelle_creux': 1000
            },
            'Pastèque': {
                'saison': [1,2,3,4,5,12], 'pic': [2,3,4],
                'prix_saison': 350, 'prix_hors_saison': 700,
                'conservation': '7-10 jours', 'conso_mensuelle_pic': 1800, 'conso_mensuelle_creux': 400
            },
            'Baobab': {
                'saison': list(range(1,13)), 'pic': [1,2,3],
                'prix_saison': 1000, 'prix_hors_saison': 1000,
                'conservation': '180 jours (poudre)', 'conso_mensuelle_pic': 500, 'conso_mensuelle_creux': 400
            },
            'Hibiscus/Bissap': {
                'saison': [11,12,1,2,3], 'pic': [12,1],
                'prix_saison': 800, 'prix_hors_saison': 1200,
                'conservation': '365 jours (séché)', 'conso_mensuelle_pic': 800, 'conso_mensuelle_creux': 300
            },
            'Tamarin': {
                'saison': [1,2,3,4,12], 'pic': [2,3],
                'prix_saison': 600, 'prix_hors_saison': 900,
                'conservation': '90 jours', 'conso_mensuelle_pic': 600, 'conso_mensuelle_creux': 200
            },
        }
        
        mois_actuel = datetime.now().month
        
        # ============================================
        # STATUT ACTUEL
        # ============================================
        st.markdown(f"### Statut - {MOIS_FR[mois_actuel]} {datetime.now().year}")
        
        status_data = []
        for fruit, info in fruits_nnam.items():
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
            
            # Stock recommandé (2 semaines de conso)
            stock_recommande = conso / 2
            
            status_data.append({
                'Fruit': fruit,
                'Statut': statut,
                'Prix_Actuel': prix,
                'Conso_Mensuelle': conso,
                'Stock_Recommande': stock_recommande,
                'Conservation': info['conservation']
            })
        
        df_status = pd.DataFrame(status_data)
        
        # Résumé
        en_saison_df = df_status[df_status['Statut'].str.contains('En saison')]
        en_pic = df_status[df_status['Statut'] == 'PIC']
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1: st.metric("En saison", len(en_saison_df))
        with col_s2: st.metric("PIC", len(en_pic))
        with col_s3: st.metric("Hors saison", len(df_status) - len(en_saison_df))
        
        # Tableau avec recommandations stock
        st.markdown("#### Besoins en fruits et stocks recommandés")
        
        df_status['Prix_F'] = df_status['Prix_Actuel'].apply(lambda x: f"{x:,.0f} FCFA")
        df_status['Conso_F'] = df_status['Conso_Mensuelle'].apply(lambda x: f"{x:,.0f}/mois")
        df_status['Stock_F'] = df_status['Stock_Recommande'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(
            df_status[['Fruit', 'Statut', 'Prix_F', 'Conso_F', 'Stock_F', 'Conservation']],
            use_container_width=True, hide_index=True,
            column_config={
                'Fruit': 'Fruit',
                'Statut': 'Statut',
                'Prix_F': 'Prix actuel estimé',
                'Conso_F': 'Conso. mensuelle',
                'Stock_F': 'Stock recommandé',
                'Conservation': 'Conservation'
            }
        )
        
        # ============================================
        # PRÉDICTIONS 6 MOIS
        # ============================================
        st.markdown("---")
        st.markdown("### Prédictions Stocks & Achats - 6 prochains mois")
        
        # Générer les prédictions pour les 6 prochains mois
        mois_futurs = []
        for i in range(6):
            m = mois_actuel + i
            if m > 12: m -= 12
            mois_futurs.append(m)
        
        predictions = []
        for fruit, info in fruits_nnam.items():
            for i, m in enumerate(mois_futurs):
                en_saison = m in info['saison']
                pic = m in info['pic']
                
                if pic:
                    conso = info['conso_mensuelle_pic'] * 1.2
                    prix = info['prix_saison']
                    recommandation = "STOCKER ++ (PIC)"
                elif en_saison:
                    conso = info['conso_mensuelle_pic']
                    prix = info['prix_saison']
                    recommandation = "Acheter normal"
                else:
                    conso = info['conso_mensuelle_creux']
                    prix = info['prix_hors_saison']
                    recommandation = "Prix élevé - Stock minimum"
                
                predictions.append({
                    'Fruit': fruit,
                    'Mois': MOIS_ABBR[m],
                    'Mois_Num': m,
                    'Conso_Prevue': conso,
                    'Prix_Prevu': prix,
                    'Budget_Mensuel': conso * prix,
                    'Recommandation': recommandation
                })
        
        df_pred = pd.DataFrame(predictions)
        
        # Filtre par fruit
        fruit_selected = st.selectbox("Choisir un fruit pour voir les prédictions", 
                                       list(fruits_nnam.keys()))
        
        df_pred_fruit = df_pred[df_pred['Fruit'] == fruit_selected]
        
        # Graphique prédictions
        fig_pred = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_pred.add_trace(go.Bar(
            x=df_pred_fruit['Mois'], y=df_pred_fruit['Conso_Prevue'],
            name='Conso prévue (unités)', marker_color=COLORS['primary']
        ), secondary_y=False)
        
        fig_pred.add_trace(go.Scatter(
            x=df_pred_fruit['Mois'], y=df_pred_fruit['Prix_Prevu'],
            name='Prix prévu (FCFA)', mode='lines+markers',
            line=dict(color=COLORS['danger'], width=3), marker=dict(size=10)
        ), secondary_y=True)
        
        fig_pred.update_layout(
            title=f"Prédictions 6 mois - {fruit_selected}",
            xaxis_title="Mois", hovermode='x unified',
            template='plotly_white', height=400
        )
        fig_pred.update_yaxes(title_text="Consommation (unités)", secondary_y=False)
        fig_pred.update_yaxes(title_text="Prix unitaire (FCFA)", secondary_y=True)
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Tableau détails
        df_pred_fruit['Budget_F'] = df_pred_fruit['Budget_Mensuel'].apply(format_cfa)
        df_pred_fruit['Prix_F'] = df_pred_fruit['Prix_Prevu'].apply(lambda x: f"{x:,.0f} FCFA")
        df_pred_fruit['Conso_F'] = df_pred_fruit['Conso_Prevue'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(
            df_pred_fruit[['Mois', 'Conso_F', 'Prix_F', 'Budget_F', 'Recommandation']],
            use_container_width=True, hide_index=True,
            column_config={
                'Mois': 'Mois',
                'Conso_F': 'Conso prévue',
                'Prix_F': 'Prix prévu',
                'Budget_F': 'Budget estimé',
                'Recommandation': 'Action recommandée'
            }
        )
        
        # Résumé budget total fruits (6 mois)
        budget_total_6mois = df_pred.groupby('Fruit')['Budget_Mensuel'].sum()
        
        st.markdown("---")
        st.markdown("### Budget fruits estimé - 6 prochains mois")
        
        fig_budget = px.bar(
            x=budget_total_6mois.values, y=budget_total_6mois.index, orientation='h',
            title="Budget total par fruit (6 mois)", color=budget_total_6mois.values,
            color_continuous_scale='Oranges', text=budget_total_6mois.apply(format_cfa)
        )
        fig_budget.update_traces(textposition='outside')
        fig_budget.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_budget, use_container_width=True)
        
        budget_global = budget_total_6mois.sum()
        st.metric("Budget total fruits (6 mois)", format_cfa(budget_global))
        
        
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
                    lambda x: '🔴 Critique' if x <= seuil_bas else ('🟠 Faible' if x <= seuil_bas*3 else '🟢 OK')
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
                        lambda x: 'URGENT' if x == 0 else ('⚠️ Réappro.' if x < 10 else '📋 Surveiller')
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
                    st.download_button("📥 Exporter CSV", csv_alerte, f"alerte_stock_{datetime.now():%Y%m%d}.csv")
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
# FOOTER
# ==========================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>TBC - To Be Confirmed </b><br>
</div>
""", unsafe_allow_html=True)
