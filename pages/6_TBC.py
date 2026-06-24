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
warnings.filterwarnings('ignore')

# ==========================================================
# CONFIGURATION & STYLE
# ==========================================================

st.set_page_config(
    page_title="TBC - Tableau de Bord Complet",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    'Saison des pluies 🌧️': [4, 5, 6, 7, 8, 9, 10],
    'Saison sèche ☀️': [11, 12, 1, 2, 3]
}

def get_saison(mois):
    """Détermine la saison en fonction du mois"""
    for saison, mois_liste in SAISONS.items():
        if mois in mois_liste:
            return saison
    return 'Saison sèche ☀️'

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
            st.error("❌ Impossible de charger les données des factures")
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

st.title("📊 TBC - Tableau de Bord Complet")
st.caption("Prévisions ML · Corrélation Saisons · Cartographie · Approvisionnements · Relance Clients · Nouveaux Conditionnements")

# Chargement des données
data = load_tbc_data()
if data is None:
    st.stop()

# ==========================================================
# SIDEBAR - FILTRES
# ==========================================================

with st.sidebar:
    st.markdown("## 🎯 Filtres d'analyse")
    st.markdown("---")
    
    # Période
    if not data['facture'].empty and 'ANNEE' in data['facture'].columns:
        annees_dispo = sorted(data['facture']['ANNEE'].dropna().unique(), reverse=True)
        selected_annee = st.selectbox("Année", annees_dispo, index=0 if annees_dispo else 0)
    else:
        selected_annee = datetime.now().year
    
    # Mois
    mois_numeros = st.multiselect(
        "Mois",
        list(range(1, 13)),
        format_func=lambda x: MOIS_FR[x],
        default=[]
    )
    
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
        st.caption(f"📦 {len(data['stock'])} produits en stock")
    if not data['facture'].empty:
        st.caption(f"📄 {len(data['facture'])} factures chargées")
    if not data['personne'].empty:
        st.caption(f"👥 {len(data['personne'])} personnes")

# ==========================================================
# APPLICATION DES FILTRES
# ==========================================================

# Filtrer les factures
facture_filtered = data['facture'].copy()
if selected_annee in annees_dispo:
    facture_filtered = facture_filtered[facture_filtered['ANNEE'] == selected_annee]
if mois_numeros:
    facture_filtered = facture_filtered[facture_filtered['MOIS'].isin(mois_numeros)]

# ==========================================================
# KPI GLOBAUX
# ==========================================================

st.markdown("## 📊 Indicateurs clés de performance")

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
    st.metric("💰 Chiffre d'affaires", format_cfa(ca_total))

with col2:
    st.metric("📄 Factures", f"{nb_factures:,}")

with col3:
    st.metric("👥 Clients actifs", f"{nb_clients:,}")

with col4:
    st.metric("🛒 Panier moyen", format_cfa(panier_moyen))

with col5:
    st.metric("📦 Stock total", f"{stock_total:,.0f}")

with col6:
    delta_color = "inverse" if taux_perte > 5 else "normal"
    st.metric("⚠️ Taux de perte", f"{taux_perte:.1f}%", delta="Objectif < 5%", delta_color=delta_color)

st.markdown("---")

# ==========================================================
# ONGLETS PRINCIPAUX
# ==========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Prévisions ML",
    "🌤️ Corrélation Saisons",
    "🗺️ Cartographie",
    "📦 Approvisionnements",
    "📞 Relance Clients",
    "🧪 Nouveaux Conditionnements"
])

# ==========================================================
# TAB 1 : PRÉVISIONS ML
# ==========================================================

with tab1:
    st.markdown("## 📈 Prévisions ML & Ruptures de stock")
    
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
            st.markdown("### 🎯 Performance du modèle")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Erreur absolue moyenne", format_cfa(mae))
            with col2:
                st.metric("MAPE", f"{mape:.1f}%")
            with col3:
                st.metric("R²", f"{r2:.2%}")
            
            # Prévisions 6 mois
            st.markdown("### 📊 Prévisions des 6 prochains mois")
            
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
            st.markdown("### 📈 Évolution et prévisions")
            
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
            tendance = "📈 HAUSSIÈRE" if moyenne_previsions > moyenne_historique else "📉 BAISSIÈRE"
            
            st.info(f"**Tendance : {tendance}** | Prévision moyenne : {format_cfa(moyenne_previsions)} vs Historique : {format_cfa(moyenne_historique)}")
            
            # ============================================
            # RISQUE DE RUPTURE DE STOCK
            # ============================================
            st.markdown("---")
            st.markdown("### ⚠️ Analyse du risque de rupture de stock")
            
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
                    st.warning(f"⚠️ {len(stock_risque)} produits sont en dessous du seuil critique")
                    
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
                    st.success("✅ Aucun produit en risque de rupture critique")
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
    st.markdown("## 🌤️ Corrélation Ventes / Saisons")
    
    if not facture_filtered.empty and 'SAISON' in facture_filtered.columns and 'MONTANT_NET' in facture_filtered.columns:
        # Analyse par saison
        ca_par_saison = facture_filtered.groupby('SAISON')['MONTANT_NET'].agg(['sum', 'mean', 'count']).reset_index()
        ca_par_saison.columns = ['Saison', 'CA_Total', 'CA_Moyen', 'Nombre_Factures']
        
        # Graphique comparatif saisons
        st.markdown("### 📊 Performance par saison")
        
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
        st.markdown("### 📈 Évolution mensuelle du CA")
        
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
                'Saison des pluies 🌧️': COLORS['primary'],
                'Saison sèche ☀️': COLORS['secondary']
            }
        )
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights saisonniers
        st.markdown("### 💡 Insights saisonniers")
        
        col1, col2 = st.columns(2)
        with col1:
            meilleure_saison = ca_par_saison.loc[ca_par_saison['CA_Total'].idxmax()]
            st.success(f"""
            **🏆 Meilleure saison : {meilleure_saison['Saison']}**
            - CA Total : {format_cfa(meilleure_saison['CA_Total'])}
            - Nombre de factures : {meilleure_saison['Nombre_Factures']:.0f}
            - CA Moyen par facture : {format_cfa(meilleure_saison['CA_Moyen'])}
            """)
        
        with col2:
            if not ca_par_mois.empty:
                mois_pic = ca_par_mois.loc[ca_par_mois['MONTANT_NET'].idxmax()]
                mois_creux = ca_par_mois.loc[ca_par_mois['MONTANT_NET'].idxmin()]
                st.info(f"""
                **📈 Mois de pic : {mois_pic['MOIS_NOM']}**
                - CA : {format_cfa(mois_pic['MONTANT_NET'])}
                
                **📉 Mois le plus creux : {mois_creux['MOIS_NOM']}**
                - CA : {format_cfa(mois_creux['MONTANT_NET'])}
                """)
        
        # Recommandations actuelles
        saison_actuelle = get_saison(datetime.now().month)
        ca_moyen_saison = ca_par_saison[ca_par_saison['Saison'] == saison_actuelle]['CA_Moyen'].values
        ca_moyen_saison = ca_moyen_saison[0] if len(ca_moyen_saison) > 0 else 0
        
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;'>
            <b>🔄 Recommandations pour la {saison_actuelle}</b><br>
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
    st.markdown("## 🗺️ Géolocalisation des ventes")
    
    st.info("🔍 Analyse des ventes par zone géographique")
    
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
        st.markdown("### 📊 Top zones par chiffre d'affaires")
        
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
        st.markdown("### 📋 Détail par zone")
        
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
        st.markdown("### 🗺️ Carte de densité des ventes")
        
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
                **🏪 Zone la plus performante : {top_zone['Zone']}**
                - CA : {format_cfa(top_zone['CA_Total'])}
                - {top_zone['Nombre_Ventes']} ventes
                """)
            with col2:
                st.warning(f"""
                **📍 Zone à potentiel : {bottom_zone['Zone']}**
                - CA : {format_cfa(bottom_zone['CA_Total'])}
                - {bottom_zone['Nombre_Ventes']} ventes
                - Opportunité de développement
                """)
        
        st.info("""
        💡 **Prochaines étapes pour la géolocalisation :**
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
    st.markdown("## 📦 Indicateurs Fournisseurs & Approvisionnements")
    
    # Performance fournisseurs
    st.markdown("### 🏆 Performance des fournisseurs")
    
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
            st.markdown("### 📋 Détail des fournisseurs")
            
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
    st.markdown("### 📊 Niveaux de stock par magasin")
    
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
    st.markdown("### 🔮 Prévision des besoins d'approvisionnement")
    
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
                    st.error("⚠️ **URGENT** - Stock critique ! Approvisionnement nécessaire immédiatement")
                elif mois_couverture < 3:
                    st.warning("⚠️ Stock à surveiller - Prévoir approvisionnement")
                else:
                    st.success("✅ Niveau de stock satisfaisant")
        else:
            st.info("Colonnes manquantes pour le calcul des besoins")
    else:
        st.info("Données insuffisantes pour la prévision des besoins")

# ==========================================================
# TAB 5 : RELANCE CLIENTS
# ==========================================================

with tab5:
    st.markdown("## 📞 Relance des anciens clients")
    
    if not data['facture'].empty and not data['personne'].empty:
        # Fusion factures et personnes
        facture_client = data['facture'].merge(
            data['personne'][['ID_PERSONNE', 'NOM', 'TELEPHONE']], 
            on='ID_PERSONNE', 
            how='left'
        ) if 'TELEPHONE' in data['personne'].columns else data['facture'].merge(
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
            
            # Segmentation
            derniers_achats['Segment'] = pd.cut(
                derniers_achats['Jours_Depuis'],
                bins=[0, 30, 60, 90, 180, float('inf')],
                labels=['🟢 Actif', '🟡 À risque', '🟠 Perdu récemment', '🔴 Perdu', '⚫ Très ancien']
            )
            
            # Clients à relancer (>30 jours)
            clients_relance = derniers_achats[derniers_achats['Jours_Depuis'] > 30].copy()
            
            # KPI relance
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👤 Clients à relancer", len(clients_relance))
            with col2:
                ca_potentiel = clients_relance['CA_Total'].sum()
                st.metric("💰 CA historique", format_cfa(ca_potentiel))
            with col3:
                delai_moyen = clients_relance['Jours_Depuis'].mean()
                st.metric("📅 Délai moyen", f"{delai_moyen:.0f} jours")
            
            # Distribution par segment
            st.markdown("### 📊 Répartition des clients")
            
            segment_counts = derniers_achats['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Nombre']
            
            colors_segments = {
                '🟢 Actif': COLORS['success'],
                '🟡 À risque': COLORS['warning'],
                '🟠 Perdu récemment': COLORS['secondary'],
                '🔴 Perdu': COLORS['danger'],
                '⚫ Très ancien': COLORS['gray']
            }
            
            fig = px.pie(
                segment_counts,
                values='Nombre',
                names='Segment',
                title="Segmentation des clients",
                color='Segment',
                color_discrete_map=colors_segments,
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Liste des clients à relancer
            st.markdown("### 📋 Top clients à relancer")
            
            if not clients_relance.empty:
                clients_relance['Dernier_Achat_F'] = clients_relance['Dernier_Achat'].dt.strftime('%d/%m/%Y')
                clients_relance['CA_Total_F'] = clients_relance['CA_Total'].apply(format_cfa)
                clients_relance = clients_relance.sort_values('CA_Total', ascending=False)
                
                st.dataframe(
                    clients_relance[['Nom', 'Dernier_Achat_F', 'Jours_Depuis', 'CA_Total_F', 'Nb_Achats', 'Segment']].head(20),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Nom': 'Client',
                        'Dernier_Achat_F': 'Dernier achat',
                        'Jours_Depuis': 'Jours inactif',
                        'CA_Total_F': 'CA Total',
                        'Nb_Achats': 'Nb achats',
                        'Segment': 'Statut'
                    }
                )
                
                # Export
                csv_relance = clients_relance[['Nom', 'Dernier_Achat_F', 'Jours_Depuis', 'CA_Total']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Exporter la liste de relance (CSV)",
                    data=csv_relance,
                    file_name=f"relance_clients_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.success("✅ Tous vos clients sont actifs !")
            
            # Stratégie de relance
            st.markdown("### 💡 Stratégie de relance")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **🟡 Clients à risque (30-60 jours)**
                - 📧 Newsletter avec nouveautés
                - 🎁 Offre de fidélité
                - 📱 Message WhatsApp personnalisé
                """)
            with col2:
                st.markdown("""
                **🔴 Clients perdus (>60 jours)**
                - 📞 Appel téléphonique direct
                - 🏷️ Offre spéciale "Retour client"
                - 📋 Enquête de satisfaction
                """)
        else:
            st.info("Colonnes manquantes pour l'analyse clients")
    else:
        st.info("Données clients insuffisantes")

# ==========================================================
# TAB 6 : NOUVEAUX CONDITIONNEMENTS
# ==========================================================

with tab6:
    st.markdown("## 🧪 Nouveaux Conditionnements")
    st.caption("Analyse prédictive pour PET, 33cl, 1L, Sachets Kids")
    
    st.info("🔬 Cette section simule l'impact de l'introduction de nouveaux conditionnements sur votre chiffre d'affaires")
    
    if not facture_filtered.empty and 'MONTANT_NET' in facture_filtered.columns:
        ca_total = facture_filtered['MONTANT_NET'].sum()
        
        # Simulation des parts de marché actuelles
        st.markdown("### 📊 Répartition actuelle estimée du marché")
        
        conditionnements_actuels = ['Bouteille 33cl', 'Bouteille 1L', 'Sachet standard', 'Autres formats']
        parts_actuelles = [35, 40, 20, 5]
        
        df_actuel = pd.DataFrame({
            'Conditionnement': conditionnements_actuels,
            'Part_Marche': parts_actuelles,
            'CA_Estime': [ca_total * p/100 for p in parts_actuelles]
        })
        
        fig = px.pie(
            df_actuel,
            values='CA_Estime',
            names='Conditionnement',
            title="Répartition actuelle par conditionnement",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Simulation nouveaux conditionnements
        st.markdown("### 🔮 Prédiction d'impact des nouveaux conditionnements")
        
        nouveaux_cond = ['PET 1.5L', '33cl (nouveau)', '1L Premium', 'Sachets Kids']
        
        # Scénarios
        scenarios = {
            'Optimiste 📈': [15, 12, 10, 8],
            'Moyen 📊': [10, 8, 6, 6],
            'Pessimiste 📉': [5, 3, 2, 4]
        }
        
        scenario_choisi = st.selectbox("Scénario", list(scenarios.keys()))
        croissance = scenarios[scenario_choisi]
        
        impacts = []
        for i, cond in enumerate(nouveaux_cond):
            impact_ca = ca_total * (croissance[i] / 100)
            impacts.append({
                'Conditionnement': cond,
                'Croissance': croissance[i],
                'Impact_CA': impact_ca
            })
        
        df_impacts = pd.DataFrame(impacts)
        
        # Métriques
        ca_additionnel = df_impacts['Impact_CA'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CA actuel", format_cfa(ca_total))
        with col2:
            st.metric("CA additionnel estimé", format_cfa(ca_additionnel))
        with col3:
            st.metric("Croissance totale", f"+{sum(croissance):.0f}%")
        
        # Graphique d'impact
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_impacts['Conditionnement'],
            y=df_impacts['Impact_CA'],
            text=df_impacts['Impact_CA'].apply(format_cfa),
            textposition='outside',
            marker_color=COLORS['primary'],
            hovertemplate='<b>%{x}</b><br>Impact: %{text}<br>Croissance: %{customdata}%<extra></extra>',
            customdata=df_impacts['Croissance']
        ))
        
        fig.update_layout(
            title=f"Impact estimé - Scénario {scenario_choisi}",
            xaxis_title="Conditionnement",
            yaxis_title="CA supplémentaire estimé (FCFA)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommandation
        best_cond = df_impacts.loc[df_impacts['Impact_CA'].idxmax()]
        
        st.success(f"""
        **✅ Recommandation prioritaire : {best_cond['Conditionnement']}**
        - Impact estimé : {format_cfa(best_cond['Impact_CA'])}
        - Croissance : +{best_cond['Croissance']}% du CA
        - Potentiel de marché le plus élevé
        """)
        
        # Simulation ROI
        st.markdown("### 💰 Simulation de rentabilité")
        
        col1, col2 = st.columns(2)
        with col1:
            investissement = st.number_input(
                "Investissement nécessaire (FCFA)",
                min_value=100000,
                value=2000000,
                step=100000,
                help="Coût total estimé pour le lancement"
            )
        with col2:
            taux_marge = st.slider(
                "Taux de marge estimé (%)",
                min_value=10,
                max_value=60,
                value=30,
                step=5
            )
        
        marge_additionnelle = ca_additionnel * (taux_marge / 100)
        roi = ((marge_additionnelle - investissement) / investissement * 100) if investissement > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Marge additionnelle estimée", format_cfa(marge_additionnelle))
        with col2:
            st.metric("ROI estimé", f"{roi:.0f}%")
        
        if roi > 30:
            st.success("✅ Projet très rentable - Lancement fortement recommandé")
        elif roi > 0:
            st.warning("⚠️ Projet rentable mais modéré - Analyse complémentaire recommandée")
        else:
            st.error("❌ Projet non rentable dans les conditions actuelles")
    else:
        st.info("Données de ventes insuffisantes pour l'analyse")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>📊 TBC - Tableau de Bord Complet</b><br>
    Prévisions ML · Saisons · Cartographie · Approvisionnements · Relance · Conditionnements<br>
    <small>Données actualisées en temps réel | Powered by Streamlit</small>
</div>
""", unsafe_allow_html=True)
