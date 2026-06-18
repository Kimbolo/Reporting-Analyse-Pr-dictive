import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from db import get_data
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# Configuration de la page
st.set_page_config(
    page_title="Analytics — Audit de performance",
    layout="wide"
)

# ============================================
# CONSTANTES & CONFIGURATION
# ============================================

MOIS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}

ORDRE_MOIS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", 
              "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]

# ============================================
# CHARGEMENT DES DONNÉES (CACHÉ)
# ============================================

@st.cache_data(ttl=300, show_spinner="Chargement des données...")
def load_all_data() -> Dict[str, Any]:
    """Charge toutes les données nécessaires de manière autonome"""
    try:
        # Chargement des tables principales
        data = {
            'facture': get_data("SELECT * FROM facture_client"),
            'paiement': get_data("SELECT * FROM paiement_facture"),
            'stock': get_data("SELECT * FROM stock"),
            'personne': get_data("SELECT ID_PERSONNE, NOM FROM personne"),
            'magasin': get_data("SELECT ID_MAGASIN, NOM_MAGASIN FROM magasin"),
            'produit': get_data("SELECT ID_PRODUIT, NOM_PRODUIT FROM produit") if not get_data("SELECT ID_PRODUIT, NOM_PRODUIT FROM produit").empty else pd.DataFrame()
        }
        
        # Vérification des données critiques
        if data['facture'] is None or data['facture'].empty:
            st.error("Impossible de charger les données des factures")
            return None
        
        # Conversion des dates
        if 'DATE_CREATION' in data['facture'].columns:
            data['facture']['DATE_CREATION'] = pd.to_datetime(data['facture']['DATE_CREATION'])
            data['facture']['ANNEE'] = data['facture']['DATE_CREATION'].dt.year
            data['facture']['MOIS'] = data['facture']['DATE_CREATION'].dt.month
            data['facture']['MOIS_NOM'] = data['facture']['MOIS'].map(MOIS_FR)
        
        if data['paiement'] is not None and not data['paiement'].empty and 'DATE_PAIEMENT' in data['paiement'].columns:
            data['paiement']['DATE_PAIEMENT'] = pd.to_datetime(data['paiement']['DATE_PAIEMENT'])
            data['paiement']['MOIS'] = data['paiement']['DATE_PAIEMENT'].dt.month
            data['paiement']['MOIS_NOM'] = data['paiement']['MOIS'].map(MOIS_FR)
        else:
            data['paiement'] = pd.DataFrame(columns=['DATE_PAIEMENT', 'MONTANT', 'MOIS_NOM'])
        
        # Fallbacks pour les tables optionnelles
        if data['personne'] is None or data['personne'].empty:
            data['personne'] = pd.DataFrame(columns=['ID_PERSONNE', 'NOM'])
        
        if data['magasin'] is None or data['magasin'].empty:
            data['magasin'] = pd.DataFrame(columns=['ID_MAGASIN', 'NOM_MAGASIN'])
        
        if data['stock'] is None or data['stock'].empty:
            data['stock'] = pd.DataFrame(columns=['ID_MAGASIN', 'ID_PRODUIT', 'QUANTITE'])
        
        return data
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return None

# ============================================
# FONCTIONS DE FILTRAGE
# ============================================

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Applique les filtres sélectionnés par l'utilisateur"""
    if df is None or df.empty:
        return df
    
    result = df.copy()
    
    # Filtre année
    if filters.get('annee') and 'ANNEE' in result.columns:
        result = result[result['ANNEE'] == filters['annee']]
    
    # Filtre mois
    if filters.get('mois') and len(filters['mois']) > 0 and 'MOIS_NOM' in result.columns:
        result = result[result['MOIS_NOM'].isin(filters['mois'])]
    
    # Filtre client
    if filters.get('client') and len(filters['client']) > 0 and 'ID_PERSONNE' in result.columns:
        result = result[result['ID_PERSONNE'].isin(filters['client'])]
    
    # Filtre magasin (pour stock)
    if filters.get('magasin') and len(filters['magasin']) > 0 and 'ID_MAGASIN' in result.columns:
        result = result[result['ID_MAGASIN'].isin(filters['magasin'])]
    
    return result

def get_filter_options(data: Dict[str, Any]) -> Dict[str, Any]:
    """Récupère les options disponibles pour les filtres"""
    options = {
        'annees': [],
        'mois': ORDRE_MOIS,
        'clients': [],
        'magasins': []
    }
    
    if data and data['facture'] is not None and not data['facture'].empty:
        if 'ANNEE' in data['facture'].columns:
            options['annees'] = sorted(data['facture']['ANNEE'].unique(), reverse=True)
        
        if 'ID_PERSONNE' in data['facture'].columns and data['personne'] is not None:
            facture_clients = data['facture'][['ID_PERSONNE']].drop_duplicates()
            clients_with_names = facture_clients.merge(data['personne'], on='ID_PERSONNE', how='left')
            clients_with_names['NOM'] = clients_with_names['NOM'].fillna(f"Client {clients_with_names['ID_PERSONNE']}")
            options['clients'] = clients_with_names[['ID_PERSONNE', 'NOM']].to_dict('records')
    
    if data and data['magasin'] is not None and not data['magasin'].empty:
        options['magasins'] = data['magasin'][['ID_MAGASIN', 'NOM_MAGASIN']].to_dict('records')
    
    return options

# ============================================
# FONCTIONS DE VISUALISATION
# ============================================

def display_kpi_metrics(df_facture: pd.DataFrame, filters: Dict[str, Any]):
    """Affiche les KPI principaux"""
    col1, col2, col3, col4 = st.columns(4)
    
    ca_total = df_facture['MONTANT_NET'].sum() if not df_facture.empty else 0
    nb_factures = len(df_facture)
    nb_clients = df_facture['ID_PERSONNE'].nunique() if 'ID_PERSONNE' in df_facture.columns else 0
    panier_moyen = ca_total / nb_factures if nb_factures > 0 else 0
    
    col1.metric("Chiffre d'affaires", f"{ca_total:,.0f} CFA")
    col2.metric("Nombre de factures", f"{nb_factures:,}")
    col3.metric("Clients actifs", f"{nb_clients:,}")
    col4.metric("Panier moyen", f"{panier_moyen:,.0f} CFA")
    
    return ca_total, nb_factures, nb_clients, panier_moyen

def display_ca_evolution(df_facture: pd.DataFrame):
    """Graphique d'évolution du CA mensuel"""
    if df_facture.empty or 'MOIS_NOM' not in df_facture.columns:
        st.info("Aucune donnée disponible pour l'évolution du CA")
        return
    
    ca_mensuel = df_facture.groupby('MOIS_NOM', sort=False)['MONTANT_NET'].sum().reset_index()
    
    fig = px.line(
        ca_mensuel,
        x='MOIS_NOM',
        y='MONTANT_NET',
        markers=True,
        title="Évolution mensuelle du chiffre d'affaires",
        labels={'MONTANT_NET': 'CA (CFA)', 'MOIS_NOM': 'Mois'}
    )
    fig.update_traces(line=dict(color='#1f77b4', width=3), marker=dict(size=10))
    fig.update_layout(hovermode='x unified', height=450)
    st.plotly_chart(fig, use_container_width=True, key="ca_evolution_chart")

def display_top_clients(df_facture: pd.DataFrame, data: Dict[str, Any]):
    """Top clients et analyse de concentration"""
    if df_facture.empty or 'ID_PERSONNE' not in df_facture.columns:
        st.info("Aucune donnée client disponible")
        return 0
    
    facture_client = df_facture.merge(data['personne'], on='ID_PERSONNE', how='left')
    facture_client['NOM'] = facture_client['NOM'].fillna(f"Client {facture_client['ID_PERSONNE']}")
    
    ca_client = facture_client.groupby('NOM')['MONTANT_NET'].sum().sort_values(ascending=False)
    
    if ca_client.empty:
        st.info("Aucune donnée client disponible")
        return 0
    
    top10 = ca_client.head(10)
    
    fig = px.bar(
        x=top10.values,
        y=top10.index,
        orientation='h',
        title="Top 10 clients par chiffre d'affaires",
        color=top10.values,
        color_continuous_scale='Blues',
        labels={'x': 'CA (CFA)', 'y': 'Client'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, key="top_clients_chart")
    
    part_top5 = ca_client.head(5).sum() / ca_client.sum() * 100
    
    if part_top5 > 60:
        st.warning(f" **Forte dépendance commerciale** - Le Top 5 clients représente {part_top5:.1f}% du CA")
    else:
        st.success(f" **Répartition équilibrée** - Le Top 5 clients représente {part_top5:.1f}% du CA")
    
    return part_top5

def display_top_produits(df_facture: pd.DataFrame, data: Dict[str, Any]):
    """Top produits vendus"""
    if df_facture.empty or 'ID_PRODUIT' not in df_facture.columns:
        st.info("Aucune donnée produit disponible")
        return
    
    ca_produit = df_facture.groupby('ID_PRODUIT')['MONTANT_NET'].sum().sort_values(ascending=False).head(10)
    
    if ca_produit.empty:
        st.info("Aucune donnée produit disponible")
        return
    
    # Fusion avec les noms de produits si disponibles
    if data['produit'] is not None and not data['produit'].empty:
        df_produits = pd.DataFrame({
            'ID_PRODUIT': ca_produit.index,
            'CA': ca_produit.values
        }).merge(data['produit'], on='ID_PRODUIT', how='left')
        df_produits['NOM'] = df_produits['NOM_PRODUIT'].fillna(f"Produit {df_produits['ID_PRODUIT']}")
        df_produits = df_produits.sort_values('CA', ascending=True)
        
        fig = px.bar(
            df_produits,
            x='CA',
            y='NOM',
            orientation='h',
            title="Top 10 produits par chiffre d'affaires",
            color='CA',
            color_continuous_scale='Viridis',
            labels={'CA': 'CA (CFA)', 'NOM': 'Produit'}
        )
    else:
        df_produits = pd.DataFrame({
            'Produit': ca_produit.index.astype(str),
            'CA': ca_produit.values
        }).sort_values('CA', ascending=True)
        
        fig = px.bar(
            df_produits,
            x='CA',
            y='Produit',
            orientation='h',
            title="Top 10 produits par chiffre d'affaires",
            color='CA',
            color_continuous_scale='Viridis'
        )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, key="top_produits_chart")

def display_stock_analysis(df_stock: pd.DataFrame, data: Dict[str, Any], df_facture: pd.DataFrame):
    """Analyse des stocks par magasin"""
    if df_stock.empty or 'ID_MAGASIN' not in df_stock.columns:
        st.info("Aucune donnée de stock disponible")
        return
    
    if not data['magasin'].empty:
        stock_with_names = df_stock.merge(data['magasin'], on='ID_MAGASIN', how='left')
        stock_with_names['NOM_MAGASIN'] = stock_with_names['NOM_MAGASIN'].fillna(f"Magasin {stock_with_names['ID_MAGASIN']}")
        stock_mag = stock_with_names.groupby('NOM_MAGASIN')['QUANTITE'].sum().sort_values(ascending=False)
    else:
        stock_mag = df_stock.groupby('ID_MAGASIN')['QUANTITE'].sum()
        stock_mag.index = [f"Magasin {idx}" for idx in stock_mag.index]
    
    if stock_mag.empty:
        st.info("Aucune donnée de stock disponible")
        return
    
    fig = px.bar(
        x=stock_mag.values,
        y=stock_mag.index,
        orientation='h',
        title="Niveau de stock par magasin",
        color=stock_mag.values,
        color_continuous_scale='Reds',
        labels={'x': 'Quantité en stock', 'y': 'Magasin'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key="stock_analysis_chart")

def display_cash_flow(df_facture: pd.DataFrame, df_paiement: pd.DataFrame):
    """Comparaison CA vs Encaissements"""
    if df_facture.empty:
        st.info("Aucune donnée de facturation disponible")
        return
    
    ca_mensuel = df_facture.groupby('MOIS_NOM', sort=False)['MONTANT_NET'].sum().reset_index()
    ca_mensuel.columns = ['MOIS_NOM', 'CA']
    
    if not df_paiement.empty and 'MONTANT' in df_paiement.columns and 'MOIS_NOM' in df_paiement.columns:
        enc_mensuel = df_paiement.groupby('MOIS_NOM', sort=False)['MONTANT'].sum().reset_index()
        enc_mensuel.columns = ['MOIS_NOM', 'Encaissements']
        df_cash = pd.merge(ca_mensuel, enc_mensuel, on='MOIS_NOM', how='outer').fillna(0)
    else:
        df_cash = ca_mensuel.copy()
        df_cash['Encaissements'] = 0
    
    df_cash['MOIS_NOM'] = pd.Categorical(df_cash['MOIS_NOM'], categories=ORDRE_MOIS, ordered=True)
    df_cash = df_cash.sort_values('MOIS_NOM')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_cash['MOIS_NOM'], y=df_cash['CA'],
        name='Chiffre d\'affaires', mode='lines+markers',
        line=dict(color='#1f77b4', width=3), marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df_cash['MOIS_NOM'], y=df_cash['Encaissements'],
        name='Encaissements', mode='lines+markers',
        line=dict(color='#2ca02c', width=3), marker=dict(size=8)
    ))
    fig.update_layout(
        title="Décalage CA vs Trésorerie",
        xaxis_title="Mois", yaxis_title="Montant (CFA)",
        hovermode='x unified', height=450
    )
    st.plotly_chart(fig, use_container_width=True, key="cash_flow_chart")
    
    if df_cash['CA'].sum() > 0:
        taux_enc = df_cash['Encaissements'].sum() / df_cash['CA'].sum() * 100
        st.metric("Taux d'encaissement global", f"{taux_enc:.1f}%")

# ============================================
# INTERFACE PRINCIPALE
# ============================================

st.title("Analytics — Audit de performance")
st.caption("Analyse expliquée et diagnostique des performances commerciales")

# Chargement des données
data = load_all_data()
if data is None:
    st.stop()

# Récupération des options de filtres
filter_options = get_filter_options(data)

# ============================================
# PANEL DE FILTRES (SIDEBAR)
# ============================================

with st.sidebar:
    st.markdown("## Filtres d'analyse")
    st.markdown("---")
    
    # Filtre année
    annees_dispo = filter_options['annees']
    if annees_dispo:
        selected_annee = st.selectbox("Année", annees_dispo, index=0)
    else:
        selected_annee = datetime.now().year
        st.info("Aucune année disponible dans les données")
    
    # Filtre mois
    selected_mois = st.multiselect("Mois", filter_options['mois'], default=[])
    
    # Filtre client
    if filter_options['clients']:
        client_options = {f"{c['NOM']} (ID: {c['ID_PERSONNE']})": c['ID_PERSONNE'] 
                         for c in filter_options['clients']}
        selected_clients_names = st.multiselect("Clients", list(client_options.keys()), default=[])
        selected_clients = [client_options[name] for name in selected_clients_names]
    else:
        selected_clients = []
    
    # Filtre magasin
    if filter_options['magasins']:
        magasin_options = {m['NOM_MAGASIN']: m['ID_MAGASIN'] for m in filter_options['magasins']}
        selected_magasins_names = st.multiselect("Magasins", list(magasin_options.keys()), default=[])
        selected_magasins = [magasin_options[name] for name in selected_magasins_names]
    else:
        selected_magasins = []
    
    st.markdown("---")
    st.caption(f"{len(data['facture'])} factures disponibles")

# Construction du dictionnaire des filtres
filters = {
    'annee': selected_annee if annees_dispo else None,
    'mois': selected_mois,
    'client': selected_clients,
    'magasin': selected_magasins
}

# Application des filtres aux données
df_facture_filtered = apply_filters(data['facture'], filters)
df_paiement_filtered = apply_filters(data['paiement'], filters) if not data['paiement'].empty else data['paiement']
df_stock_filtered = apply_filters(data['stock'], filters) if not data['stock'].empty else data['stock']

# Vérification après filtrage
if df_facture_filtered.empty:
    st.warning("Aucune donnée ne correspond aux filtres sélectionnés")
    st.info("Veuillez élargir vos critères de filtrage")
    st.stop()

# ============================================
# 1. CONTEXTE & KPI
# ============================================

st.markdown("## Contexte analytique")
st.info(f"Analyse de la performance commerciale basée sur **{len(df_facture_filtered)} factures**")

st.markdown("## Indicateurs clés")
ca_total, nb_factures, nb_clients, panier_moyen = display_kpi_metrics(df_facture_filtered, filters)

# ============================================
# 2. DYNAMIQUE DES VENTES
# ============================================

st.markdown("## Dynamique du chiffre d'affaires")
display_ca_evolution(df_facture_filtered)

with st.expander("Lecture & interprétation"):
    st.markdown("""
    - **Identification des périodes** de croissance et de ralentissement
    - Les **variations importantes** indiquent un effet saisonnier ou opérationnel
    - Une **baisse prolongée** peut signaler une contrainte structurelle
    """)

# ============================================
# 3. ANALYSE CLIENTS
# ============================================

st.markdown("## Analyse clients — concentration du CA")
part_top5 = display_top_clients(df_facture_filtered, data)

# ============================================
# 4. ANALYSE STOCK
# ============================================

st.markdown("## Stock & impact sur les ventes")
display_stock_analysis(df_stock_filtered, data, df_facture_filtered)

# ============================================
# 6. ANALYSE TRÉSORERIE
# ============================================

st.markdown("## Ventes vs Encaissements")
display_cash_flow(df_facture_filtered, df_paiement_filtered)

# ============================================
# 7. SYNTHÈSE FINALE
# ============================================

st.markdown("## Synthèse & diagnostic final")

# Génération dynamique de la synthèse
diagnostic_messages = []

if part_top5 > 60:
    diagnostic_messages.append(" **Dépendance client critique** - Risque de concentration élevé")
else:
    diagnostic_messages.append(" **Clientèle diversifiée** - Risque de concentration maîtrisé")

if ca_total > 0:
    if nb_factures < 100:
        diagnostic_messages.append(" **Volume d'activité modéré** - Potentiel de croissance significatif")
    else:
        diagnostic_messages.append(" **Volume d'activité soutenu** - Bonne dynamique commerciale")

if panier_moyen > 1000:
    diagnostic_messages.append(" **Panier moyen élevé** - Positionnement premium confirmé")

st.success(f"""
### Synthèse de la performance {selected_annee if annees_dispo else 'actuelle'}

{chr(10).join(f'- {msg}' for msg in diagnostic_messages)}

---
Indicateurs clés:
- Chiffre d'affaires total: **{ca_total:,.0f} CFA**
- Nombre de factures: **{nb_factures:,}**
- Clients actifs: **{nb_clients:,}**
- Panier moyen: **{panier_moyen:,.0f} CFA**

Recommandations:
- Diversifier le portefeuille clients pour réduire la dépendance
- Optimiser la gestion des stocks pour améliorer la trésorerie
- Renforcer le suivi des encaissements pour réduire les délais
""")
