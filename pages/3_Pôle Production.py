from plotly import data
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from db import get_data

# ==========================================================
# CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="N'NAM Jus - Pôle Production",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# STYLE
# ==========================================================
st.markdown("""
<style>
    .stMetricValue { font-size: 1.6rem !important; white-space: normal !important; }
    div[data-testid="column"] { min-width: 140px !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# CONSTANTES
# ==========================================================
MOIS_FR = {1:"Janvier", 2:"Février", 3:"Mars", 4:"Avril", 5:"Mai", 6:"Juin",
           7:"Juillet", 8:"Août", 9:"Septembre", 10:"Octobre", 11:"Novembre", 12:"Décembre"}
MOIS_ABBR = {1:"Jan", 2:"Fév", 3:"Mar", 4:"Avr", 5:"Mai", 6:"Juin",
             7:"Juil", 8:"Aoû", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Déc"}

COLORS = {
    'primary': '#1f77b4', 'secondary': '#ff7f0e', 'success': '#2ca02c',
    'danger': '#d62728', 'warning': '#ffbb78', 'purple': '#9467bd',
    'gray': '#7f7f7f', 'olive': '#bcbd22', 'cyan': '#17becf'
}

def format_cfa(v):
    if v is None or pd.isna(v): return "0 FCFA"
    return f"{v:,.0f}".replace(",", " ") + " FCFA"

def format_nombre(v):
    if v is None or pd.isna(v): return "0"
    return f"{v:,.0f}".replace(",", " ")

# ==========================================================
# CHARGEMENT DES DONNÉES
# ==========================================================
@st.cache_data(ttl=300, show_spinner="Chargement...")
def load_data():
    facture = get_data("SELECT * FROM facture_client")
    stock = get_data("SELECT * FROM stock")
    magasin = get_data("SELECT * FROM magasin")
    produit = get_data("SELECT * FROM produit")
    production = get_data("SELECT * FROM production")
    
    if facture is not None and not facture.empty:
        facture['DATE_CREATION'] = pd.to_datetime(facture['DATE_CREATION'], errors='coerce')
        facture['ANNEE'] = facture['DATE_CREATION'].dt.year
        facture['MOIS'] = facture['DATE_CREATION'].dt.month
        facture['MOIS_NOM'] = facture['MOIS'].map(MOIS_FR)
    
    return facture, stock, magasin, produit, production

facture, stock, magasin, produit, production = load_data()

if facture is None or facture.empty:
    st.error("Aucune donnée trouvée. Vérifiez la connexion à la base de données.")
    st.stop()

# ==========================================================
# SIDEBAR - FILTRES
# ==========================================================
with st.sidebar:
    st.markdown("## Filtres")
    
    annees_dispo = sorted(facture['ANNEE'].dropna().unique(), reverse=True)
    annee = st.selectbox("Année", annees_dispo, index=0)
    
    mois_dispo = sorted(facture[facture['ANNEE'] == annee]['MOIS'].unique())
    mois_sel = st.multiselect("Mois", mois_dispo, default=mois_dispo,
                              format_func=lambda x: MOIS_FR.get(x, str(x)))
    
    # Filtre magasin
    if magasin is not None and not magasin.empty:
        magasins_dispo = magasin['NOM_MAGASIN'].tolist() if 'NOM_MAGASIN' in magasin.columns else []
        magasin_sel = st.multiselect("Magasin", magasins_dispo, default=[])
    else:
        magasin_sel = []
    
    st.markdown("---")
    st.caption(f" {len(facture):,} factures chargées")
    st.caption(f" {len(stock):,} produits en stock" if stock is not None else "")

# Filtrer les factures
if mois_sel:
    facture_f = facture[(facture['ANNEE'] == annee) & (facture['MOIS'].isin(mois_sel))]
else:
    facture_f = facture[facture['ANNEE'] == annee]

# ==========================================================
# TITRE
# ==========================================================
st.title("Pôle Production")
st.caption(f"Analyse production – Année {annee} | {len(mois_sel)} mois")

# ==========================================================
# SOUS-ONGLETS
# ==========================================================
tab_prod_ventes, tab_stocks, tab_mp, tab_fournisseurs, tab_conditionnement = st.tabs([
    "Production vs Ventes",
    "Stocks & Magasins",
    "Magasin Matières Premières",
    "Performance Fournisseurs",
    "Nouveaux Conditionnements"
])

# ==========================================================
# TAB 1 : PRODUCTION VS VENTES
# ==========================================================

with tab_prod_ventes:
    st.markdown("## Comparaison Production vs Ventes")
    st.caption("Analyse des écarts entre volumes produits et volumes vendus (avec stock initial)")

    # ============================================
    # 1. VOLUME VENDU (depuis ligne_facture_client)
    # ============================================
    ventes_reelles = get_data(f"""
        SELECT 
            MONTH(fc.DATE_CREATION) as MOIS,
            SUM(lfc.QUANTITE) as QUANTITE_VENDUE
        FROM ligne_facture_client lfc
        JOIN facture_client fc ON lfc.ID_FACTURE_CLIENT = fc.ID_FACTURE_CLIENT
        WHERE YEAR(fc.DATE_CREATION) = {annee}
        GROUP BY MONTH(fc.DATE_CREATION)
        ORDER BY MOIS
    """)

    # ============================================
    # 2. VOLUME PRODUIT (depuis conditionnement_production)
    # ============================================
    production_reelle = get_data(f"""
        SELECT 
            MONTH(DATE_PRODUCTION) as MOIS,
            SUM(QUANTITE_REELLE) as QUANTITE_PRODUITE
        FROM conditionnement_production
        WHERE YEAR(DATE_PRODUCTION) = {annee}
        GROUP BY MONTH(DATE_PRODUCTION)
        ORDER BY MOIS
    """)

    # ============================================
    # 3. STOCK INITIAL (produits finis au 1er janvier)
    # ============================================
    stock_initial = get_data(f"""
        SELECT SUM(s.QUANTITE) as STOCK_INITIAL
        FROM stock s
        JOIN produit p ON s.ID_PRODUIT = p.ID_PRODUIT
        WHERE p.ID_CATEGORIE_PRODUIT = 9
    """)
    
    stock_initial_val = 0
    if stock_initial is not None and not stock_initial.empty:
        stock_initial_val = stock_initial['STOCK_INITIAL'].iloc[0] if 'STOCK_INITIAL' in stock_initial.columns else 0

    # ============================================
    # 4. STOCK FINAL ACTUEL (produits finis)
    # ============================================
    stock_final = get_data(f"""
        SELECT SUM(s.QUANTITE) as STOCK_FINAL
        FROM stock s
        JOIN produit p ON s.ID_PRODUIT = p.ID_PRODUIT
        WHERE p.ID_CATEGORIE_PRODUIT = 9
    """)
    
    stock_final_val = 0
    if stock_final is not None and not stock_final.empty:
        stock_final_val = stock_final['STOCK_FINAL'].iloc[0] if 'STOCK_FINAL' in stock_final.columns else 0

    # ============================================
    # 5. CONSTRUCTION DE LA COMPARAISON
    # ============================================
    if ventes_reelles is not None and not ventes_reelles.empty:
        ventes_reelles['Mois_Nom'] = ventes_reelles['MOIS'].map(MOIS_ABBR)
        
        if production_reelle is not None and not production_reelle.empty:
            production_reelle['Mois_Nom'] = production_reelle['MOIS'].map(MOIS_ABBR)
            comparaison = ventes_reelles.merge(
                production_reelle[['MOIS', 'QUANTITE_PRODUITE']],
                on='MOIS', how='outer'
            ).fillna(0)
        else:
            comparaison = ventes_reelles.copy()
            comparaison['QUANTITE_PRODUITE'] = 0
        
        # Calcul du stock disponible cumulé (stock initial + production cumulée)
        comparaison = comparaison.sort_values('MOIS')
        comparaison['Production_Cumul'] = comparaison['QUANTITE_PRODUITE'].cumsum()
        comparaison['Ventes_Cumul'] = comparaison['QUANTITE_VENDUE'].cumsum()
        comparaison['Stock_Disponible'] = stock_initial_val + comparaison['Production_Cumul']
        comparaison['Stock_Theorique_Fin_Mois'] = comparaison['Stock_Disponible'] - comparaison['Ventes_Cumul']
        comparaison['Ecart'] = comparaison['QUANTITE_PRODUITE'] - comparaison['QUANTITE_VENDUE']
        comparaison['Taux_Ecart'] = (comparaison['Ecart'] / comparaison['QUANTITE_PRODUITE'].replace(0, 1) * 100).round(1)

        # ============================================
        # KPIs GLOBAUX
        # ============================================
        total_vendu = comparaison['QUANTITE_VENDUE'].sum()
        total_produit = comparaison['QUANTITE_PRODUITE'].sum()
        total_disponible = stock_initial_val + total_produit
        ecart_global = total_disponible - total_vendu

        col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
        with col_k1:
            st.metric("Stock Initial", format_nombre(stock_initial_val))
        with col_k2:
            st.metric("Volume Produit", format_nombre(total_produit))
        with col_k3:
            st.metric("Total Disponible", format_nombre(total_disponible))
        with col_k4:
            st.metric("Volume Vendu", format_nombre(total_vendu))
        with col_k5:
            delta_color = "normal" if ecart_global >= 0 else "inverse"
            st.metric("Stock Final Théorique", format_nombre(ecart_global), delta_color=delta_color)

        # Cohérence
        st.markdown("---")
        if ecart_global >= 0:
            st.success(f"""
            **Cohérence OK** — Le stock initial ({format_nombre(stock_initial_val)}) + la production ({format_nombre(total_produit)}) 
            = **{format_nombre(total_disponible)}** unités disponibles, pour **{format_nombre(total_vendu)}** vendues.
            Stock final théorique : **{format_nombre(ecart_global)}** unités.
            """)
        else:
            st.error(f"""
            **INCOHÉRENCE DÉTECTÉE** — {format_nombre(total_vendu)} unités vendues pour seulement {format_nombre(total_disponible)} disponibles.
            Écart inexpliqué de **{format_nombre(abs(ecart_global))}** unités.
            
            **Causes possibles :**
            - Données de production manquantes (autre table de production ?)
            - Ventes saisies en double dans `ligne_facture_client`
            - Stocks non enregistrés dans la table `stock`
            """)

        st.markdown("---")

        # ============================================
        # Graphique : Stock disponible vs Ventes
        # ============================================
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=comparaison['Mois_Nom'], y=comparaison['Stock_Disponible'],
            name='Stock Disponible (Initial + Prod Cumulée)',
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=comparaison['Mois_Nom'], y=comparaison['Ventes_Cumul'],
            name='Ventes Cumulées',
            mode='lines+markers',
            line=dict(color=COLORS['success'], width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Bar(
            x=comparaison['Mois_Nom'], y=comparaison['QUANTITE_PRODUITE'],
            name='Production Mensuelle', marker_color=COLORS['secondary'],
            opacity=0.5, yaxis='y2'
        ))
        fig.add_trace(go.Bar(
            x=comparaison['Mois_Nom'], y=comparaison['QUANTITE_VENDUE'],
            name='Ventes Mensuelles', marker_color=COLORS['danger'],
            opacity=0.5, yaxis='y2'
        ))

        fig.update_layout(
            title=f"Stock disponible vs Ventes — {annee}",
            template='plotly_white', height=450, hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            yaxis=dict(title="Unités (cumul)"),
            yaxis2=dict(title="Unités (mensuel)", overlaying='y', side='right')
        )
        st.plotly_chart(fig, use_container_width=True)

        # ============================================
        # Tableau détaillé
        # ============================================
        st.markdown("### Détail mensuel")
        
        comparaison_display = comparaison[['Mois_Nom', 'QUANTITE_PRODUITE', 'QUANTITE_VENDUE', 
                                             'Stock_Disponible', 'Stock_Theorique_Fin_Mois', 'Taux_Ecart']].copy()
        comparaison_display.columns = ['Mois', 'Produit', 'Vendu', 'Stock Dispo', 'Stock Fin Mois', 'Taux Écart']
        comparaison_display['Produit_F'] = comparaison_display['Produit'].apply(format_nombre)
        comparaison_display['Vendu_F'] = comparaison_display['Vendu'].apply(format_nombre)
        comparaison_display['Stock_Dispo_F'] = comparaison_display['Stock Dispo'].apply(format_nombre)
        comparaison_display['Stock_Fin_F'] = comparaison_display['Stock Fin Mois'].apply(format_nombre)

        st.dataframe(
            comparaison_display[['Mois', 'Produit_F', 'Vendu_F', 'Stock_Dispo_F', 'Stock_Fin_F', 'Taux Écart']],
            use_container_width=True, hide_index=True, height=380,
            column_config={
                'Mois': 'Mois',
                'Produit_F': 'Volume Produit',
                'Vendu_F': 'Volume Vendu',
                'Stock_Dispo_F': 'Stock Disponible',
                'Stock_Fin_F': 'Stock Fin Mois',
                'Taux Écart': st.column_config.NumberColumn('Taux écart', format="%.1f%%")
            }
        )

# ==========================================================
# TAB 2 : STOCKS & MAGASINS
# ==========================================================

with tab_stocks:
        st.markdown("## Analyse des Stocks par Magasin")
        st.caption("Inventaire complet · Niveaux critiques · Comparaison magasins")
        
        if stock is not None and not stock.empty and magasin is not None and not magasin.empty:
            # Fusion stock avec magasins
            stock_complet = stock.copy()
            stock_complet = stock_complet.merge(magasin, on='ID_MAGASIN', how='left')

            # Ajouter la catégorie produit depuis la table produit
            produit_cat = get_data("SELECT ID_PRODUIT, ID_CATEGORIE_PRODUIT FROM produit")
            if produit_cat is not None and not produit_cat.empty:
                stock_complet = stock_complet.merge(produit_cat, on='ID_PRODUIT', how='left')
            
            # ============================================
            # DÉTERMINER LA COLONNE DE DÉSIGNATION
            # ============================================
            if 'DESIGNATION' in stock_complet.columns:
                nom_produit_col = 'DESIGNATION'
            elif 'NOM_PRODUIT' in stock_complet.columns:
                nom_produit_col = 'NOM_PRODUIT'
            elif 'NOM' in stock_complet.columns:
                nom_produit_col = 'NOM'
            else:
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
                # KPI magasin
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
                
                # VUE RÉSUMÉ
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
                    
                    st.markdown("#### Top 10 produits en stock")
                    top10 = stock_magasin.nlargest(10, 'QUANTITE')[[nom_produit_col, 'QUANTITE']]
                    top10 = top10.drop_duplicates(subset=[nom_produit_col])
                    
                    fig_top = px.bar(
                        top10, x='QUANTITE', y=nom_produit_col, orientation='h',
                        title="Top 10 produits en stock",
                        color='QUANTITE', color_continuous_scale='Blues',
                        text=top10['QUANTITE'].apply(lambda x: f"{x:,.0f}")
                    )
                    fig_top.update_traces(textposition='outside')
                    fig_top.update_layout(height=400, yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                    st.plotly_chart(fig_top, use_container_width=True)
                
                # VUE INVENTAIRE COMPLET
                elif vue == "Inventaire complet":
                    st.markdown(f"#### Inventaire détaillé - {magasin_selected}")
                    
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
                        lambda x: 'Critique' if x <= seuil_bas else ('Faible' if x <= seuil_bas*3 else 'OK')
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
                    
                    st.caption(f"{len(stock_display)} produits | Seuil critique: ≤{seuil_bas:.0f}")
                
                # VUE ALERTES
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
            
            st.markdown("---")
            
            # Répartition par magasin
            st.markdown("### Répartition par magasin")

            # Filtrer uniquement les produits finis (catégorie 9)
            if 'ID_CATEGORIE_PRODUIT' in stock_complet.columns:
                stock_produits_finis = stock_complet[stock_complet['ID_CATEGORIE_PRODUIT'] == 9].copy()
            else:
                # Fallback : utiliser la catégorisation textuelle
                stock_produits_finis = stock_complet[
                    stock_complet['Catégorie'].isin(['Jus Fruits Tropicaux', 'Jus Agrumes', 'Jus Bien-être', 'Jus Floraux', 'Sirops', 'Eau'])
                ].copy()

            if not stock_produits_finis.empty:
                stock_par_mag = stock_produits_finis.groupby(nom_magasin_col)['QUANTITE'].sum().sort_values(ascending=False)

                fig_mag = px.bar(
                    x=stock_par_mag.values,
                    y=stock_par_mag.index,
                    orientation='h',
                    color=stock_par_mag.values,
                    color_continuous_scale='Reds',
                    title="Stock de produits finis par magasin",
                    labels={'x': 'Quantité', 'y': 'Magasin'},
                    text=stock_par_mag.values
                )
                fig_mag.update_traces(textposition='outside')
                fig_mag.update_layout(height=350)
                st.plotly_chart(fig_mag, use_container_width=True)
            else:
                st.info("Aucun produit fini trouvé en stock.")

# ==========================================================
# TAB 3 : MAGASIN MATIÈRES PREMIÈRES (COMPLET)
# ==========================================================
with tab_mp:
    st.markdown("## Magasin Matières Premières")
    st.caption("Inventaire réel · Prédictions stocks & achats · Simulation de production")

    # Charger les données de stock avec catégories
    stock_mp = get_data("""
        SELECT 
            p.ID_PRODUIT,
            p.DESIGNATION,
            p.ID_CATEGORIE_PRODUIT,
            SUM(s.QUANTITE) as QUANTITE
        FROM stock s
        JOIN produit p ON s.ID_PRODUIT = p.ID_PRODUIT
        WHERE p.ID_CATEGORIE_PRODUIT IN (1, 2, 4)
        GROUP BY p.ID_PRODUIT, p.DESIGNATION, p.ID_CATEGORIE_PRODUIT
        ORDER BY p.ID_CATEGORIE_PRODUIT, p.DESIGNATION
    """)

    if stock_mp is not None and not stock_mp.empty:
        def categoriser_mp(id_cat):
            if id_cat == 1: return 'Ingrédients'
            elif id_cat == 2: return 'Fruits & Matières Premières'
            elif id_cat == 4: return 'Emballages & Étiquettes'
            else: return 'Autres'
        
        stock_mp['Catégorie'] = stock_mp['ID_CATEGORIE_PRODUIT'].apply(categoriser_mp)

        total_mp = stock_mp['QUANTITE'].sum()
        nb_references = len(stock_mp)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1: st.metric("Total Matières Premières", format_nombre(total_mp))
        with col_m2: st.metric("Références", nb_references)

        st.markdown("---")

        # Tableau des matières premières (Fruits & Ingrédients)
        st.markdown("### Matières Premières & Fruits")
        mp_fruits = stock_mp[stock_mp['ID_CATEGORIE_PRODUIT'].isin([1, 2])].copy()
        
        if not mp_fruits.empty:
            # Déterminer le stock minimum (20% du stock max)
            seuil_bas = mp_fruits['QUANTITE'].quantile(0.2) if len(mp_fruits) > 3 else 10
            
            mp_fruits['Niveau'] = mp_fruits['QUANTITE'].apply(
                lambda x: 'Critique' if x <= seuil_bas else ('Faible' if x <= seuil_bas * 3 else 'Normal')
            )
            mp_fruits['% Stock Max'] = (mp_fruits['QUANTITE'] / mp_fruits['QUANTITE'].max() * 100).round(0)
            
            st.dataframe(
                mp_fruits[['DESIGNATION', 'Catégorie', 'QUANTITE', 'Niveau', '% Stock Max']],
                use_container_width=True, hide_index=True, height=280,
                column_config={
                    'DESIGNATION': 'Matière Première',
                    'Catégorie': 'Type',
                    'QUANTITE': st.column_config.NumberColumn('Quantité en stock', format="%,.0f"),
                    'Niveau': 'Niveau de stock',
                    '% Stock Max': st.column_config.ProgressColumn('% du stock max', format="%.0f%%", min_value=0, max_value=100)
                }
            )
            
            # Alerte si produits critiques
            nb_critiques = len(mp_fruits[mp_fruits['Niveau'] == 'Critique'])
            if nb_critiques > 0:
                st.warning(f"{nb_critiques} matière(s) première(s) en niveau critique — Réapprovisionnement nécessaire.")
        else:
            st.info("Aucune matière première trouvée.")

        st.markdown("---")

        # Tableau des emballages
        st.markdown("### Emballages & Étiquettes")
        mp_emballages = stock_mp[stock_mp['ID_CATEGORIE_PRODUIT'] == 4].copy()
        
        if not mp_emballages.empty:
            seuil_bas_emb = mp_emballages['QUANTITE'].quantile(0.2) if len(mp_emballages) > 3 else 10
            
            mp_emballages['Niveau'] = mp_emballages['QUANTITE'].apply(
                lambda x: 'Critique' if x <= seuil_bas_emb else ('Faible' if x <= seuil_bas_emb * 3 else 'Normal')
            )
            mp_emballages['% Stock Max'] = (mp_emballages['QUANTITE'] / mp_emballages['QUANTITE'].max() * 100).round(0)
            
            st.dataframe(
                mp_emballages[['DESIGNATION', 'Catégorie', 'QUANTITE', 'Niveau', '% Stock Max']],
                use_container_width=True, hide_index=True, height=280,
                column_config={
                    'DESIGNATION': 'Emballage',
                    'Catégorie': 'Type',
                    'QUANTITE': st.column_config.NumberColumn('Quantité en stock', format="%,.0f"),
                    'Niveau': 'Niveau de stock',
                    '% Stock Max': st.column_config.ProgressColumn('% du stock max', format="%.0f%%", min_value=0, max_value=100)
                }
            )
        else:
            st.info("Aucun emballage trouvé.")

    else:
        st.warning("Aucune matière première trouvée dans le stock.")

    st.markdown("---")

    # ============================================
    # PRÉDICTIONS STOCKS & ACHATS
    # ============================================
    st.markdown("### Prédictions Stocks & Achats — 6 prochains mois")
    st.caption("Anticipez vos besoins en matières premières selon les saisons")

    # Fruits avec données saisonnières (clés en majuscules, sans accents)
    fruits_data = {
        'ANANAS': {
            'saison': [3,4,5,6,7], 'pic': [5,6],
            'prix_saison': 200, 'prix_hors_saison': 450,
            'conservation': '5-7 jours', 'conso_pic': 5000, 'conso_creux': 1500
        },
        'CITRON': {
            'saison': [11,12,1,2], 'pic': [12,1],
            'prix_saison': 150, 'prix_hors_saison': 350,
            'conservation': '21 jours', 'conso_pic': 3000, 'conso_creux': 800
        },
        'GINGEMBRE': {
            'saison': list(range(1,13)), 'pic': [3,4,9,10],
            'prix_saison': 500, 'prix_hors_saison': 500,
            'conservation': '30 jours', 'conso_pic': 1500, 'conso_creux': 1200
        },
        'BAOBAB': {
            'saison': list(range(1,13)), 'pic': [1,2,3],
            'prix_saison': 1000, 'prix_hors_saison': 1000,
            'conservation': '180 jours (poudre)', 'conso_pic': 500, 'conso_creux': 400
        },
        'OSEILLE': {
            'saison': [11,12,1,2,3], 'pic': [12,1],
            'prix_saison': 800, 'prix_hors_saison': 1200,
            'conservation': '365 jours (séché)', 'conso_pic': 800, 'conso_creux': 300
        },
        'COROSSOL': {
            'saison': [3,4,5,6,7,8], 'pic': [5,6],
            'prix_saison': 300, 'prix_hors_saison': 600,
            'conservation': '3-5 jours', 'conso_pic': 2000, 'conso_creux': 500
        },
        'MENTHE': {
            'saison': list(range(1,13)), 'pic': [3,4,5],
            'prix_saison': 300, 'prix_hors_saison': 400,
            'conservation': '7 jours', 'conso_pic': 600, 'conso_creux': 300
        },
    }

    # Extraire les fruits du stock (catégorie 2) - sans doublons
    fruits_dispo = stock_mp[stock_mp['ID_CATEGORIE_PRODUIT'] == 2]['DESIGNATION'].unique().tolist() if stock_mp is not None and not stock_mp.empty else list(fruits_data.keys())
    
    # Filtrer : on garde les fruits du stock qui sont dans notre dictionnaire
    fruits_selectionnables = [f for f in fruits_dispo if f.upper() in [k.upper() for k in fruits_data.keys()]]
    
    # Si aucun match, on prend tous les fruits du dictionnaire
    if not fruits_selectionnables:
        fruits_selectionnables = list(fruits_data.keys())

    # Normaliser : trouver la clé exacte dans fruits_data
    def trouver_cle(fruit):
        for k in fruits_data.keys():
            if k.upper() == fruit.upper():
                return k
        return fruit

    col_f1, col_f2 = st.columns([1, 2])
    
    with col_f1:
        fruit_selected = st.selectbox(
            "Choisir un fruit pour les prédictions",
            fruits_selectionnables,
            help="Sélectionnez un fruit pour voir ses prévisions d'achat sur 6 mois"
        )

    # Normaliser le fruit sélectionné
    fruit_selected_cle = trouver_cle(fruit_selected) if fruit_selected else None

    if fruit_selected_cle and fruit_selected_cle in fruits_data:
        info = fruits_data[fruit_selected_cle]
        mois_actuel = datetime.now().month

        with col_f2:
            en_saison = mois_actuel in info['saison']
            pic = mois_actuel in info['pic']
            
            if pic:
                statut = 'PIC de saison'
                prix_actuel = info['prix_saison']
            elif en_saison:
                statut = 'En saison'
                prix_actuel = info['prix_saison']
            else:
                statut = 'Hors saison'
                prix_actuel = info['prix_hors_saison']
            
            st.info(f"**{fruit_selected}** — {statut} | Prix estimé : **{prix_actuel} FCFA** | Conservation : **{info['conservation']}**")

        st.markdown("---")

        # Générer les prédictions pour 6 mois
        mois_futurs = []
        for i in range(6):
            m = mois_actuel + i
            if m > 12: m -= 12
            mois_futurs.append(m)

        predictions = []
        for m in mois_futurs:
            en_saison = m in info['saison']
            pic = m in info['pic']
            
            if pic:
                conso = info['conso_pic'] * 1.2
                prix = info['prix_saison']
                recommandation = "STOCKER ++ (PIC)"
            elif en_saison:
                conso = info['conso_pic']
                prix = info['prix_saison']
                recommandation = "Acheter normal"
            else:
                conso = info['conso_creux']
                prix = info['prix_hors_saison']
                recommandation = "Prix élevé — Stock minimum"
            
            predictions.append({
                'Mois': MOIS_ABBR[m],
                'Mois_Num': m,
                'Conso_Prévue': conso,
                'Prix_Prévu': prix,
                'Budget_Mensuel': conso * prix,
                'Stock_Recommande': conso / 2,
                'Recommandation': recommandation
            })

        df_pred = pd.DataFrame(predictions)

        # Graphique
        st.markdown(f"#### Prévisions 6 mois — {fruit_selected}")
        
        fig_pred = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pred.add_trace(go.Bar(
            x=df_pred['Mois'], y=df_pred['Conso_Prévue'],
            name='Conso prévue (unités)', marker_color=COLORS['primary']
        ), secondary_y=False)
        fig_pred.add_trace(go.Scatter(
            x=df_pred['Mois'], y=df_pred['Prix_Prévu'],
            name='Prix prévu (FCFA)', mode='lines+markers',
            line=dict(color=COLORS['danger'], width=3), marker=dict(size=10)
        ), secondary_y=True)
        fig_pred.update_layout(
            title=f"Consommation et prix prévus — {fruit_selected}",
            template='plotly_white', height=400, hovermode='x unified'
        )
        fig_pred.update_yaxes(title_text="Consommation (unités)", secondary_y=False)
        fig_pred.update_yaxes(title_text="Prix unitaire (FCFA)", secondary_y=True)
        st.plotly_chart(fig_pred, use_container_width=True)

        # Tableau détaillé
        st.markdown("#### Détail des prévisions")
        
        df_pred['Conso_F'] = df_pred['Conso_Prévue'].apply(lambda x: f"{x:,.0f}")
        df_pred['Prix_F'] = df_pred['Prix_Prévu'].apply(lambda x: f"{x:,.0f} FCFA")
        df_pred['Budget_F'] = df_pred['Budget_Mensuel'].apply(format_cfa)
        df_pred['Stock_F'] = df_pred['Stock_Recommande'].apply(lambda x: f"{x:,.0f}")

        st.dataframe(
            df_pred[['Mois', 'Conso_F', 'Prix_F', 'Budget_F', 'Stock_F', 'Recommandation']],
            use_container_width=True, hide_index=True,
            column_config={
                'Mois': 'Mois',
                'Conso_F': 'Conso prévue',
                'Prix_F': 'Prix prévu',
                'Budget_F': 'Budget estimé',
                'Stock_F': 'Stock recommandé',
                'Recommandation': 'Action'
            }
        )

        # Résumé budget
        budget_total = df_pred['Budget_Mensuel'].sum()
        st.metric(f"Budget total estimé sur 6 mois pour {fruit_selected}", format_cfa(budget_total))

        # Budget tous fruits
        st.markdown("---")
        st.markdown("### Budget Fruits Estimé — 6 prochains mois")
        
        budget_tous = {}
        for fruit, info in fruits_data.items():
            budget_fruit = 0
            for m in mois_futurs:
                en_saison = m in info['saison']
                pic = m in info['pic']
                if pic:
                    budget_fruit += info['conso_pic'] * 1.2 * info['prix_saison']
                elif en_saison:
                    budget_fruit += info['conso_pic'] * info['prix_saison']
                else:
                    budget_fruit += info['conso_creux'] * info['prix_hors_saison']
            budget_tous[fruit] = budget_fruit

        budget_df = pd.DataFrame([
            {'Fruit': k, 'Budget 6 mois': v} for k, v in sorted(budget_tous.items(), key=lambda x: x[1], reverse=True)
        ])
        budget_df['Budget_F'] = budget_df['Budget 6 mois'].apply(format_cfa)

        fig_budget = px.bar(
            budget_df, x='Budget 6 mois', y='Fruit', orientation='h',
            color='Budget 6 mois', color_continuous_scale='Oranges',
            title="Budget total estimé par fruit (6 mois)",
            text=budget_df['Budget_F']
        )
        fig_budget.update_traces(textposition='outside')
        fig_budget.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_budget, use_container_width=True)

        budget_global = sum(budget_tous.values())
        st.metric("Budget Global Fruits (6 mois)", format_cfa(budget_global))

    st.markdown("---")

    # ============================================
    # SIMULATEUR DE PRODUCTION
    # ============================================
    st.markdown("### Simulateur : Stocks → Production Possible")
    st.caption("Estimez combien de bouteilles vous pouvez produire avec les ressources disponibles")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Stocks disponibles**")
        stock_fruit = st.number_input("Fruits (kg)", value=500.0, step=10.0)
        stock_sucre = st.number_input("Sucre (kg)", value=200.0, step=10.0)
        stock_eau = st.number_input("Eau traitée (L)", value=5000.0, step=100.0)
        stock_aromes = st.number_input("Arômes (L)", value=20.0, step=1.0)
        stock_emballages = st.number_input("Emballages (unités)", value=10000, step=100)
        stock_etiquettes = st.number_input("Étiquettes (unités)", value=10000, step=100)

    with col_s2:
        if st.button("Calculer la production possible", type="primary", key="btn_sim_mp"):
            conso = {
                'Fruits': 0.300, 'Sucre': 0.025, 'Eau': 0.500,
                'Arômes': 0.002, 'Emballages': 1, 'Étiquettes': 1
            }
            
            stocks = [stock_fruit, stock_sucre, stock_eau, stock_aromes, stock_emballages, stock_etiquettes]
            noms = list(conso.keys())
            
            bouteilles = []
            for i, nom in enumerate(noms):
                if conso[nom] > 0:
                    bouteilles.append(int(stocks[i] / conso[nom]))
                else:
                    bouteilles.append(float('inf'))
            
            prod_max = min(bouteilles)
            limitant = noms[bouteilles.index(prod_max)]
            
            st.success(f"**Production max : {prod_max:,} bouteilles**")
            st.warning(f"Facteur limitant : **{limitant}**")
            
            # Tableau détaillé
            detail = pd.DataFrame({
                'Ressource': noms,
                'Stock disponible': [f"{s:,.1f}" for s in stocks],
                'Bouteilles possibles': [f"{b:,}" for b in bouteilles],
                'Statut': ['LIMITANT' if b == prod_max else 'OK' for b in bouteilles]
            })
            st.dataframe(detail, use_container_width=True, hide_index=True)
            
            # CA potentiel
            prix_vente = st.number_input("Prix de vente estimé/bouteille (FCFA)", value=1000, step=100, key="prix_sim_mp")
            ca_potentiel = prod_max * prix_vente
            st.metric("CA Potentiel avec ce stock", format_cfa(ca_potentiel))
            

# ==========================================================
# TAB 4 : PERFORMANCE FOURNISSEURS
# ==========================================================
with tab_fournisseurs:
    st.markdown("## Performance des Fournisseurs")
    st.caption("Suivi des commandes · Délais · Conformité · Comparaison")

    # Charger les fournisseurs
    fournisseur = get_data("SELECT * FROM fournisseur")

    if fournisseur is not None and not fournisseur.empty:
        if 'NOM_FOURNISSEUR' in fournisseur.columns:
            fournisseurs_noms = fournisseur['NOM_FOURNISSEUR'].tolist()

            # Données simulées pour les commandes
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

            # KPIs
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            with col_f1:
                st.metric("Total Fournisseurs", len(fournisseurs_noms))
            with col_f2:
                st.metric("Montant Total Commandes", format_cfa(perf_fournisseur['Montant_Commandes'].sum()))
            with col_f3:
                st.metric("Délai Moyen Global", f"{perf_fournisseur['Delai_Moyen'].mean():.1f} jours")
            with col_f4:
                st.metric("Conformité Moyenne", f"{perf_fournisseur['Taux_Conformite'].mean():.1f}%")

            st.markdown("---")

            # Graphique Top fournisseurs
            st.markdown("### Top 10 fournisseurs par montant commandé")

            fig_fourn = px.bar(
                perf_fournisseur.head(10),
                x='Nom',
                y='Montant_Commandes',
                color='Taux_Conformite',
                color_continuous_scale='RdYlGn',
                labels={'Montant_Commandes': 'Montant (FCFA)', 'Taux_Conformite': 'Conformité (%)'},
                text=perf_fournisseur.head(10)['Montant_Commandes'].apply(format_cfa)
            )
            fig_fourn.update_traces(textposition='outside')
            fig_fourn.update_layout(height=450, xaxis_tickangle=-45)
            st.plotly_chart(fig_fourn, use_container_width=True)

            st.markdown("---")

            # Graphique Délais vs Conformité
            st.markdown("### Délais de livraison vs Conformité")

            fig_delai = px.scatter(
                perf_fournisseur,
                x='Delai_Moyen',
                y='Taux_Conformite',
                size='Montant_Commandes',
                color='Nb_Commandes',
                hover_name='Nom',
                color_continuous_scale='Blues',
                title="Délai moyen vs Taux de conformité",
                labels={'Delai_Moyen': 'Délai moyen (jours)', 'Taux_Conformite': 'Conformité (%)'}
            )
            fig_delai.update_layout(height=400)
            st.plotly_chart(fig_delai, use_container_width=True)

            st.markdown("---")

            # Tableau détaillé
            st.markdown("### Détail complet des fournisseurs")

            perf_display = perf_fournisseur.copy()
            perf_display['Montant_Commandes_F'] = perf_display['Montant_Commandes'].apply(format_cfa)
            perf_display['Rang'] = range(1, len(perf_display) + 1)

            # Évaluation
            def evaluer(row):
                if row['Taux_Conformite'] >= 95 and row['Delai_Moyen'] <= 5:
                    return 'Excellent'
                elif row['Taux_Conformite'] >= 90 and row['Delai_Moyen'] <= 10:
                    return 'Bon'
                elif row['Taux_Conformite'] >= 85:
                    return 'Moyen'
                else:
                    return 'À surveiller'

            perf_display['Évaluation'] = perf_display.apply(evaluer, axis=1)

            st.dataframe(
                perf_display[['Rang', 'Nom', 'Montant_Commandes_F', 'Nb_Commandes', 'Delai_Moyen', 'Taux_Conformite', 'Évaluation']],
                use_container_width=True,
                hide_index=True,
                height=400,
                column_config={
                    'Rang': st.column_config.NumberColumn('Rang', format="%d"),
                    'Nom': 'Fournisseur',
                    'Montant_Commandes_F': 'Montant commandes',
                    'Nb_Commandes': 'Nb commandes',
                    'Delai_Moyen': 'Délai moyen (j)',
                    'Taux_Conformite': st.column_config.ProgressColumn(
                        'Conformité',
                        format="%.1f%%",
                        min_value=80,
                        max_value=100
                    ),
                    'Évaluation': 'Évaluation'
                }
            )

            # Stocker pour le rapport
            st.session_state["fournisseurs_data"] = perf_fournisseur.copy()

            # Meilleurs et moins bons
            st.markdown("---")
            col_top, col_flop = st.columns(2)

            with col_top:
                st.success(f"""
                **Meilleur fournisseur**
                
                **{perf_fournisseur.iloc[0]['Nom']}**
                - Montant : {format_cfa(perf_fournisseur.iloc[0]['Montant_Commandes'])}
                - Délai : {perf_fournisseur.iloc[0]['Delai_Moyen']:.0f} jours
                - Conformité : {perf_fournisseur.iloc[0]['Taux_Conformite']:.1f}%
                """)

            with col_flop:
                moins_bon = perf_fournisseur.sort_values('Taux_Conformite').iloc[0]
                st.warning(f"""
                **À surveiller**
                
                **{moins_bon['Nom']}**
                - Délai : {moins_bon['Delai_Moyen']:.0f} jours
                - Conformité : {moins_bon['Taux_Conformite']:.1f}%
                """)

        else:
            st.info("Structure de la table fournisseur incomplète (colonne NOM_FOURNISSEUR manquante)")
    else:
        st.info("Aucune donnée fournisseur disponible. Vérifiez la table 'fournisseur' dans la base.")

# ==========================================================
# TAB 5 : NOUVEAUX CONDITIONNEMENTS
# ==========================================================
with tab_conditionnement:
    st.markdown("## Nouveaux Conditionnements")
    st.caption("Suivi des conditionnements et pertes de production")

    # Données de conditionnement
    conditionnement = get_data("SELECT * FROM conditionnement_production")
    
    if conditionnement is not None and not conditionnement.empty:
        # Normaliser les colonnes
        conditionnement.columns = [col.upper() for col in conditionnement.columns]
        
        if 'DATE_PRODUCTION' in conditionnement.columns:
            conditionnement['DATE_PRODUCTION'] = pd.to_datetime(conditionnement['DATE_PRODUCTION'], errors='coerce')
            conditionnement['MOIS'] = conditionnement['DATE_PRODUCTION'].dt.month
            conditionnement['MOIS_NOM'] = conditionnement['MOIS'].map(MOIS_ABBR)

        # KPIs
        qte_totale = conditionnement['QUANTITE_REELLE'].sum() if 'QUANTITE_REELLE' in conditionnement.columns else 0
        nb_lots = len(conditionnement)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1: st.metric("Quantité produite", format_nombre(qte_totale))
        with col_c2: st.metric("Nombre de lots", nb_lots)

        # Pertes
        colonnes_pertes = ['PERDE_EN_BOUTEILLE', 'PERDE_EN_CAPSULE', 'PERDE_EN_ETIQUETTE', 'PERDE_EN_CARTON', 'QUANTITE_AVARIE']
        colonnes_presentes = [col for col in colonnes_pertes if col in conditionnement.columns]
        
        if colonnes_presentes:
            conditionnement['PERTES_TOTALES'] = conditionnement[colonnes_presentes].fillna(0).sum(axis=1)
            pertes_totales = conditionnement['PERTES_TOTALES'].sum()
            taux_perte = (pertes_totales / qte_totale * 100) if qte_totale > 0 else 0
            
            st.metric("Taux de perte", f"{taux_perte:.1f}%")
            
            # Graphique pertes par type
            pertes_par_type = conditionnement[colonnes_presentes].sum()
            type_labels = {
                'PERDE_EN_BOUTEILLE': 'Bouteilles', 'PERDE_EN_CAPSULE': 'Capsules',
                'PERDE_EN_ETIQUETTE': 'Étiquettes', 'PERDE_EN_CARTON': 'Cartons',
                'QUANTITE_AVARIE': 'Avariée'
            }
            pertes_par_type.index = [type_labels.get(i, i) for i in pertes_par_type.index]
            
            fig_pertes = px.pie(values=pertes_par_type.values, names=pertes_par_type.index,
                               title="Répartition des pertes par type", hole=0.4)
            fig_pertes.update_layout(height=350)
            st.plotly_chart(fig_pertes, use_container_width=True)

        # Tableau détaillé
        st.markdown("### Détail des conditionnements")
        cols_afficher = ['DATE_PRODUCTION', 'MOIS_NOM'] if 'DATE_PRODUCTION' in conditionnement.columns else []
        if 'QUANTITE_ATTENDUE' in conditionnement.columns:
            cols_afficher.append('QUANTITE_ATTENDUE')
        if 'QUANTITE_REELLE' in conditionnement.columns:
            cols_afficher.append('QUANTITE_REELLE')
        if 'PERTES_TOTALES' in conditionnement.columns:
            cols_afficher.append('PERTES_TOTALES')
        
        if cols_afficher:
            st.dataframe(
                conditionnement[cols_afficher].head(50),
                use_container_width=True, hide_index=True, height=380,
                column_config={
                    'QUANTITE_ATTENDUE': st.column_config.NumberColumn('Qté Attendue', format="%,.0f"),
                    'QUANTITE_REELLE': st.column_config.NumberColumn('Qté Réelle', format="%,.0f"),
                    'PERTES_TOTALES': st.column_config.NumberColumn('Pertes', format="%,.0f")
                }
            )
    else:
        st.info("Aucune donnée de conditionnement disponible.")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption(f"© N'NAM Jus - Pôle Production | {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
