import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from db import get_data

# ==========================================================
# CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="N'NAM Jus - Accueil",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# STYLE
# ==========================================================
st.markdown("""
<style>
    .stMetricValue { font-size: 1.8rem !important; white-space: normal !important; }
    div[data-testid="column"] { min-width: 150px !important; }
    .explication-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 15px 0;
    }
    .resultat-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2ca02c;
        margin: 15px 0;
    }
    .alerte-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #ff7f0e;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# CONSTANTES
# ==========================================================
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

def format_cfa(v):
    if v is None or pd.isna(v): return "0 FCFA"
    return f"{v:,.0f}".replace(",", " ") + " FCFA"

def format_nombre(v):
    if v is None or pd.isna(v): return "0"
    return f"{v:,.0f}".replace(",", " ")

# ==========================================================
# CHARGEMENT DES DONNÉES
# ==========================================================
@st.cache_data(ttl=300, show_spinner="Chargement des données...")
def load_data():
    facture = get_data("SELECT * FROM facture_client")
    stock = get_data("SELECT * FROM stock")
    
    if facture is not None and not facture.empty:
        facture['DATE_CREATION'] = pd.to_datetime(facture['DATE_CREATION'], errors='coerce')
        facture['ANNEE'] = facture['DATE_CREATION'].dt.year
        facture['MOIS'] = facture['DATE_CREATION'].dt.month
        facture['MOIS_NOM'] = facture['MOIS'].map(MOIS_FR)
    
    return facture, stock

facture, stock = load_data()

if facture is None or facture.empty:
    st.error("Aucune donnée trouvée. Vérifiez la connexion à la base de données.")
    st.stop()

# ==========================================================
# SIDEBAR - FILTRES
# ==========================================================
with st.sidebar:
    st.markdown("## Période d'analyse")
    
    annees_dispo = list(np.sort(facture['ANNEE'].dropna().unique())[::-1])
    annee = st.selectbox("Année", annees_dispo, index=0)
    
    mois_dispo = list(np.sort(facture[facture['ANNEE'] == annee]['MOIS'].unique()))
    mois_sel = st.multiselect(
        "Mois",
        mois_dispo,
        default=mois_dispo,
        format_func=lambda x: MOIS_FR.get(x, str(x))
    )
    
    st.markdown("---")
    st.caption(f" {len(facture):,} factures chargées")
    if stock is not None and not stock.empty:
        st.caption(f" {len(stock):,} produits en stock")

# Appliquer les filtres
if mois_sel:
    facture_f = facture[(facture['ANNEE'] == annee) & (facture['MOIS'].isin(mois_sel))]
else:
    facture_f = facture[facture['ANNEE'] == annee]

# ==========================================================
# TITRE
# ==========================================================
st.title("Accueil")
st.caption(f"Vue d'ensemble – Année {annee} | {len(mois_sel)} mois analysés")

# ==========================================================
# PARTIE 1 : KPIs GLOBAUX
# ==========================================================
st.markdown("---")
st.markdown("## Indicateurs Clés de Performance")

ca_total = facture_f['MONTANT_NET'].sum() if not facture_f.empty and 'MONTANT_NET' in facture_f.columns else 0
nb_clients = facture_f['ID_PERSONNE'].nunique() if not facture_f.empty and 'ID_PERSONNE' in facture_f.columns else 0
nb_commandes = len(facture_f)
panier_moyen = ca_total / nb_clients if nb_clients > 0 else 0
stock_total = stock['QUANTITE'].sum() if stock is not None and not stock.empty and 'QUANTITE' in stock.columns else 0
taux_marge_estime = 55  # Taux de marge brut estimé

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        "Chiffre d'Affaires",
        format_cfa(ca_total),
        help="Montant total des ventes sur la période sélectionnée"
    )
with col2:
    st.metric(
        "Clients Actifs",
        format_nombre(nb_clients),
        help="Nombre de clients uniques ayant effectué au moins un achat"
    )
with col3:
    st.metric(
        "Commandes",
        format_nombre(nb_commandes),
        help="Nombre total de transactions"
    )
with col4:
    st.metric(
        "Panier Moyen",
        format_cfa(panier_moyen),
        help="Dépense moyenne par client"
    )
with col5:
    st.metric(
        "Stock Total",
        format_nombre(stock_total),
        help="Quantité totale en stock tous produits confondus"
    )
with col6:
    st.metric(
        "Marge Brute Estimée",
        f"{taux_marge_estime}%",
        help="Taux de marge brute moyen estimé (PV - Coût revient) / PV"
    )

# ==========================================================
# PARTIE 2 : ÉVOLUTION MENSUELLE
# ==========================================================
st.markdown("---")
st.markdown("## Évolution Mensuelle des Ventes")

if not facture_f.empty:
    ca_mensuel = facture_f.groupby(['MOIS', 'MOIS_NOM'])['MONTANT_NET'].agg(['sum', 'count']).reset_index()
    ca_mensuel.columns = ['Mois_Num', 'Mois', 'CA', 'Nb_Commandes']
    ca_mensuel = ca_mensuel.sort_values('Mois_Num')
    
    # Calcul de la moyenne pour référence
    moyenne_mensuelle = ca_mensuel['CA'].mean()
    
    # Graphique
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=ca_mensuel['Mois'],
        y=ca_mensuel['CA'],
        marker_color='#1f77b4',
        name='CA Mensuel',
        hovertemplate='<b>%{x}</b><br>CA: %{y:,.0f} FCFA<br>Commandes: %{customdata}<extra></extra>',
        customdata=ca_mensuel['Nb_Commandes']
    ))
    
    # Ligne de moyenne (discrète)
    fig.add_hline(
        y=moyenne_mensuelle,
        line_dash="dash",
        line_color="#052D63",
        opacity=1,
        annotation_text=f"Moyenne: {format_cfa(moyenne_mensuelle)}"
    )
    
    fig.update_layout(
        title=f"Chiffre d'Affaires Mensuel – {annee}",
        template='plotly_white',
        height=400,
        hovermode='x unified',
        showlegend=False
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="CA (FCFA)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé
    st.markdown("### Détail Mensuel")
    
    ca_mensuel['CA_Format'] = ca_mensuel['CA'].apply(format_cfa)
    ca_mensuel['Ecart_Moyenne'] = ca_mensuel['CA'] - moyenne_mensuelle
    ca_mensuel['Ecart_Format'] = ca_mensuel['Ecart_Moyenne'].apply(
        lambda x: f"+{format_cfa(x)}" if x >= 0 else f"{format_cfa(x)}"
    )
    ca_mensuel['Tendance'] = ca_mensuel['Ecart_Moyenne'].apply(
        lambda x: 'Au-dessus' if x > 0 else ('En-dessous' if x < 0 else 'Moyenne')
    )
    
    st.dataframe(
        ca_mensuel[['Mois', 'CA_Format', 'Nb_Commandes', 'Ecart_Format', 'Tendance']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'Mois': 'Mois',
            'CA_Format': 'Chiffre d\'Affaires',
            'Nb_Commandes': st.column_config.NumberColumn('Nb Commandes', format="%d"),
            'Ecart_Format': 'Écart vs Moyenne',
            'Tendance': 'Position'
        }
    )
    
    # Insight
    mois_max = ca_mensuel.loc[ca_mensuel['CA'].idxmax()]
    mois_min = ca_mensuel.loc[ca_mensuel['CA'].idxmin()]
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.success(f"""
        ** Meilleur mois : {mois_max['Mois']}**
        - CA : {format_cfa(mois_max['CA'])}
        - {int(mois_max['Nb_Commandes'])} commandes
        - {format_cfa(mois_max['Ecart_Moyenne'])} au-dessus de la moyenne
        """)
    with col_i2:
        st.warning(f"""
        ** Mois le plus faible : {mois_min['Mois']}**
        - CA : {format_cfa(mois_min['CA'])}
        - {int(mois_min['Nb_Commandes'])} commandes
        - {format_cfa(mois_min['Ecart_Moyenne'])} en-dessous de la moyenne
        """)

# ==========================================================
# PARTIE 3 : SIMULATION DIRECTION
# ==========================================================
st.markdown("---")
st.markdown("## Simulation Direction — Augmentation de Capital")
st.caption("Analyse d'impact financier · Répartition des ressources · Projection de rentabilité")

with st.expander("Paramètres de la Simulation", expanded=True):
    
    # ============================================
    # SECTION 1 : SITUATION ACTUELLE
    # ============================================
    st.markdown("### Situation Actuelle de l'Entreprise")
    
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        capital_actuel = st.number_input(
            "Capital Social Actuel (FCFA)",
            value=10000000, step=1000000, format="%d",
            help="Capital social inscrit au bilan"
        )
    with col_a2:
        ca_actuel = st.number_input(
            "Chiffre d'Affaires Annuel (FCFA)",
            value=int(ca_total) if ca_total > 0 else 50000000,
            step=1000000, format="%d",
            help="CA réalisé sur les 12 derniers mois"
        )
    with col_a3:
        resultat_actuel_input = st.number_input(
            "Résultat Net Actuel (FCFA)",
            value=5000000, step=500000, format="%d",
            help="Bénéfice ou perte nette actuelle"
        )
    
    st.markdown("---")
    
    # ============================================
    # SECTION 2 : MONTANT ET AFFECTATION
    # ============================================
    st.markdown("### Augmentation de Capital & Affectation")
    
    col_b1, col_b2 = st.columns([1, 2])
    
    with col_b1:
        augmentation = st.number_input(
            "Montant de l'Augmentation (FCFA)",
            value=5000000, step=1000000, format="%d",
            help="Montant total de l'augmentation de capital envisagée"
        )
    
    with col_b2:
        st.markdown("**Répartition de l'investissement :**")
        pct_production = st.slider("Outil productif (%)", 0, 100, 50, 5, 
                                    help="Machines, équipements, ligne d'embouteillage, maintenance")
        pct_commercial = st.slider("Développement commercial (%)", 0, 100 - pct_production, 30, 5,
                                    help="Marketing, force de vente, nouveaux points de distribution")
        pct_fdr = 100 - pct_production - pct_commercial
        st.caption(f"Fonds de roulement : **{pct_fdr}%** (trésorerie, stocks, imprévus)")
    
    # Affichage des montants
    montant_production = augmentation * pct_production / 100
    montant_commercial = augmentation * pct_commercial / 100
    montant_fdr = augmentation * pct_fdr / 100
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Production", format_cfa(montant_production))
    with col_m2:
        st.metric("Commercial", format_cfa(montant_commercial))
    with col_m3:
        st.metric("Trésorerie", format_cfa(montant_fdr))
    
    st.markdown("---")
    
    # ============================================
    # SECTION 3 : HYPOTHÈSES DÉTAILLÉES
    # ============================================
    st.markdown("### Hypothèses de Projection")
    
    st.caption("Ces hypothèses sont basées sur des ratios standards de l'industrie agroalimentaire.")
    
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    
    with col_h1:
        gain_productivite = st.slider(
            "Gain de productivité (%)",
            0, 50, 15, 5,
            help="Réduction du coût de production unitaire grâce aux nouveaux équipements"
        )
    
    with col_h2:
        hausse_ca = st.slider(
            "Hausse du CA attendue (%)",
            0, 100, 25, 5,
            help="Augmentation des ventes grâce à la force commerciale renforcée et aux nouveaux équipements"
        )
    
    with col_h3:
        duree_amortissement = st.selectbox(
            "Durée d'amortissement",
            [3, 5, 7, 10],
            index=1,
            help="Durée sur laquelle l'investissement productif est amorti (ans)"
        )
    
    with col_h4:
        taux_charges_fixes = st.slider(
            "Charges fixes / CA (%)",
            10, 50, 30, 5,
            help="Ratio charges fixes sur CA (loyer, salaires, électricité, etc.)"
        )

# ============================================
# BOUTON DE CALCUL
# ============================================
if st.button("Lancer la Simulation Financière", type="primary", use_container_width=True):
    
    # Calculs détaillés
    nouveau_capital = capital_actuel + augmentation
    
    # CA projeté
    nouveau_ca = ca_actuel * (1 + hausse_ca / 100)
    
    # Charges fixes (proportionnelles au CA pour simplifier)
    charges_fixes_actuelles = ca_actuel * taux_charges_fixes / 100
    amortissement_annuel = montant_production / duree_amortissement
    nouvelles_charges_fixes = charges_fixes_actuelles + amortissement_annuel + (montant_commercial * 0.2)
    
    # Charges variables (matières premières, emballages)
    taux_cv_base = 45  # % du CA par défaut
    charges_var_actuelles = ca_actuel * taux_cv_base / 100
    nouveau_taux_cv = taux_cv_base * (1 - gain_productivite / 100)
    nouvelles_charges_var = nouveau_ca * nouveau_taux_cv / 100
    
    # Résultats
    charges_totales_actuelles = charges_fixes_actuelles + charges_var_actuelles
    charges_totales_nouvelles = nouvelles_charges_fixes + nouvelles_charges_var
    resultat_actuel = ca_actuel - charges_totales_actuelles
    nouveau_resultat = nouveau_ca - charges_totales_nouvelles
    
    # Ratios
    marge_nette_avant = (resultat_actuel / ca_actuel * 100) if ca_actuel > 0 else 0
    marge_nette_apres = (nouveau_resultat / nouveau_ca * 100) if nouveau_ca > 0 else 0
    rcp_avant = (resultat_actuel / capital_actuel * 100) if capital_actuel > 0 else 0
    rcp_apres = (nouveau_resultat / nouveau_capital * 100) if nouveau_capital > 0 else 0
    
    # Seuil de rentabilité
    marge_cv = 1 - nouveau_taux_cv / 100
    seuil_rentabilite = nouvelles_charges_fixes / marge_cv if marge_cv > 0 else 0
    point_mort_mois = (seuil_rentabilite / (nouveau_ca / 12)) if nouveau_ca > 0 else 0
    
    # ROI
    delta_resultat = nouveau_resultat - resultat_actuel
    roi = (delta_resultat / augmentation * 100) if augmentation > 0 else 0
    delai_recup = (augmentation / delta_resultat) if delta_resultat > 0 else float('inf')
    
    # ============================================
    # AFFICHAGE DES RÉSULTATS
    # ============================================
    st.markdown("---")
    st.markdown("## Résultats de la Simulation")
    
    # KPIs principaux
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    
    with col_k1:
        st.metric("CA Projeté", format_cfa(nouveau_ca),
                 delta=f"+{format_cfa(nouveau_ca - ca_actuel)}",
                 help="Chiffre d'affaires annuel après investissement")
    with col_k2:
        st.metric("Résultat Net", format_cfa(nouveau_resultat),
                 delta=f"{format_cfa(delta_resultat)} vs actuel",
                 delta_color="normal" if delta_resultat >= 0 else "inverse")
    with col_k3:
        st.metric("Seuil de Rentabilité", format_cfa(seuil_rentabilite),
                 help="CA minimum pour couvrir toutes les charges")
    with col_k4:
        st.metric("ROI", f"{roi:.1f}%",
                 delta="Excellent" if roi > 20 else ("Bon" if roi > 10 else "Faible"),
                 help="Retour sur investissement annuel")
    
    # Ratios de rentabilité
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric("Marge Nette", f"{marge_nette_apres:.1f}%",
                 delta=f"{marge_nette_apres - marge_nette_avant:+.1f} pts")
    with col_r2:
        st.metric("RCP", f"{rcp_apres:.1f}%",
                 delta=f"{rcp_apres - rcp_avant:+.1f} pts",
                 help="Rentabilité des Capitaux Propres")
    with col_r3:
        st.metric("Point Mort", f"{point_mort_mois:.1f} mois",
                 help="Mois d'activité pour atteindre le seuil de rentabilité")
    with col_r4:
        st.metric("Délai Récupération", f"{delai_recup:.1f} an(s)" if delai_recup != float('inf') else "> 10 ans")
    
    st.markdown("---")
    
    # ============================================
    # COMPTE DE RÉSULTAT COMPLET
    # ============================================
    st.markdown("### Compte de Résultat Comparatif")
    
    cr_data = pd.DataFrame([
        {"Poste": "CHIFFRE D'AFFAIRES", "Actuel": ca_actuel, "Projeté": nouveau_ca, 
         "Variation": nouveau_ca - ca_actuel, "Type": "CA"},
        {"Poste": "Charges Variables", "Actuel": charges_var_actuelles, "Projeté": nouvelles_charges_var,
         "Variation": nouvelles_charges_var - charges_var_actuelles, "Type": "Charge"},
        {"Poste": "Marge sur CV", "Actuel": ca_actuel - charges_var_actuelles, 
         "Projeté": nouveau_ca - nouvelles_charges_var,
         "Variation": (nouveau_ca - nouvelles_charges_var) - (ca_actuel - charges_var_actuelles), "Type": "Marge"},
        {"Poste": "Charges Fixes", "Actuel": charges_fixes_actuelles, "Projeté": nouvelles_charges_fixes,
         "Variation": nouvelles_charges_fixes - charges_fixes_actuelles, "Type": "Charge"},
        {"Poste": "RÉSULTAT D'EXPLOITATION", "Actuel": resultat_actuel, "Projeté": nouveau_resultat,
         "Variation": delta_resultat, "Type": "Résultat"},
    ])
    
    cr_data['Actuel_F'] = cr_data['Actuel'].apply(format_cfa)
    cr_data['Projeté_F'] = cr_data['Projeté'].apply(format_cfa)
    cr_data['Variation_F'] = cr_data['Variation'].apply(lambda x: f"{x:+,.0f}".replace(",", " ") + " FCFA")
    cr_data['% CA Actuel'] = (cr_data['Actuel'] / ca_actuel * 100).round(1)
    cr_data['% CA Projeté'] = (cr_data['Projeté'] / nouveau_ca * 100).round(1)
    
    st.dataframe(
        cr_data[['Poste', 'Actuel_F', '% CA Actuel', 'Projeté_F', '% CA Projeté', 'Variation_F']],
        use_container_width=True, hide_index=True,
        column_config={
            'Poste': 'Poste',
            'Actuel_F': 'Actuel',
            '% CA Actuel': '% CA',
            'Projeté_F': 'Projeté',
            '% CA Projeté': '% CA',
            'Variation_F': 'Variation'
        }
    )
    
    st.markdown("---")
    
    # ============================================
    # DÉTAIL DE L'AFFECTATION DU CAPITAL
    # ============================================
    st.markdown("### Impact de l'Affectation du Capital")
    
    col_impact1, col_impact2, col_impact3 = st.columns(3)
    
    with col_impact1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1f77b4, #4facfe); 
                    padding: 20px; border-radius: 15px; color: white;">
            <h4 style="margin:0 0 10px 0;"> Outil Productif</h4>
            <h2 style="margin:0 0 5px 0;">{format_cfa(montant_production)}</h2>
            <p style="margin:0; font-size: 0.9em;">{pct_production}% de l'investissement</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <small>• Gain productivité : <b>{gain_productivite}%</b></small><br>
            <small>• Amortissement/an : <b>{format_cfa(amortissement_annuel)}</b></small><br>
            <small>• Durée : <b>{duree_amortissement} ans</b></small><br>
            <small>• Économie CV : <b>{format_cfa(charges_var_actuelles - (ca_actuel * nouveau_taux_cv / 100))}</b>/an</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col_impact2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff7f0e, #f5576c); 
                    padding: 20px; border-radius: 15px; color: white;">
            <h4 style="margin:0 0 10px 0;">Développement Commercial</h4>
            <h2 style="margin:0 0 5px 0;">{format_cfa(montant_commercial)}</h2>
            <p style="margin:0; font-size: 0.9em;">{pct_commercial}% de l'investissement</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <small>• Hausse CA visée : <b>+{hausse_ca}%</b></small><br>
            <small>• CA additionnel : <b>{format_cfa(nouveau_ca - ca_actuel)}</b></small><br>
            <small>• Coût marketing/an : <b>{format_cfa(montant_commercial * 0.2)}</b></small><br>
            <small>• Nouveaux clients estimés : <b>+{hausse_ca // 5}%</b></small>
        </div>
        """, unsafe_allow_html=True)
    
    with col_impact3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #2ca02c, #00f2fe); 
                    padding: 20px; border-radius: 15px; color: white;">
            <h4 style="margin:0 0 10px 0;">Fonds de Roulement</h4>
            <h2 style="margin:0 0 5px 0;">{format_cfa(montant_fdr)}</h2>
            <p style="margin:0; font-size: 0.9em;">{pct_fdr}% de l'investissement</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <small>• Stock sécurité : <b>{format_cfa(montant_fdr * 0.6)}</b></small><br>
            <small>• Trésorerie : <b>{format_cfa(montant_fdr * 0.4)}</b></small><br>
            <small>• Couvre <b>{point_mort_mois:.1f} mois</b> de charges fixes</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("---")
    
    # ============================================
    # DÉTAIL DES INVESTISSEMENTS PAR PRODUIT/SAVEUR
    # ============================================
    st.markdown("### Détail des Investissements par Produit & Saveur")
    st.caption("Répartition stratégique des ressources pour maximiser le retour sur investissement")
    
    # Produits N'NAM avec données de marché
    produits_data = {
        "Jus d'Ananas": {
            "part_marche": 28, "marge": 55, "croissance": "+12%", "saison": "Mars-Juillet",
            "investissement_recommande": 25, "potentiel": "Très fort",
            "actions": ["Nouvelle ligne d'embouteillage", "Campagne digitale", "Partenariat distributeurs"],
            "ca_actuel_estime": ca_actuel * 0.28,
            "ca_additionnel_estime": ca_actuel * 0.28 * 0.12
        },
        "Jus de Gingembre": {
            "part_marche": 22, "marge": 60, "croissance": "+18%", "saison": "Toute l'année",
            "investissement_recommande": 20, "potentiel": "Très fort",
            "actions": ["Renforcement marketing", "Nouveaux conditionnements", "Export sous-région"],
            "ca_actuel_estime": ca_actuel * 0.22,
            "ca_additionnel_estime": ca_actuel * 0.22 * 0.18
        },
        "Jus de Baobab": {
            "part_marche": 18, "marge": 65, "croissance": "+25%", "saison": "Toute l'année",
            "investissement_recommande": 20, "potentiel": "Excellent",
            "actions": ["Labellisation bio", "Certification export", "Partenariats internationaux"],
            "ca_actuel_estime": ca_actuel * 0.18,
            "ca_additionnel_estime": ca_actuel * 0.18 * 0.25
        },
        "Jus de Bissap": {
            "part_marche": 15, "marge": 50, "croissance": "+8%", "saison": "Novembre-Février",
            "investissement_recommande": 15, "potentiel": "Fort",
            "actions": ["Développement recettes", "Packaging premium", "Réseau hôtels/restaurants"],
            "ca_actuel_estime": ca_actuel * 0.15,
            "ca_additionnel_estime": ca_actuel * 0.15 * 0.08
        },
        "Jus de Tamarin": {
            "part_marche": 10, "marge": 58, "croissance": "+15%", "saison": "Janvier-Avril",
            "investissement_recommande": 10, "potentiel": "Bon",
            "actions": ["Notoriété de la saveur", "Dégustations points de vente", "Communication santé"],
            "ca_actuel_estime": ca_actuel * 0.10,
            "ca_additionnel_estime": ca_actuel * 0.10 * 0.15
        },
        "Jus de Citron": {
            "part_marche": 7, "marge": 48, "croissance": "+5%", "saison": "Novembre-Février",
            "investissement_recommande": 10, "potentiel": "Modéré",
            "actions": ["Optimisation coûts", "Vente en gros", "Marché professionnel"],
            "ca_actuel_estime": ca_actuel * 0.07,
            "ca_additionnel_estime": ca_actuel * 0.07 * 0.05
        },
    }
    
    # ============================================
    # TABLEAU DÉTAILLÉ PAR PRODUIT
    # ============================================
    st.markdown("#### Matrice d'Investissement par Produit")
    
    matrice_data = []
    for produit, data in produits_data.items():
        montant_investi = montant_production * data['investissement_recommande'] / 100
        ca_add = data['ca_additionnel_estime']
        roi_produit = (ca_add * data['marge'] / 100) / montant_investi * 100 if montant_investi > 0 else 0
        
        matrice_data.append({
            'Produit': produit,
            'Part de Marché': f"{data['part_marche']}%",
            'Marge Brute': f"{data['marge']}%",
            'Croissance': data['croissance'],
            'Saison Forte': data['saison'],
            'Invest. Recommandé': f"{data['investissement_recommande']}%",
            'Montant Investi': format_cfa(montant_investi),
            'CA Additionnel Estimé': format_cfa(ca_add),
            'ROI Produit': f"{roi_produit:.1f}%",
            'Potentiel': data['potentiel']
        })
    
    df_matrice = pd.DataFrame(matrice_data)
    
    st.dataframe(
        df_matrice,
        use_container_width=True, hide_index=True,
        column_config={
            'Produit': st.column_config.TextColumn('Produit', width='medium'),
            'Part de Marché': 'Part Marché',
            'Marge Brute': 'Marge',
            'Croissance': 'Croiss.',
            'Saison Forte': 'Saison',
            'Invest. Recommandé': 'Invest. %',
            'Montant Investi': 'Montant',
            'CA Additionnel Estimé': 'CA Additionnel',
            'ROI Produit': st.column_config.TextColumn('ROI', width='small'),
            'Potentiel': 'Potentiel'
        }
    )
    
    st.markdown("---")
    
    # ============================================
    # RÉPARTITION VISUELLE DU CAPITAL PAR PRODUIT
    # ============================================
    st.markdown("#### Répartition du Capital Productif par Produit")
    
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        # Treemap des investissements
        treemap_data = []
        for produit, data in produits_data.items():
            treemap_data.append({
                'Produit': produit,
                'Investissement': montant_production * data['investissement_recommande'] / 100,
                'CA Additionnel': data['ca_additionnel_estime'],
                'ROI': (data['ca_additionnel_estime'] * data['marge'] / 100) / 
                       (montant_production * data['investissement_recommande'] / 100) * 100 
                       if (montant_production * data['investissement_recommande'] / 100) > 0 else 0
            })
        
        df_treemap = pd.DataFrame(treemap_data)
        
        fig_treemap = px.treemap(
            df_treemap,
            path=['Produit'],
            values='Investissement',
            color='ROI',
            color_continuous_scale='RdYlGn',
            title="Répartition de l'investissement productif par produit",
            hover_data={'Investissement': ':,d', 'CA Additionnel': ':,d', 'ROI': ':.1f'}
        )
        fig_treemap.update_layout(height=400)
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    with col_chart2:
        # Graphique ROI par produit
        roi_data = []
        for produit, data in produits_data.items():
            montant = montant_production * data['investissement_recommande'] / 100
            ca_add = data['ca_additionnel_estime']
            roi_prod = (ca_add * data['marge'] / 100) / montant * 100 if montant > 0 else 0
            roi_data.append({'Produit': produit, 'ROI': roi_prod, 'Investissement': montant})
        
        df_roi = pd.DataFrame(roi_data).sort_values('ROI', ascending=True)
        
        fig_roi = px.bar(
            df_roi,
            x='ROI', y='Produit', orientation='h',
            color='ROI', color_continuous_scale='RdYlGn',
            title="ROI estimé par produit",
            text=df_roi['ROI'].apply(lambda x: f"{x:.1f}%"),
            labels={'ROI': 'ROI (%)', 'Produit': ''}
        )
        fig_roi.update_traces(textposition='outside')
        fig_roi.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_roi, use_container_width=True)
    
    st.markdown("---")
    
    # ============================================
    # PLAN D'ACTION DÉTAILLÉ PAR PRODUIT
    # ============================================
    st.markdown("#### Plan d'Action par Produit")
    
    for produit, data in produits_data.items():
        with st.expander(f" {produit} — {data['potentiel']} ({data['investissement_recommande']}% de l'investissement)", expanded=False):
            
            col_p1, col_p2 = st.columns([1, 2])
            
            with col_p1:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                    <b> Chiffres clés</b><br><br>
                    Part de marché : <b>{data['part_marche']}%</b><br>
                    Marge brute : <b>{data['marge']}%</b><br>
                    Croissance : <b>{data['croissance']}</b><br>
                    Saison forte : <b>{data['saison']}</b><br>
                    CA actuel estimé : <b>{format_cfa(data['ca_actuel_estime'])}</b><br>
                    CA additionnel : <b>{format_cfa(data['ca_additionnel_estime'])}</b><br>
                    Invest. alloué : <b>{format_cfa(montant_production * data['investissement_recommande'] / 100)}</b>
                </div>
                """, unsafe_allow_html=True)
            
            with col_p2:
                st.markdown("** Actions recommandées :**")
                for i, action in enumerate(data['actions']):
                    st.markdown(f"**{i+1}.** {action}")
                
                st.markdown("---")
                st.markdown("** Résultats attendus :**")
                
                invest_produit = montant_production * data['investissement_recommande'] / 100
                gain_net = data['ca_additionnel_estime'] * data['marge'] / 100
                roi_action = gain_net / invest_produit * 100 if invest_produit > 0 else 0
                
                st.markdown(f"""
                • CA additionnel : **{format_cfa(data['ca_additionnel_estime'])}**
                • Gain net estimé : **{format_cfa(gain_net)}**
                • ROI de l'action : **{roi_action:.1f}%**
                • Point mort : **{invest_produit / (gain_net / 12):.1f} mois** si gain > 0
                """)
    
    st.markdown("---")
    
    # ============================================
    # COMPARAISON DES SCÉNARIOS D'INVESTISSEMENT
    # ============================================
    st.markdown("### Comparaison des Scénarios d'Investissement")
    st.caption("Simulez différentes répartitions pour optimiser votre retour")
    
    col_sc1, col_sc2, col_sc3 = st.columns(3)
    
    with col_sc1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1f77b4, #4facfe); 
                    padding: 20px; border-radius: 15px; color: white;">
            <h4 style="margin:0 0 10px 0;"> Scénario PRUDENT</h4>
            <p style="margin:0;">Focaliser sur les 2 produits phares</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <b>Ananas : 40% | Gingembre : 35%</b><br>
            Autres : 25%<br><br>
            <b>CA estimé : """ + format_cfa(nouveau_ca * 0.85) + """</b><br>
            <small>Risque faible — Croissance modérée</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sc2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff7f0e, #f5576c); 
                    padding: 20px; border-radius: 15px; color: white;">
            <h4 style="margin:0 0 10px 0;"> Scénario CROISSANCE</h4>
            <p style="margin:0;">Répartition équilibrée</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <b>Tous produits selon part de marché</b><br>
            Nouveaux emballages inclus<br><br>
            <b>CA estimé : """ + format_cfa(nouveau_ca) + """</b><br>
            <small>Risque modéré — Croissance cible</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sc3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2ca02c, #00f2fe); 
                    padding: 20px; border-radius: 15px; color: white;">
            <h4 style="margin:0 0 10px 0;"> Scénario AMBITIEUX</h4>
            <p style="margin:0;">Forte diversification + export</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <b>Baobab : 30% | Gingembre : 30%</b><br>
            Bissap : 20% | Export : 20%<br><br>
            <b>CA estimé : """ + format_cfa(nouveau_ca * 1.2) + """</b><br>
            <small>Risque élevé — Fort potentiel</small>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # ANALYSE DU SEUIL DE RENTABILITÉ
    # ============================================
    st.markdown("### Analyse du Seuil de Rentabilité")
    
    col_s1, col_s2 = st.columns([1, 1])
    
    with col_s1:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6;">
            <b>Formule :</b> Seuil = Charges Fixes ÷ (1 - Charges Variables/CA)<br><br>
            <b>Calcul :</b><br>
            Charges Fixes : {format_cfa(nouvelles_charges_fixes)}<br>
            Charges Variables : {nouveau_taux_cv:.1f}% du CA<br>
            Marge sur CV : {marge_cv*100:.1f}%<br>
            Seuil = {format_cfa(nouvelles_charges_fixes)} ÷ {marge_cv:.2f}<br>
            <b>= {format_cfa(seuil_rentabilite)}</b>
        </div>
        """, unsafe_allow_html=True)
    
    with col_s2:
        # Jauge de couverture
        taux_couverture = (nouveau_ca / seuil_rentabilite * 100) if seuil_rentabilite > 0 else 0
        
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6;">
            <b>Interprétation :</b><br><br>
            CA Projeté : <b>{format_cfa(nouveau_ca)}</b><br>
            Seuil : <b>{format_cfa(seuil_rentabilite)}</b><br>
            Marge de sécurité : <b>{format_cfa(nouveau_ca - seuil_rentabilite)}</b><br>
            Taux de couverture : <b>{taux_couverture:.0f}%</b><br>
            Point mort atteint en : <b>{point_mort_mois:.1f} mois</b><br><br>
            {"Situation CONFORTABLE" if taux_couverture > 150 else "Situation TENUE" if taux_couverture > 110 else "Situation RISQUÉE"}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # GRAPHIQUE COMPARATIF
    # ============================================
    st.markdown("### Comparaison Visuelle")
    
    fig = go.Figure()
    
    categories = ['Capital', 'CA', 'Résultat Net', 'Charges Fixes', 'Seuil Rentab.']
    valeurs_avant = [capital_actuel, ca_actuel, resultat_actuel, charges_fixes_actuelles, 
                    charges_fixes_actuelles / (1 - taux_cv_base/100) if (1 - taux_cv_base/100) > 0 else 0]
    valeurs_apres = [nouveau_capital, nouveau_ca, nouveau_resultat, nouvelles_charges_fixes, seuil_rentabilite]
    
    fig.add_trace(go.Bar(x=categories, y=valeurs_avant, name='Avant',
                         marker_color='#ff7f0e', text=[format_cfa(v) for v in valeurs_avant],
                         textposition='outside'))
    fig.add_trace(go.Bar(x=categories, y=valeurs_apres, name='Après',
                         marker_color='#2ca02c', text=[format_cfa(v) for v in valeurs_apres],
                         textposition='outside'))
    
    fig.update_layout(
        title="Avant / Après Investissement",
        barmode='group', template='plotly_white', height=450,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption(f"© N'NAM Jus - Tableau de Bord Direction | Données actualisées le {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
with col_f2:
    st.caption("PÔLE STRATÉGIE & PILOTAGE")
