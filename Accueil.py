import streamlit as st
import pandas as pd
import numpy as np
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
    
    annees_dispo = sorted(facture['ANNEE'].dropna().unique(), reverse=True)
    annee = st.selectbox("Année", annees_dispo, index=0)
    
    mois_dispo = sorted(facture[facture['ANNEE'] == annee]['MOIS'].unique())
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
st.markdown("## Simulation Direction")
st.caption("Évaluez l'impact d'une augmentation de capital sur la performance financière")

with st.expander("Paramètres de simulation", expanded=True):
    
    st.markdown("#### Données actuelles de l'entreprise")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        capital_actuel = st.number_input(
            "Capital Actuel (FCFA)",
            value=10000000, step=1000000, format="%d",
            help="Capital social actuel de l'entreprise"
        )
    with col_s2:
        augmentation = st.number_input(
            "Augmentation envisagée (FCFA)",
            value=5000000, step=1000000, format="%d",
            help="Montant de l'augmentation de capital que vous souhaitez simuler"
        )
    with col_s3:
        ca_actuel_input = st.number_input(
            "CA Annuel Actuel (FCFA)",
            value=int(ca_total) if ca_total > 0 else 50000000,
            step=1000000, format="%d",
            help="Chiffre d'affaires annuel de référence"
        )
    with col_s4:
        charges_fixes = st.number_input(
            "Charges Fixes Annuelles (FCFA)",
            value=15000000, step=1000000, format="%d",
            help="Loyer, salaires, électricité, maintenance, etc."
        )
    
    st.markdown("#### Hypothèses de projection")
    col_h1, col_h2, col_h3 = st.columns(3)
    
    with col_h1:
        croissance_ca = st.slider(
            "Croissance du CA attendue (%)",
            0, 100, 20, 5,
            help="Augmentation estimée du chiffre d'affaires grâce à l'investissement"
        )
    with col_h2:
        charges_var_pct = st.slider(
            "Charges Variables (% du CA)",
            0, 100, 45, 5,
            help="Matières premières, emballages, transport (en % du CA)"
        )
    with col_h3:
        reduction_cv = st.slider(
            "Gain d'efficacité (% réduction CV)",
            0, 30, 5, 1,
            help="Réduction des charges variables grâce aux nouveaux équipements"
        )

# Bouton de calcul
if st.button("Lancer la Simulation", type="primary", use_container_width=True):
    
    # ============================================
    # CALCULS
    # ============================================
    nouveau_capital = capital_actuel + augmentation
    nouveau_ca = ca_actuel_input * (1 + croissance_ca / 100)
    
    # Charges variables actuelles et nouvelles
    cv_actuelles = ca_actuel_input * (charges_var_pct / 100)
    new_cv_pct = charges_var_pct * (1 - reduction_cv / 100)
    nouvelles_cv = nouveau_ca * (new_cv_pct / 100)
    
    # Amortissement du nouvel investissement (15% par an)
    amortissement = augmentation * 0.15
    nouvelles_cf = charges_fixes + amortissement
    
    # Résultats
    resultat_actuel = ca_actuel_input - cv_actuelles - charges_fixes
    nouveau_resultat = nouveau_ca - nouvelles_cv - nouvelles_cf
    
    # Seuil de rentabilité
    taux_marge_cv = 1 - (new_cv_pct / 100)
    seuil_rentabilite = nouvelles_cf / taux_marge_cv if taux_marge_cv > 0 else 0
    
    # ROI et délai de récupération
    delta_resultat = nouveau_resultat - resultat_actuel
    roi = (delta_resultat / augmentation * 100) if augmentation > 0 else 0
    delai_recup = (augmentation / delta_resultat) if delta_resultat > 0 else float('inf')
    
    # Rentabilité des capitaux propres
    rcp_avant = (resultat_actuel / capital_actuel * 100) if capital_actuel > 0 else 0
    rcp_apres = (nouveau_resultat / nouveau_capital * 100) if nouveau_capital > 0 else 0
    
    # Marge nette
    marge_nette_avant = (resultat_actuel / ca_actuel_input * 100) if ca_actuel_input > 0 else 0
    marge_nette_apres = (nouveau_resultat / nouveau_ca * 100) if nouveau_ca > 0 else 0
    
    # Point mort en mois de CA
    point_mort_mois = (seuil_rentabilite / (nouveau_ca / 12)) if nouveau_ca > 0 else 0
    
    # ============================================
    # AFFICHAGE DES RÉSULTATS
    # ============================================
    st.markdown("---")
    st.markdown("## Résultats de la Simulation")
    
    # KPIs comparatifs
    st.markdown("### Indicateurs Financiers")
    
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    
    with col_r1:
        st.metric(
            "CA Projeté",
            format_cfa(nouveau_ca),
            delta=f"+{format_cfa(nouveau_ca - ca_actuel_input)}",
            help="Chiffre d'affaires annuel après investissement"
        )
    with col_r2:
        st.metric(
            "Résultat Net",
            format_cfa(nouveau_resultat),
            delta=f"{format_cfa(delta_resultat)} vs actuel",
            delta_color="normal" if delta_resultat >= 0 else "inverse",
            help="Bénéfice ou perte après toutes les charges"
        )
    with col_r3:
        st.metric(
            "Seuil de Rentabilité",
            format_cfa(seuil_rentabilite),
            help="CA minimum à atteindre pour couvrir toutes les charges"
        )
    with col_r4:
        st.metric(
            "ROI",
            f"{roi:.1f}%",
            help="Retour sur investissement annuel"
        )
    
    # Détail des marges
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Marge Nette Avant", f"{marge_nette_avant:.1f}%")
    with col_m2:
        st.metric("Marge Nette Après", f"{marge_nette_apres:.1f}%",
                 delta=f"{marge_nette_apres - marge_nette_avant:+.1f} pts")
    with col_m3:
        st.metric("RCP Après", f"{rcp_apres:.1f}%",
                 delta=f"{rcp_apres - rcp_avant:+.1f} pts",
                 help="Rentabilité des Capitaux Propres")
    
    # ============================================
    # INTERPRÉTATION DÉTAILLÉE
    # ============================================
    st.markdown("---")
    st.markdown("### Analyse Détaillée")
    
    # Explication de la structure des coûts
    st.markdown(f"""
    <div class="explication-box">
    <b>Structure financière projetée :</b><br><br>
    
    <b>1. Chiffre d'Affaires :</b> {format_cfa(nouveau_ca)} 
    (soit une hausse de <b>{croissance_ca}%</b> grâce à l'investissement de {format_cfa(augmentation)})<br><br>
    
    <b>2. Charges Variables :</b> {format_cfa(nouvelles_cv)} 
    ({new_cv_pct:.1f}% du CA, contre {charges_var_pct}% actuellement, soit <b>{reduction_cv}% d'économies</b>)<br>
    <small> Cette réduction provient de l'efficacité des nouveaux équipements financés par l'investissement.</small><br><br>
    
    <b>3. Charges Fixes :</b> {format_cfa(nouvelles_cf)}
    (dont {format_cfa(amortissement)} d'amortissement du nouvel investissement)<br>
    <small> L'amortissement représente 15% de l'investissement par an, étalé sur la durée de vie des équipements.</small><br><br>
    
    <b>4. Résultat Net :</b> {format_cfa(nouveau_resultat)}
    (CA − Charges Variables − Charges Fixes)<br>
    <small> Le résultat net s'améliore de {format_cfa(delta_resultat)} par rapport à la situation actuelle.</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Interprétation du seuil de rentabilité
    st.markdown(f"""
    <div class="explication-box">
    <b>Analyse du Seuil de Rentabilité :</b><br><br>
    
    Le seuil de rentabilité est de <b>{format_cfa(seuil_rentabilite)}</b>.<br>
    Cela signifie que l'entreprise doit réaliser au minimum ce chiffre d'affaires pour couvrir l'ensemble de ses charges
    (fixes + variables).<br><br>
    
    Avec un CA projeté de <b>{format_cfa(nouveau_ca)}</b>, le seuil est atteint en 
    <b>{point_mort_mois:.1f} mois</b> d'activité.<br>
    La marge de sécurité est de <b>{format_cfa(nouveau_ca - seuil_rentabilite)}</b> 
    ({(nouveau_ca - seuil_rentabilite) / nouveau_ca * 100:.1f}% du CA).
    </div>
    """, unsafe_allow_html=True)
    
    # Interprétation du ROI
    st.markdown(f"""
    <div class="explication-box">
    <b>Analyse du Retour sur Investissement :</b><br><br>
    
    Le ROI est de <b>{roi:.1f}%</b> par an.<br>
    Cela signifie que chaque franc investi génère <b>{roi:.1f} francs</b> de résultat supplémentaire par an.<br><br>
    
    Le délai de récupération de l'investissement est estimé à 
    <b>{delai_recup:.1f} an(s)</b>{" (investissement non récupérable en l'état)" if delai_recup == float('inf') else ""}.
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # AVIS DÉCISIONNEL
    # ============================================
    st.markdown("---")
    st.markdown("### Avis Décisionnel")
    
    if nouveau_resultat > resultat_actuel and roi > 15 and delai_recup <= 3:
        st.markdown(f"""
        <div class="resultat-box">
        <b>RECOMMANDATION FAVORABLE : Procéder à l'investissement</b><br><br>
        
        L'augmentation de capital de <b>{format_cfa(augmentation)}</b> est <b>recommandée</b> pour les raisons suivantes :<br><br>
        
        Le résultat net progresse de <b>{format_cfa(delta_resultat)}</b> (+{marge_nette_apres - marge_nette_avant:.1f} pts de marge nette)<br>
        Le ROI de <b>{roi:.1f}%</b> est supérieur au seuil minimum de 15%<br>
        L'investissement est récupéré en <b>{delai_recup:.1f} an(s)</b><br>
        La rentabilité des capitaux propres passe de {rcp_avant:.1f}% à <b>{rcp_apres:.1f}%</b><br>
        Le seuil de rentabilité de {format_cfa(seuil_rentabilite)} est largement couvert par le CA projeté<br><br>
        
        <b>Prochaine étape :</b> Présenter ce dossier au conseil d'administration pour validation.
        </div>
        """, unsafe_allow_html=True)
        
    elif nouveau_resultat > resultat_actuel:
        st.markdown(f"""
        <div class="alerte-box">
        <b>RECOMMANDATION MODÉRÉE : Étudier plus en détail</b><br><br>
        
        L'investissement de <b>{format_cfa(augmentation)}</b> améliore le résultat mais présente un ROI modéré.<br><br>
        
        Le résultat net progresse de <b>{format_cfa(delta_resultat)}</b><br>
        Le ROI de <b>{roi:.1f}%</b> est inférieur aux attentes<br>
        Le délai de récupération de <b>{delai_recup:.1f} ans</b> est long<br><br>
        
        <b>Suggestions d'amélioration :</b><br>
        • Augmenter la part allouée au développement commercial pour booster le CA<br>
        • Négocier de meilleures conditions d'achat pour réduire les charges variables<br>
        • Revoir le montant de l'investissement à la baisse
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown(f"""
        <div class="alerte-box">
        <b>RECOMMANDATION DÉFAVORABLE : Ne pas investir en l'état</b><br><br>
        
        L'investissement de <b>{format_cfa(augmentation)}</b> n'améliore pas suffisamment la situation.<br><br>
        
        Le résultat net ne progresse que de <b>{format_cfa(delta_resultat)}</b> (insuffisant)<br>
        Le ROI est négatif ou trop faible ({roi:.1f}%)<br><br>
        
        <b>Actions à mener avant d'investir :</b><br>
        • Réduire les charges fixes avant d'engager de nouveaux investissements<br>
        • Optimiser la production actuelle pour améliorer la rentabilité<br>
        • Chercher des financements alternatifs (subventions, crédits à taux réduit)
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # GRAPHIQUE COMPARATIF
    # ============================================
    st.markdown("---")
    st.markdown("### Comparaison Visuelle")
    
    fig_comp = go.Figure()
    
    categories = ['Capital', 'CA Annuel', 'Résultat Net', 'Seuil Rentabilité']
    valeurs_avant = [capital_actuel, ca_actuel_input, resultat_actuel, 
                    charges_fixes / (1 - charges_var_pct/100) if (1 - charges_var_pct/100) > 0 else 0]
    valeurs_apres = [nouveau_capital, nouveau_ca, nouveau_resultat, seuil_rentabilite]
    
    fig_comp.add_trace(go.Bar(
        x=categories,
        y=valeurs_avant,
        name='Avant Investissement',
        marker_color='#ff7f0e',
        textposition='none',
        hovertemplate='<b>%{x}</b><br>Avant: %{customdata}<extra></extra>',
        customdata=[format_cfa(v) for v in valeurs_avant]
    ))
    
    fig_comp.add_trace(go.Bar(
        x=categories,
        y=valeurs_apres,
        name='Après Investissement',
        marker_color='#2ca02c',
        textposition='none',
        hovertemplate='<b>%{x}</b><br>Après: %{customdata}<extra></extra>',
        customdata=[format_cfa(v) for v in valeurs_apres]
    ))
    
    fig_comp.update_layout(
        title="Comparaison Avant / Après Investissement",
        barmode='group',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption(f"© N'NAM Jus - Tableau de Bord Direction | Données actualisées le {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
with col_f2:
    st.caption("PÔLE STRATÉGIE & PILOTAGE")
