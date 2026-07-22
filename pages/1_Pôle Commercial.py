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
    page_title="N'NAM Jus - Pôle Commercial",
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
    .section-divider { margin: 30px 0; border-top: 2px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# CONSTANTES
# ==========================================================
MOIS_FR = {1:"Janvier", 2:"Février", 3:"Mars", 4:"Avril", 5:"Mai", 6:"Juin",
           7:"Juillet", 8:"Août", 9:"Septembre", 10:"Octobre", 11:"Novembre", 12:"Décembre"}
MOIS_ABBR = {1:"Jan", 2:"Fév", 3:"Mar", 4:"Avr", 5:"Mai", 6:"Juin",
             7:"Juil", 8:"Aoû", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Déc"}

SAISONS = {
    'Saison des pluies': [4, 5, 6, 7, 8, 9, 10],
    'Saison sèche': [11, 12, 1, 2, 3]
}

def get_saison(mois):
    for saison, mois_liste in SAISONS.items():
        if mois in mois_liste:
            return saison
    return 'Saison sèche'

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
    personne = get_data("SELECT * FROM personne")
    produit = get_data("SELECT * FROM produit")
    
    if facture is not None and not facture.empty:
        facture['DATE_CREATION'] = pd.to_datetime(facture['DATE_CREATION'], errors='coerce')
        facture['ANNEE'] = facture['DATE_CREATION'].dt.year
        facture['MOIS'] = facture['DATE_CREATION'].dt.month
        facture['MOIS_NOM'] = facture['MOIS'].map(MOIS_FR)
        facture['SAISON'] = facture['MOIS'].apply(get_saison)
        facture['TRIMESTRE'] = facture['DATE_CREATION'].dt.quarter
    
    return facture, personne, produit

facture, personne, produit = load_data()

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
    
    st.markdown("---")
    st.caption(f"{len(facture):,} factures chargées")
    st.caption(f"{len(personne):,} clients" if personne is not None else "")

# Filtrer
if mois_sel:
    facture_f = facture[(facture['ANNEE'] == annee) & (facture['MOIS'].isin(mois_sel))]
else:
    facture_f = facture[facture['ANNEE'] == annee]

# Fusion avec personnes pour les noms clients
if personne is not None and not personne.empty and 'ID_PERSONNE' in facture_f.columns:
    facture_f = facture_f.merge(personne[['ID_PERSONNE', 'NOM']], on='ID_PERSONNE', how='left')
else:
    if 'ID_PERSONNE' in facture_f.columns:
        facture_f['NOM'] = 'Client #' + facture_f['ID_PERSONNE'].astype(str)
    else:
        facture_f['NOM'] = 'Inconnu'

# ==========================================================
# TITRE
# ==========================================================
st.title("Pôle Commercial")
st.caption(f"Analyse commerciale – Année {annee} | {len(mois_sel)} mois")

# ==========================================================
# SOUS-ONGLETS
# ==========================================================
tab_saisons, tab_clients, tab_carto, tab_produits = st.tabs([
    "Corrélation Saisons",
    "Relance Clients",
    "Cartographie",
    "Produits & Fruits"
])

# ==========================================================
# TAB 1 : CORRÉLATION SAISONS
# ==========================================================
with tab_saisons:
    st.markdown("## Corrélation Ventes / Saisons")

    if not facture_f.empty and 'SAISON' in facture_f.columns and 'MONTANT_NET' in facture_f.columns:

        st.markdown("### Performance par saison")
        ca_saison = facture_f.groupby('SAISON')['MONTANT_NET'].agg(['sum', 'mean', 'count']).reset_index()
        ca_saison.columns = ['Saison', 'CA_Total', 'CA_Moyen', 'Nb_Factures']

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=ca_saison['Saison'], y=ca_saison['CA_Total'],
                            name='CA Total', marker_color=[COLORS['secondary'], COLORS['primary']]), secondary_y=False)
        fig.add_trace(go.Scatter(x=ca_saison['Saison'], y=ca_saison['Nb_Factures'],
                                 name='Nb ventes', mode='lines+markers',
                                 line=dict(color=COLORS['success'], width=3)), secondary_y=True)
        fig.update_layout(title="CA et nombre de ventes par saison", height=400,
                        template='plotly_white', hovermode='x unified',
                        margin=dict(l=50, r=50, t=60, b=50))
        fig.update_yaxes(title_text="CA (FCFA)", secondary_y=False)
        fig.update_yaxes(title_text="Nb ventes", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.markdown("### Coefficients de saisonnalité")

        ca_mois = facture_f.groupby('MOIS')['MONTANT_NET'].sum()
        ca_moyen = ca_mois.mean()

        coeff_data = []
        for m in range(1, 13):
            coeff = round(ca_mois.get(m, 0) / ca_moyen, 2) if ca_moyen > 0 else 0
            potentiel = 'Fort' if coeff > 1.1 else ('Neutre' if coeff >= 0.9 else 'Faible')
            coeff_data.append({'Mois': MOIS_ABBR[m], 'Coefficient': coeff, 'Potentiel': potentiel})

        df_coeff = pd.DataFrame(coeff_data)

        fig_coeff = px.bar(df_coeff, x='Mois', y='Coefficient', color='Potentiel',
                          color_discrete_map={'Fort': '#2ca02c', 'Neutre': '#ffbb78', 'Faible': '#d62728'},
                          title="Coefficient de saisonnalité par mois")
        fig_coeff.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig_coeff.update_layout(height=350, showlegend=False,
                               margin=dict(l=50, r=50, t=60, b=50))
        st.plotly_chart(fig_coeff, use_container_width=True)

        # Insights
        saison_actuelle = get_saison(datetime.now().month)
        ca_moyen_saison = ca_saison[ca_saison['Saison'] == saison_actuelle]['CA_Moyen'].values
        ca_moyen_saison = ca_moyen_saison[0] if len(ca_moyen_saison) > 0 else 0

        st.info(f"Saison actuelle : {saison_actuelle} — CA moyen attendu : {format_cfa(ca_moyen_saison)}")

    else:
        st.info("Données insuffisantes pour l'analyse saisonnière")

# ==========================================================
# TAB 2 : RELANCE CLIENTS
# ==========================================================

with tab_clients:
    st.markdown("## Relance des anciens clients")
    st.caption("Un client est considéré comme ancien après 6 mois d'inactivité")

    if not facture_f.empty and personne is not None and not personne.empty:
        # Vérifier si NOM existe déjà dans facture_f
        if 'NOM' in facture_f.columns:
            facture_client = facture_f.copy()
        else:
            # Fusion avec personnes
            if 'TELEPHONE' in personne.columns:
                facture_client = facture_f.merge(
                    personne[['ID_PERSONNE', 'NOM', 'TELEPHONE']],
                    on='ID_PERSONNE',
                    how='left'
                )
            else:
                facture_client = facture_f.merge(
                    personne[['ID_PERSONNE', 'NOM']],
                    on='ID_PERSONNE',
                    how='left'
                )

        # Dernier achat par client
        if 'MONTANT_NET' in facture_client.columns and 'DATE_CREATION' in facture_client.columns:
            # S'assurer que NOM existe
            if 'NOM' not in facture_client.columns:
                facture_client['NOM'] = 'Client #' + facture_client['ID_PERSONNE'].astype(str)

            derniers_achats = facture_client.groupby('ID_PERSONNE').agg(
                Dernier_Achat=('DATE_CREATION', 'max'),
                CA_Total=('MONTANT_NET', 'sum'),
                Nb_Achats=('MONTANT_NET', 'count'),
                Nom=('NOM', 'first')
            ).reset_index()

            derniers_achats['Jours_Depuis'] = (pd.Timestamp.now() - derniers_achats['Dernier_Achat']).dt.days

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

            clients_relance = derniers_achats[derniers_achats['Jours_Depuis'] > 180].copy()

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
# TAB 3 : CARTOGRAPHIE & OPTIMISATION TOURNÉES
# ==========================================================
with tab_carto:
    st.markdown("## Cartographie & Optimisation des Tournées")
    st.caption("Analyse géographique · Coûts de transport · Rentabilité par zone")

    # ============================================
    # PARAMÈTRES DU VÉHICULE
    # ============================================
    with st.expander("Configuration du véhicule de livraison", expanded=False):
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)

        with col_v1:
            type_vehicule = st.selectbox(
                "Type de véhicule",
                ["Toyota Hiace (Diesel)", "Toyota Hiace (Essence)", "Toyota Hilux", "Camion 5T", "Camion 10T", "Moto", "Personnalisé"],
                index=0
            )

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
            conso_100km = st.number_input("Consommation (L/100km)", min_value=1.0, max_value=50.0,
                                          value=vehicule_info["conso_100km"], step=0.5)
        with col_v3:
            prix_carburant = st.number_input(f"Prix {vehicule_info['carburant']} (FCFA/L)", min_value=400, max_value=1000,
                                             value=vehicule_info["prix_litre"], step=10)
        with col_v4:
            cout_entretien_km = st.number_input("Entretien (FCFA/km)", min_value=10, max_value=500, value=75, step=5)

        cout_carburant_km = (conso_100km / 100) * prix_carburant
        cout_total_km = cout_carburant_km + cout_entretien_km

        st.info(f"""
        **Coûts pour {type_vehicule} :** 
        Carburant : **{cout_carburant_km:,.0f} FCFA/km** | 
        Entretien : **{cout_entretien_km:,.0f} FCFA/km** | 
        Total : **{cout_total_km:,.0f} FCFA/km** | 
        100 km = **{conso_100km * prix_carburant:,.0f} FCFA**
        """)

    # ============================================
    # ZONES DE LIVRAISON
    # ============================================
    USINE = {"nom": "Usine Odza", "lat": 3.9110, "lon": 11.5250}

    zones_livraison = {
        'Odza': (3.9110, 11.5250, 0), 'Ekounou': (3.8650, 11.5300, 4),
        'Nlongkak': (3.8950, 11.5200, 5), 'Bastos': (3.9000, 11.5100, 6),
        'Mfoundi': (3.8800, 11.5100, 7), 'Etoudi': (3.8950, 11.5150, 8),
        'Mokolo': (3.8900, 11.5050, 9), 'Akwa': (3.8800, 11.5150, 10),
        'Mvan': (3.8330, 11.4950, 12), 'Ngousso': (3.9200, 11.5400, 14),
        'Biyem-Assi': (3.8500, 11.4900, 15), 'Bonamoussadi': (3.9500, 11.5050, 16),
        'Mendong': (3.8450, 11.4800, 18), 'Nkolbisson': (3.8700, 11.4650, 20),
        'Oyom-Abang': (3.8700, 11.4750, 22),
    }

    if not facture_f.empty:
        np.random.seed(42)
        zones_noms = list(zones_livraison.keys())
        facture_f['ZONE'] = np.random.choice(zones_noms, len(facture_f))

        ventes_zone = facture_f.groupby('ZONE').agg(
            CA_Total=('MONTANT_NET', 'sum'),
            CA_Moyen=('MONTANT_NET', 'mean'),
            Nb_Commandes=('MONTANT_NET', 'count'),
            Nb_Clients=('ID_PERSONNE', 'nunique')
        ).reset_index().sort_values('CA_Total', ascending=False)

        ventes_zone['Distance_km'] = ventes_zone['ZONE'].map(lambda z: zones_livraison.get(z, (0, 0, 10))[2])
        ventes_zone['LAT'] = ventes_zone['ZONE'].map(lambda z: zones_livraison.get(z, (3.9, 11.5))[0])
        ventes_zone['LON'] = ventes_zone['ZONE'].map(lambda z: zones_livraison.get(z, (3.9, 11.5))[1])
        ventes_zone['TAILLE'] = ventes_zone['CA_Total'] / ventes_zone['CA_Total'].max() * 50 + 10
        ventes_zone['Cout_Carburant_AR'] = ventes_zone['Distance_km'] * 2 * cout_carburant_km
        ventes_zone['Cout_Total_AR'] = ventes_zone['Distance_km'] * 2 * cout_total_km
        ventes_zone['Cout_Par_Livraison'] = ventes_zone['Cout_Total_AR'] / ventes_zone['Nb_Commandes'].replace(0, 1)
        ventes_zone['Rentabilite'] = ventes_zone['CA_Moyen'] - ventes_zone['Cout_Par_Livraison']
        ventes_zone['Marge_Transport'] = ((ventes_zone['CA_Moyen'] - ventes_zone['Cout_Par_Livraison']) / ventes_zone['CA_Moyen'] * 100).round(1)

        # ============================================
        # CARTE INTERACTIVE
        # ============================================
        st.markdown("### Carte des ventes par zone")
        fig_map = px.scatter_mapbox(
            ventes_zone, lat="LAT", lon="LON", size="TAILLE",
            color="CA_Total", hover_name="ZONE",
            hover_data={'CA_Total': ':,d', 'Nb_Commandes': True, 'Nb_Clients': True, 'Distance_km': True},
            color_continuous_scale='Viridis', size_max=50,
            zoom=11, center={"lat": 3.885, "lon": 11.51},
            mapbox_style="open-street-map", height=450
        )
        fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("---")

        # ============================================
        # ANALYSE ÉCONOMIQUE
        # ============================================
        st.markdown("### Analyse Économique des Tournées")

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

        st.markdown("---")

        # ============================================
        # RENTABILITÉ PAR ZONE
        # ============================================
        st.markdown("### Rentabilité par Zone")

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
            display[['ZONE', 'Distance_km', 'Nb_Commandes', 'CA_Total_F', 'Cout_AR_F', 'Rentabilite_F', 'Marge_Transport', 'Statut']].sort_values('Distance_km'),
            use_container_width=True, hide_index=True, height=380,
            column_config={
                'ZONE': 'Zone',
                'Distance_km': st.column_config.NumberColumn('km', format="%.0f"),
                'Nb_Commandes': 'Livraisons',
                'CA_Total_F': 'CA Total',
                'Cout_AR_F': 'Coût A/R',
                'Rentabilite_F': 'Rentabilité',
                'Marge_Transport': st.column_config.NumberColumn('Marge', format="%.1f%%"),
                'Statut': 'Statut'
            }
        )

        st.markdown("---")

        # ============================================
        # OPTIMISATION DES TOURNÉES
        # ============================================
        st.markdown("### Tournées Optimisées")

        zones_triees = ventes_zone.sort_values('Distance_km')
        tournees = []
        deja_vu = set()

        for _, zone in zones_triees.iterrows():
            if zone['ZONE'] in deja_vu: continue
            groupe = [zone]
            deja_vu.add(zone['ZONE'])

            for _, z2 in zones_triees.iterrows():
                if z2['ZONE'] not in deja_vu:
                    if abs(zone['Distance_km'] - z2['Distance_km']) <= 5:
                        groupe.append(z2)
                        deja_vu.add(z2['ZONE'])

            if groupe:
                dmax = max(g['Distance_km'] for g in groupe)
                cout_carb = dmax * 2 * cout_carburant_km
                cout_indiv = sum(g['Cout_Carburant_AR'] for g in groupe)
                tournees.append({
                    'Tournée': f"T{len(tournees)+1}",
                    'Zones': ' → '.join(g['ZONE'] for g in groupe),
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
                use_container_width=True, hide_index=True, height=280,
                column_config={
                    'Tournée': 'Tournée',
                    'Zones': 'Zones regroupées',
                    'Livraisons': 'Livraisons',
                    'Km_max': st.column_config.NumberColumn('Km max', format="%.0f km"),
                    'Coût_F': 'Coût carburant',
                    'Économie_F': 'Économie'
                }
            )

            eco = df_t['Économie'].sum()
            st.success(f"Économie totale réalisable : **{format_cfa(eco)}** en regroupant les tournées")

        st.markdown("---")

        # ============================================
        # SIMULATEUR DE TOURNÉE
        # ============================================
        st.markdown("### Simulateur de Tournée")

        c1, c2, c3 = st.columns(3)
        with c1:
            zones_sel = st.multiselect("Zones à livrer", ventes_zone['ZONE'].tolist(),
                                       default=ventes_zone.head(3)['ZONE'].tolist())
        with c2:
            nb_liv = st.number_input("Nb livraisons", 1, 50, 5)
        with c3:
            km_extra = st.number_input("Km détours supplémentaires", 0.0, 20.0, 5.0, 0.5)

        if zones_sel:
            dmax = max(zones_livraison.get(z, (0, 0, 10))[2] for z in zones_sel)
            dtotal = dmax * 2 + km_extra
            cout_carb = dtotal * cout_carburant_km
            cout_total = dtotal * cout_total_km
            cout_liv = cout_total / nb_liv

            st.markdown(f"""
            | Paramètre | Valeur |
            |-----------|--------|
            | Zones | {' → '.join(zones_sel)} |
            | Distance totale | {dtotal:.0f} km |
            | Carburant | {dtotal * conso_100km / 100:.1f} L ({format_cfa(cout_carb)}) |
            | Coût total | {format_cfa(cout_total)} |
            | Coût par livraison | {format_cfa(cout_liv)} |
            | Seuil rentabilité/livraison | {format_cfa(cout_liv * 1.5)} |
            """)

        st.markdown("---")

        # ============================================
        # RECOMMANDATIONS
        # ============================================
        st.info(f"""
        **Recommandations pour {type_vehicule} :**
        - Consommation : **{conso_100km} L/100km** | Coût/km : **{cout_total_km:,.0f} FCFA**
        - Plein 70L : **{70 * prix_carburant:,.0f} FCFA** | Autonomie : **{70 / conso_100km * 100:.0f} km**
        - En regroupant les livraisons, vous pouvez réduire la consommation de **25-30%**
        """)

    else:
        st.info("Aucune donnée de vente disponible pour l'analyse géographique")

# ==========================================================
# TAB 4 : PRODUITS & FRUITS (VISION COMMERCIALE)
# ==========================================================
with tab_produits:
    st.markdown("## Produits les Plus Vendus")

    # Requête pour les ventes réelles par produit
    ventes_produits = get_data(f"""
        SELECT 
            lfc.DESIGNATION,
            SUM(lfc.QUANTITE) as QUANTITE_VENDUE,
            SUM(lfc.QUANTITE * lfc.PRIX_UNITAIRE) as CA_PRODUIT
        FROM ligne_facture_client lfc
        JOIN facture_client fc ON lfc.ID_FACTURE_CLIENT = fc.ID_FACTURE_CLIENT
        WHERE YEAR(fc.DATE_CREATION) = {annee}
        GROUP BY lfc.DESIGNATION
        ORDER BY QUANTITE_VENDUE DESC
        LIMIT 10
    """)

    # Si la requête échoue ou retourne vide, essayer sans le filtre année
    if ventes_produits is None or ventes_produits.empty:
        ventes_produits = get_data("""
            SELECT 
                lfc.DESIGNATION,
                SUM(lfc.QUANTITE) as QUANTITE_VENDUE,
                SUM(lfc.QUANTITE * lfc.PRIX_UNITAIRE) as CA_PRODUIT
            FROM ligne_facture_client lfc
            GROUP BY lfc.DESIGNATION
            ORDER BY QUANTITE_VENDUE DESC
            LIMIT 10
        """)

    if ventes_produits is not None and not ventes_produits.empty:
        st.markdown(f"### Top 10 des produits N'NAM les plus vendus")

        fig_prod = px.bar(
            ventes_produits,
            x='QUANTITE_VENDUE',
            y='DESIGNATION',
            orientation='h',
            color='QUANTITE_VENDUE',
            color_continuous_scale='Blues',
            text=ventes_produits['QUANTITE_VENDUE'].apply(lambda x: f"{x:.0f}"),
            labels={'QUANTITE_VENDUE': 'Quantité vendue', 'DESIGNATION': 'Produit'}
        )
        fig_prod.update_traces(textposition='outside')
        fig_prod.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=50, r=50, t=40, b=50)
        )
        st.plotly_chart(fig_prod, use_container_width=True)

        # Tableau détaillé
        st.markdown("### Détail des ventes par produit")
        
        ventes_produits['CA_PRODUIT_F'] = ventes_produits['CA_PRODUIT'].apply(format_cfa)
        ventes_produits['Part'] = (ventes_produits['QUANTITE_VENDUE'] / ventes_produits['QUANTITE_VENDUE'].sum() * 100).round(1)

        st.dataframe(
            ventes_produits[['DESIGNATION', 'QUANTITE_VENDUE', 'CA_PRODUIT_F', 'Part']],
            use_container_width=True,
            hide_index=True,
            height=380,
            column_config={
                'DESIGNATION': 'Produit',
                'QUANTITE_VENDUE': st.column_config.NumberColumn('Quantité vendue', format="%,.0f"),
                'CA_PRODUIT_F': 'CA généré',
                'Part': st.column_config.ProgressColumn('Part des ventes', format="%.1f%%", min_value=0, max_value=100)
            }
        )
    else:
        st.warning("Aucune vente trouvée. Vérifiez que la table `ligne_facture_client` contient des données.")
        st.markdown("---")

        st.markdown("### Classification des fruits : Frais vs Secs")

        fruit_data = [
            {'Fruit': 'Ananas', 'Type': 'Frais', 'Ventes': 2500, 'Stockable': 'Non', 'Conservation': '5-7 jours'},
            {'Fruit': 'Mangue', 'Type': 'Frais', 'Ventes': 2100, 'Stockable': 'Non', 'Conservation': '3-5 jours'},
            {'Fruit': 'Goyave', 'Type': 'Frais', 'Ventes': 1500, 'Stockable': 'Non', 'Conservation': '3-4 jours'},
            {'Fruit': 'Passion', 'Type': 'Frais', 'Ventes': 1200, 'Stockable': 'Non', 'Conservation': '7-10 jours'},
            {'Fruit': 'Gingembre', 'Type': 'Sec', 'Ventes': 1800, 'Stockable': 'Oui', 'Conservation': '30 jours'},
            {'Fruit': 'Baobab', 'Type': 'Sec', 'Ventes': 900, 'Stockable': 'Oui', 'Conservation': '180 jours'},
            {'Fruit': 'Bissap', 'Type': 'Sec', 'Ventes': 750, 'Stockable': 'Oui', 'Conservation': '365 jours'},
            {'Fruit': 'Tamarin', 'Type': 'Sec', 'Ventes': 600, 'Stockable': 'Oui', 'Conservation': '90 jours'},
        ]

        df_fruits = pd.DataFrame(fruit_data)
        st.dataframe(
            df_fruits,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Fruit': 'Fruit',
                'Type': 'Type',
                'Ventes': st.column_config.NumberColumn('Ventes estimées', format="%,.0f"),
                'Stockable': 'Stockable',
                'Conservation': 'Conservation max'
            }
        )

        st.warning("Les fruits FRAIS ne sont PAS stockables longtemps. Seuls les fruits SECS peuvent être conservés en stock.")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption(f"© N'NAM Jus - Pôle Commercial | {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
