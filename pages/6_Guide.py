import streamlit as st
import pandas as pd
from datetime import datetime

# ==========================================================
# CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="N'NAM Jus - Guide & Méthodologie",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# STYLE
# ==========================================================
st.markdown("""
<style>
    .formule-box {
        background-color: #f0f4f8;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    .exemple-box {
        background-color: #e8f5e9;
        border-left: 4px solid #2ca02c;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .alerte-box {
        background-color: #fff3cd;
        border-left: 4px solid #ff7f0e;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .definition-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1565c0;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITRE
# ==========================================================
st.title("Guide, Lexique & Méthodologie")
st.caption("Dictionnaire des données · Formules · Interprétations · Seuils d'alerte")

# ==========================================================
# SOMMAIRE
# ==========================================================
with st.expander("Sommaire", expanded=True):
    st.markdown("""
    1. [Lexique des Termes](#1-lexique-des-termes)
    2. [Dictionnaire des Données](#2-dictionnaire-des-données)
    3. [Formules de Calcul](#3-formules-de-calcul)
    4. [Seuils d'Alerte](#4-seuils-dalerte)
    5. [Règles de Gestion](#5-règles-de-gestion)
    6. [Méthodologie ML](#6-méthodologie-machine-learning)
    7. [Sources de Données](#7-sources-de-données)
    """)

st.markdown("---")

# ==========================================================
# 1. LEXIQUE DES TERMES
# ==========================================================
st.markdown("## 1. Lexique des Termes")

lexique_data = {
    "Terme": [
        "CA (Chiffre d'Affaires)",
        "Client actif",
        "Commande",
        "Panier moyen client",
        "Stock total",
        "Taux de perte",
        "Seuil critique",
        "MAPE",
        "R²",
        "Coefficient de saisonnalité",
        "Seuil de rentabilité",
        "Marge transport",
        "Couverture stock",
        "Saison des pluies",
        "Saison sèche",
        "Matière première",
        "Fruit sec",
        "Fruit frais",
        "Produit fini",
        "Conditionnement",
        "Emballage",
        "ROI",
        "Facture validée",
        "Encaissement",
    ],
    "Définition": [
        "Montant total des ventes réalisées sur une période donnée, calculé à partir des factures clients validées (MONTANT_NET).",
        "Client (ID_PERSONNE unique) ayant effectué au moins un achat dans la période analysée.",
        "Regroupement des factures d'un même client. Une commande = un ou plusieurs achats.",
        "Montant moyen dépensé par client sur la période. CA Total ÷ Nombre de clients actifs.",
        "Quantité totale de produits disponibles dans l'ensemble des magasins, toutes références confondues.",
        "Pourcentage de produits perdus lors de la production par rapport à la quantité totale produite.",
        "Niveau de stock en dessous duquel un produit est considéré en risque de rupture. Calculé au 15ème percentile des stocks.",
        "Mean Absolute Percentage Error — Erreur moyenne en pourcentage des prévisions ML par rapport aux valeurs réelles.",
        "Coefficient de détermination — Indique la qualité du modèle prédictif (0 = mauvais, 1 = parfait).",
        "Ratio mesurant l'écart entre les ventes d'un mois et la moyenne mensuelle annuelle. > 1 = fort potentiel, < 1 = faible.",
        "Niveau de CA minimum à atteindre pour couvrir l'ensemble des charges (fixes + variables).",
        "Pourcentage du CA restant après déduction des coûts de transport. Indique la rentabilité d'une zone de livraison.",
        "Durée (en mois) pendant laquelle le stock actuel peut couvrir les ventes sans réapprovisionnement.",
        "Période d'avril à octobre — Précipitations fréquentes, routes difficiles, certains fruits abondants.",
        "Période de novembre à mars — Temps sec, favorable aux livraisons.",
        "Ingrédient utilisé dans la fabrication (fruits, sucre, eau, arômes, additifs).",
        "Fruit déshydraté/longue conservation (Baobab, Bissap séché, Tamarin, Gingembre). Stockable.",
        "Fruit périssable nécessitant transformation rapide (Ananas, Mangue, Pastèque). NON stockable.",
        "Jus embouteillé prêt à la vente (catégorie 9).",
        "Opération de mise en bouteille, capsulage, étiquetage et mise en carton.",
        "Matériaux de conditionnement : bouteilles PET, bouchons, étiquettes, cartons.",
        "Return On Investment — Retour sur investissement. (Gain - Coût) / Coût × 100.",
        "Facture dont le statut VALIDER = 1. Seules ces factures sont comptabilisées dans le CA.",
        "Paiement reçu correspondant à une ou plusieurs factures.",
    ]
}

df_lexique = pd.DataFrame(lexique_data)
st.dataframe(
    df_lexique,
    use_container_width=True,
    hide_index=True,
    column_config={
        'Terme': st.column_config.TextColumn('Terme', width='medium'),
        'Définition': st.column_config.TextColumn('Définition', width='large')
    }
)

st.markdown("---")

# ==========================================================
# 2. DICTIONNAIRE DES DONNÉES
# ==========================================================
st.markdown("## 2. Dictionnaire des Données")

st.markdown("### 2.1 Indicateurs de Performance (KPI)")

kpi_data = {
    "Indicateur": [
        "Chiffre d'Affaires (CA)",
        "Clients actifs",
        "Nombre de commandes",
        "CA par client",
        "CA par commande",
        "Stock total",
        "Taux de perte",
    ],
    "Formule": [
        "SUM(facture_client.MONTANT_NET) WHERE VALIDER = 1",
        "COUNT(DISTINCT facture_client.ID_PERSONNE)",
        "COUNT(facture_client) GROUP BY ID_PERSONNE",
        "CA Total ÷ Clients actifs",
        "CA Total ÷ Nombre de commandes",
        "SUM(stock.QUANTITE)",
        "(SUM(pertes) ÷ SUM(quantite_reelle)) × 100",
    ],
    "Source": [
        "facture_client",
        "facture_client",
        "facture_client",
        "Calculé",
        "Calculé",
        "stock",
        "conditionnement_production",
    ],
    "Unité": [
        "FCFA",
        "Nombre",
        "Nombre",
        "FCFA/client",
        "FCFA/commande",
        "Unités",
        "%",
    ]
}

df_kpi = pd.DataFrame(kpi_data)
st.dataframe(df_kpi, use_container_width=True, hide_index=True)

st.markdown("### 2.2 Données Sources")

sources_data = {
    "Table": [
        "facture_client",
        "ligne_facture_client",
        "stock",
        "produit",
        "magasin",
        "personne",
        "fournisseur",
        "conditionnement_production",
        "paiement_facture",
    ],
    "Contenu": [
        "Factures clients validées avec montants",
        "Détail des produits vendus par facture",
        "État des stocks par produit et magasin",
        "Catalogue produits (nom, catégorie)",
        "Liste des magasins/points de vente",
        "Données clients (nom, téléphone, adresse)",
        "Informations sur les fournisseurs",
        "Production : quantités, pertes",
        "Paiements reçus",
    ],
    "Colonnes clés": [
        "ID_FACTURE_CLIENT, DATE_CREATION, MONTANT_NET, VALIDER, ID_PERSONNE",
        "ID_LIGNE, ID_FACTURE_CLIENT, ID_PRODUIT, QUANTITE, PRIX_UNITAIRE",
        "ID_STOCK, ID_PRODUIT, ID_MAGASIN, QUANTITE",
        "ID_PRODUIT, DESIGNATION, ID_CATEGORIE_PRODUIT",
        "ID_MAGASIN, NOM_MAGASIN",
        "ID_PERSONNE, NOM, TELEPHONE, ADRESSE",
        "ID_FOURNISSEUR, NOM_FOURNISSEUR",
        "ID, DATE_PRODUCTION, QUANTITE_REELLE, QUANTITE_ATTENDUE, PERTES",
        "ID_PAIEMENT, DATE_PAIEMENT, MONTANT, ID_FACTURE_CLIENT",
    ]
}

df_sources = pd.DataFrame(sources_data)
st.dataframe(df_sources, use_container_width=True, hide_index=True)

st.markdown("---")

# ==========================================================
# 3. FORMULES DE CALCUL
# ==========================================================
st.markdown("## 3. Formules de Calcul")

with st.expander("Formules Commerciales", expanded=True):
    st.markdown("""
    <div class="formule-box">
    <b>Chiffre d'Affaires (CA)</b><br>
    CA = Σ MONTANT_NET &nbsp;&nbsp;|&nbsp;&nbsp; ∀ facture où VALIDER = 1
    </div>
    
    <div class="formule-box">
    <b>Panier Moyen Client</b><br>
    Panier = CA ÷ Nb_Clients_Actifs
    </div>
    
    <div class="formule-box">
    <b>Part de Marché par Zone</b><br>
    Part(%) = (CA_Zone ÷ CA_Total) × 100
    </div>
    
    <div class="exemple-box">
    <b> Exemple :</b><br>
    CA Total = 22 500 000 FCFA<br>
    Zone Mfoundi = 5 200 000 FCFA → Part = 23,1%
    </div>
    """)

with st.expander("Formules de Production", expanded=True):
    st.markdown("""
    <div class="formule-box">
    <b>Taux de Perte</b><br>
    Taux(%) = (Σ Pertes ÷ Σ Quantité_Réelle) × 100
    </div>
    
    <div class="formule-box">
    <b>Stock Disponible (fin de mois)</b><br>
    Stock_Fin = Stock_Initial + Production_Cumulée - Ventes_Cumulées
    </div>
    
    <div class="formule-box">
    <b>Écart Production vs Ventes</b><br>
    Écart = Volume_Produit - Volume_Vendu<br>
    Taux_Écart(%) = (Écart ÷ Volume_Produit) × 100
    </div>
    
    <div class="alerte-box">
    <b> Interprétation :</b><br>
    Écart > 0 → Surproduction (stocks qui dorment)<br>
    Écart < 0 → Sous-production (rupture potentielle)
    </div>
    """)

with st.expander("Formules de Saisonnalité", expanded=True):
    st.markdown("""
    <div class="formule-box">
    <b>Coefficient de Saisonnalité (Mois X)</b><br>
    Coeff_X = CA_Mois_X ÷ CA_Moyen_Mensuel
    </div>
    
    <div class="formule-box">
    <b>CA Moyen Mensuel</b><br>
    CA_Moyen = Σ CA_Mensuel ÷ 12
    </div>
    
    <div class="exemple-box">
    <b>Exemple :</b><br>
    CA Moyen Mensuel = 2 000 000 FCFA<br>
    CA Mai = 2 600 000 → Coeff = 1,30 (Fort potentiel)<br>
    CA Novembre = 1 500 000 → Coeff = 0,75 (Faible)
    </div>
    """)

with st.expander("Formules Transport & Logistique", expanded=True):
    st.markdown("""
    <div class="formule-box">
    <b>Coût Carburant par km</b><br>
    Coût/km = (Conso_L/100km ÷ 100) × Prix_Carburant
    </div>
    
    <div class="formule-box">
    <b>Coût Total Aller-Retour</b><br>
    Coût_AR = Distance_km × 2 × (Coût_Carburant/km + Entretien/km)
    </div>
    
    <div class="formule-box">
    <b>Marge Transport</b><br>
    Marge(%) = (CA_Moyen_Livraison - Coût_Par_Livraison) ÷ CA_Moyen_Livraison × 100
    </div>
    
    <div class="exemple-box">
    <b>Exemple (Toyota Hiace Diesel) :</b><br>
    Conso : 9,5 L/100km | Gasoil : 630 FCFA/L<br>
    Coût/km = 0,095 × 630 = 59,85 FCFA/km<br>
    Avec entretien (75 FCFA/km) → 134,85 FCFA/km<br>
    Zone à 10 km AR = 20 km → 2 697 FCFA
    </div>
    """)

with st.expander("Formules Financières", expanded=True):
    st.markdown("""
    <div class="formule-box">
    <b>Seuil de Rentabilité</b><br>
    SR = Charges_Fixes ÷ (1 - Charges_Variables/CA)
    </div>
    
    <div class="formule-box">
    <b>Résultat Net</b><br>
    Résultat = CA - Charges_Fixes - Charges_Variables
    </div>
    
    <div class="formule-box">
    <b>ROI (Retour sur Investissement)</b><br>
    ROI(%) = (Résultat_Après - Résultat_Avant) ÷ Investissement × 100
    </div>
    
    <div class="formule-box">
    <b>RCP (Rentabilité Capitaux Propres)</b><br>
    RCP(%) = Résultat_Net ÷ Capital × 100
    </div>
    
    <div class="exemple-box">
    <b> Exemple Seuil de Rentabilité :</b><br>
    Charges Fixes = 15 000 000 FCFA<br>
    Charges Variables = 45% du CA → Marge = 55%<br>
    SR = 15 000 000 ÷ 0,55 = 27 272 727 FCFA<br>
    → Il faut générer 27,3M de CA pour être à l'équilibre.
    </div>
    """)

st.markdown("---")

# ==========================================================
# 4. SEUILS D'ALERTE
# ==========================================================
st.markdown("## 4. Seuils d'Alerte")

seuils_data = {
    "Indicateur": [
        "CA Mensuel",
        "Taux d'encaissement",
        "Taux de perte",
        "Taux d'écart Production/Ventes",
        "Nombre de clients actifs",
        "Couverture stock",
        "MAPE (prévisions)",
        "R² (modèle ML)",
        "Marge transport",
        "ROI",
    ],
    "Seuil Critique": [
        "Baisse > 20% vs N-1",
        "< 70%",
        "> 10%",
        "> 25%",
        "< 10",
        "< 1 mois",
        "> 50%",
        "< 0,50",
        "< 0%",
        "< 5%",
    ],
    "Seuil Attention": [
        "Baisse 10-20%",
        "70-80%",
        "5-10%",
        "15-25%",
        "10-20",
        "1-2 mois",
        "20-50%",
        "0,50-0,70",
        "0-40%",
        "5-10%",
    ],
    "Seuil Bon": [
        "Stable ou hausse",
        "> 80%",
        "< 5%",
        "< 15%",
        "> 20",
        "> 2 mois",
        "< 20%",
        "> 0,70",
        "> 40%",
        "> 10%",
    ],
}

df_seuils = pd.DataFrame(seuils_data)
st.dataframe(df_seuils, use_container_width=True, hide_index=True)

st.markdown("---")

# ==========================================================
# 5. RÈGLES DE GESTION
# ==========================================================
st.markdown("## 5. Règles de Gestion")

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
        "Factures valides",
        "Comparaison N-1",
    ],
    "Description": [
        "Un client = un ID_PERSONNE unique. Les doublons sont exclus des comptages.",
        "Par défaut, année en cours. Toujours comparer sur des périodes équivalentes.",
        "Basée sur le calendrier camerounais : Pluies (avril-octobre), Sèche (novembre-mars).",
        "Calculé au 15ème percentile des stocks. Réapprovisionner avant d'atteindre ce seuil.",
        "Seuls les fruits SECS peuvent être stockés > 30 jours. Les FRAIS doivent être transformés sous 7 jours.",
        "Taux acceptable < 5%. Au-delà, analyser les causes (machine, opérateur, matière).",
        "Les valeurs NULL ou 0 sont exclues des moyennes pour ne pas biaiser les calculs.",
        "Nécessitent minimum 6 mois d'historique. Marge d'erreur (MAE) systématiquement affichée.",
        "Seules les factures avec VALIDER = 1 sont comptabilisées dans les analyses.",
        "Toute variation doit être comparée à la même période de l'année précédente.",
    ],
    "Impact si non respectée": [
        "Clients comptés en double → CA/client faussé",
        "Comparaisons saisonnières incorrectes",
        "Mauvaise anticipation des achats de fruits",
        "Rupture de stock → perte de ventes",
        "Pertes de fruits frais → gaspillage financier",
        "Problèmes masqués → coûts cachés",
        "Moyennes faussées → décisions erronées",
        "Décisions basées sur prévisions peu fiables",
        "CA surévalué → mauvaise analyse",
        "Tendance mal interprétée",
    ]
}

df_regles = pd.DataFrame(regles_data)
st.dataframe(
    df_regles,
    use_container_width=True,
    hide_index=True,
    column_config={
        'Règle': st.column_config.TextColumn('Règle', width='medium'),
        'Description': st.column_config.TextColumn('Description', width='large'),
        'Impact si non respectée': st.column_config.TextColumn('Impact', width='large')
    }
)

st.markdown("---")

# ==========================================================
# 6. MÉTHODOLOGIE MACHINE LEARNING
# ==========================================================
st.markdown("## 6. Méthodologie Machine Learning")

st.markdown("""
### Modèle de Prévision : Random Forest Regressor

**Principe :**
Le Random Forest est un algorithme qui crée plusieurs arbres de décision aléatoires 
et fait la moyenne de leurs prédictions. Il est robuste et fonctionne bien avec peu de données.

**Variables utilisées :**
- `ORDRE` : numéro séquentiel du mois (1, 2, 3...)
- `MOIS_SIN` : sin(2π × mois/12) → cyclicité
- `MOIS_COS` : cos(2π × mois/12) → cyclicité

**Pourquoi SIN/COS ?**
Pour que le modèle comprenne que décembre (12) est proche de janvier (1).
Sans ça, il verrait 12 comme "très loin" de 1.

**Métriques d'évaluation :**

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **MAE** | Σ\|réel - prédit\| / n | Erreur moyenne en FCFA. Plus c'est petit, mieux c'est. |
| **MAPE** | Σ\|réel - prédit\|/réel / n × 100 | Erreur en %. < 20% = bon |
| **R²** | 1 - (SS_res / SS_tot) | 0 à 1. > 0,70 = modèle fiable |

**Limites du modèle :**
- Ne prend pas en compte les événements exceptionnels (promo, crise)
- Nécessite au moins 6 mois de données
- Les prévisions sont des estimations probabilistes
- Marge d'erreur affichée via la zone de confiance (± MAE)
""")

st.markdown("---")

# ==========================================================
# 7. SOURCES DE DONNÉES
# ==========================================================
st.markdown("## 7. Sources de Données")

st.markdown("""
### Base de données : `sellams_namm`

| Table | Rôle | Fréquence MAJ |
|-------|------|---------------|
| `facture_client` | Factures et CA | Temps réel |
| `ligne_facture_client` | Détail produits vendus | Temps réel |
| `stock` | Niveaux de stock | Temps réel |
| `produit` | Catalogue produits | Rare |
| `magasin` | Points de vente | Rare |
| `personne` | Clients | À chaque nouveau client |
| `fournisseur` | Fournisseurs | Rare |
| `conditionnement_production` | Production | Quotidienne |
| `paiement_facture` | Encaissements | Temps réel |
""")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption(f"Guide & Méthodologie — N'NAM AGRO INDUSTRIE | Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y')}")
