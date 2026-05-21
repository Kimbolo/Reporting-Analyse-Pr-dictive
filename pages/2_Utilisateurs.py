import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from db import get_data, table_exists

sns.set_style("whitegrid")

# ==================================================
# TITRE & CONTEXTE
# ==================================================
st.title("Utilisateurs & Gouvernance")
st.caption("Analyse de la structure humaine et des rôles du système")

# ==================================================
# CHARGEMENT DES DONNÉES
# ==================================================

# Vérifier si la table existe
if not table_exists("utilisateur"):
    st.error("La table 'utilisateur' n'existe pas dans la base de données")
    st.info("Veuillez vérifier que la table a été correctement importée")
    st.stop()

df = get_data("SELECT * FROM utilisateur")

if df.empty:
    st.warning("Aucune donnée trouvée dans la table utilisateur")
    st.stop()

# Afficher les colonnes disponibles pour débogage (optionnel)
with st.expander("Informations techniques"):
    st.write("Colonnes disponibles:", df.columns.tolist())

# ==================================================
# DÉTECTION AUTOMATIQUE DES COLONNES
# ==================================================

# Essayer de trouver les colonnes par différents noms possibles
id_col = None
for candidate in ["ID_UTILISATEUR", "id_utilisateur", "ID_USER", "user_id", "id"]:
    if candidate in df.columns:
        id_col = candidate
        break

profil_col = None
for candidate in ["ID_PROFIL", "id_profil", "PROFIL_ID", "profile_id", "role_id"]:
    if candidate in df.columns:
        profil_col = candidate
        break

employe_col = None
for candidate in ["EMPLOYE", "est_employe", "is_employee", "status"]:
    if candidate in df.columns:
        employe_col = candidate
        break

nom_col = None
for candidate in ["NOM", "nom", "NAME", "name", "user_name"]:
    if candidate in df.columns:
        nom_col = candidate
        break

# Si aucune colonne n'est trouvée, afficher une erreur
if id_col is None:
    st.error("Impossible de trouver une colonne d'identification utilisateur")
    st.write("Colonnes disponibles:", df.columns.tolist())
    st.stop()

# Construction du DataFrame avec les colonnes trouvées
df_clean = pd.DataFrame()
df_clean["ID_UTILISATEUR"] = df[id_col]

if profil_col:
    df_clean["ID_PROFIL"] = df[profil_col]
else:
    df_clean["ID_PROFIL"] = 0
    st.info("Colonne de profil non trouvée, utilisation d'une valeur par défaut")

if employe_col:
    df_clean["EMPLOYE"] = df[employe_col]
else:
    df_clean["EMPLOYE"] = 0
    st.info("Colonne employé non trouvée, tous les utilisateurs sont considérés comme non-employés")

if nom_col:
    df_clean["NOM"] = df[nom_col]
else:
    df_clean["NOM"] = f"Utilisateur_{df[id_col]}"

# Nettoyage
df_clean = df_clean.dropna(subset=["ID_UTILISATEUR"])

# ==================================================
# MAPPING DES PROFILS
# ==================================================
profil_mapping = {
    1: "SUPER_ADMIN",
    2: "ADMINISTRATEUR",
    3: "GESTIONNAIRE DE STOCKS PRODUITS FINIS",
    4: "COORDONATION ADMINISTRATIVE",
    5: "COMPTABLE",
    6: "COMMERCIAL",
    7: "GESTIONNAIRE DE STOCKS MATIÈRE PREMIERE",
    8: "Data Analyst",
    9: "Directeur commercial",
    10: "LOGISTICIEN",
    11: "MAGASINIER"
}

# Créer une colonne avec le nom lisible du profil
df_clean["PROFIL_NOM"] = df_clean["ID_PROFIL"].map(profil_mapping)
df_clean["PROFIL_NOM"] = df_clean["PROFIL_NOM"].fillna(f"Profil_{df_clean['ID_PROFIL']}")

# ==================================================
# KPI GLOBAUX
# ==================================================
st.markdown("## Indicateurs clés")

nb_users = df_clean["ID_UTILISATEUR"].nunique()
nb_profils = df_clean["ID_PROFIL"].nunique()
taux_employes = df_clean["EMPLOYE"].mean() * 100 if employe_col else 0

col1, col2, col3 = st.columns(3)
col1.metric("Utilisateurs", nb_users)
col2.metric("Profils distincts", nb_profils)
col3.metric("Employés", f"{taux_employes:.1f} %" if employe_col else "N/A")

# ==================================================
# RÉPARTITION DES PROFILS
# ==================================================
st.markdown("## Répartition des profils utilisateurs")

profil_counts = df_clean["PROFIL_NOM"].value_counts().reset_index()
profil_counts.columns = ["Profil", "Nombre"]

if not profil_counts.empty:
    fig1 = px.bar(
        profil_counts,
        x="Profil",
        y="Nombre",
        title="Nombre d’utilisateurs par profil",
        labels={"Nombre": "Nombre d’utilisateurs", "Profil": "Profil"},
        color="Nombre",
        color_continuous_scale="Blues",
        text="Nombre"
    )

    fig1.update_traces(textposition="outside",
                       hovertemplate="<b>Profil:</b> %{x}<br><b>Utilisateurs:</b> %{y}<extra></extra>")

    fig1.update_layout(showlegend=False,
                       hovermode="closest",
                       height=500)

    st.plotly_chart(fig1, use_container_width=True)

    st.info(
        "Cette répartition permet d’identifier les profils dominants "
        "et ceux potentiellement sous‑représentés."
    )
else:
    st.info("Aucune donnée de profil disponible")

# ==================================================
# EMPLOYÉS VS NON‑EMPLOYÉS
# ==================================================
if employe_col:
    st.markdown("## Employés vs non‑employés")

    emp_counts = df_clean["EMPLOYE"].value_counts().reset_index()
    emp_counts.columns = ["Statut", "Nombre"]
    emp_counts["Statut"] = emp_counts["Statut"].map({0: "Non employé", 1: "Employé"})

    fig2 = px.pie(
        emp_counts,
        values="Nombre",
        names="Statut",
        title="Statut des utilisateurs",
        color="Statut",
        color_discrete_map={"Employé": "green", "Non employé": "red"},
        hole=0.3,
    )

    fig2.update_traces(textposition="inside",
                       textinfo="percent+label",
                       hovertemplate="<b>%{label}</b><br>Nombre: %{value}<br>Pourcentage: %{percent}<extra></extra>")

    fig2.update_layout(showlegend=True, hovermode="closest", height=500)

    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "Un fort déséquilibre entre employés et non‑employés peut poser "
        "des questions de sécurité ou de gouvernance."
    )

# ==================================================
# ANALYSE DE CONCENTRATION (RISQUE)
# ==================================================
if not profil_counts.empty:
    st.markdown("## Analyse de concentration des rôles")

    top_profil = profil_counts.iloc[0]["Profil"]
    top_count = profil_counts.iloc[0]["Nombre"]
    part_top_profil = (top_count / nb_users) * 100

    st.metric(
        "Part du profil dominant",
        f"{part_top_profil:.1f} %",
        help=f"le profil '{top_profil}' représente {top_count} utilisateurs sur {nb_users} au total."
    )

    if part_top_profil > 60:
        st.warning(
            "Un seul profil concentre une large majorité des utilisateurs. "
            "Cela peut représenter un risque organisationnel ou de sécurité."
        )
    else:
        st.success(
            "Les profils utilisateurs sont relativement bien répartis."
        )

# ==================================================
# TABLEAU UTILISATEURS (EXPLORATION)
# ==================================================
st.markdown("## Détails des utilisateurs")

# Filtres interactifs
profils_dispo = sorted(df_clean["PROFIL_NOM"].unique())
profil_filtre = st.multiselect(
    "Filtrer par profil",
    options=profils_dispo,
    default=profils_dispo if len(profils_dispo) <= 10 else profils_dispo[:10]
)

if employe_col:
    statut_filtre = st.radio(
        "Filtrer par statut",
        ["Tous", "Employé", "Non employé"],
        horizontal=True
    )

df_filtre = df_clean[df_clean["PROFIL_NOM"].isin(profil_filtre)]

if employe_col and statut_filtre == "Employé":
    df_filtre = df_filtre[df_filtre["EMPLOYE"] == 1]
elif employe_col and statut_filtre == "Non employé":
    df_filtre = df_filtre[df_filtre["EMPLOYE"] == 0]

df_display = df_filtre[["ID_UTILISATEUR", "PROFIL_NOM", "EMPLOYE", "NOM"]].copy() if nom_col else df_filtre[["ID_UTILISATEUR", "PROFIL_NOM", "EMPLOYE"]].copy()

if employe_col:
    df_display["EMPLOYE"] = df_display["EMPLOYE"].map({0: "Non employé", 1: "Employé"})

st.dataframe(
    df_display.rename(columns={
        "ID_UTILISATEUR": "ID Utilisateur",
        "PROFIL_NOM": "Profil",
        "EMPLOYE": "Statut",
        "NOM": "Nom"
    }),
    use_container_width=True,
    hide_index=True
)

# ==================================================
# SYNTHÈSE ANALYTIQUE
# ==================================================
st.markdown("## Lecture automatique & recommandations")

message = f"Le système compte **{nb_users} utilisateurs** répartis sur **{nb_profils} profils distincts**. "

if not profil_counts.empty and part_top_profil > 60:
    message += (
        "La forte concentration sur un profil dominant suggère "
        "de revoir la répartition des rôles ou les droits d’accès."
    )
else:
    message += (
        "La répartition des profils est globalement équilibrée, "
        "ce qui limite les risques organisationnels."
    )

st.info(message)

st.success("Analyse des utilisateurs mise à jour dynamiquement")