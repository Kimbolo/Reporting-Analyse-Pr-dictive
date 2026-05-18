import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from db import get_data

sns.set_style("whitegrid")

# ==================================================
# TITRE & CONTEXTE
# ==================================================
st.title("Utilisateurs & Gouvernance")
st.caption("Analyse de la structure humaine et des rôles du système")

# ==================================================
# CHARGEMENT DES DONNÉES
# ==================================================
df = get_data("SELECT * FROM utilisateur")

# Sécurisation minimale
required_cols = ["ID_UTILISATEUR", "ID_PROFIL", "EMPLOYE"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Colonnes manquantes dans la table utilisateur : {missing}")
    st.stop()

# Nettoyage
df = df[required_cols].dropna()

# Dictionnaire de correspondance ID_PROFIL -> Nom du profil
profil_mapping = {
    2: "ADMINISTRATEUR",
    3: "GESTIONNAIRE DE STOCKS PRODUITS FINIS",
    4: "COORDONATION ADMINISTRATIVE",
    5: "COMPTABLE",
    6: "COMMERCIAL",
    7: "GESTIONNAIRE DE STOCKS MATIÈRE PREMAIRE",
    8: "Data Analyst",
    9: "Directeur commercial"
}

# Créer une colonne avec le nom lisible du profil
df["PROFIL_NOM"] = df["ID_PROFIL"].map(profil_mapping)

# Si certains IDs n'ont pas de mapping, garder l'ID par défaut
df["PROFIL_NOM"] = df["PROFIL_NOM"].fillna(df["ID_PROFIL"].astype(str))

# ==================================================
# KPI GLOBAUX
# ==================================================
st.markdown("## Indicateurs clés")

nb_users = df["ID_UTILISATEUR"].nunique()
nb_profils = df["ID_PROFIL"].nunique()
taux_employes = df["EMPLOYE"].mean() * 100

col1, col2, col3 = st.columns(3)
col1.metric("Utilisateurs", nb_users)
col2.metric("Profils distincts", nb_profils)
col3.metric("Employés", f"{taux_employes:.1f} %")

# ==================================================
# RÉPARTITION DES PROFILS
# ==================================================
st.markdown("## Répartition des profils utilisateurs")

profil_counts = df["PROFIL_NOM"].value_counts().reset_index()
profil_counts.columns = ["Profil", "Nombre"]

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

st.plotly_chart(fig1,use_container_width=True)

st.info(
    "Cette répartition permet d’identifier les profils dominants "
    "et ceux potentiellement sous‑représentés."
)

# ==================================================
# EMPLOYÉS VS NON‑EMPLOYÉS
# ==================================================
st.markdown("## Employés vs non‑employés")

emp_counts = df["EMPLOYE"].value_counts().reset_index()
emp_counts.columns = ["Statut", "Nombre"]
emp_counts["Statut"] = emp_counts["Statut"].map({0: "Non employé", 1: "Employé"})


# Graphique circulaire interactif
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

st.plotly_chart(fig2,use_container_width=True)

st.info(
    "Un fort déséquilibre entre employés et non‑employés peut poser "
    "des questions de sécurité ou de gouvernance."
)

# ==================================================
# ANALYSE DE CONCENTRATION (RISQUE)
# ==================================================
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

# Graphique complémentaire : distribution détaillée
st.markdown("### Distribution détaillée des profils")

fig3 = px.bar(
    profil_counts,
    x="Profil",
    y="Nombre",
    title="Distribution des utilisateurs par profil",
    labels={"Profil": "Profil", "Nombre": "Nombre d'utilisateurs"},
    color="Nombre",
    color_continuous_scale="Viridis",
    text="Nombre"
)

fig3.update_traces(
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Utilisateurs: %{y}<br>Part: %{customdata:.1f}%<extra></extra>",
    customdata=[[(val/nb_users)*100] for val in profil_counts["Nombre"]]
)

fig3.update_layout(
    hovermode="closest",
    height=500
)

st.plotly_chart(fig3,use_container_width=True)

# ==================================================
# TABLEAU UTILISATEURS (EXPLORATION)
# ==================================================
st.markdown("## Détails des utilisateurs")

# Filtres interactifs
profils_dispo = sorted(df["PROFIL_NOM"].unique())
profil_filtre = st.multiselect(
    "Filtrer par profil",
    options=profils_dispo,
    default=profils_dispo
)

statut_filtre = st.radio(
    "Filtrer par statut",
    ["Tous", "Employé", "Non employé"],
    horizontal=True
)

df_filtre = df[df["PROFIL_NOM"].isin(profil_filtre)]

if statut_filtre == "Employé":
    df_filtre = df_filtre[df_filtre["EMPLOYE"] == 1]
elif statut_filtre == "Non employé":
    df_filtre = df_filtre[df_filtre["EMPLOYE"] == 0]

df_filtre_display = df_filtre.copy()
df_filtre_display["EMPLOYE"] = df_filtre_display["EMPLOYE"].map({0: "Non employé", 1: "Employé"})

st.dataframe(
    df_filtre_display.rename(columns={
        "ID_UTILISATEUR": "ID Utilisateur",
        "ID_PROFIL": "ID Technique",
        "PROFIL_NOM": "Profil",
        "EMPLOYE": "Statut"
    }),
    use_container_width=True,
    hide_index=True
)

# ==================================================
# SYNTHÈSE ANALYTIQUE
# ==================================================
st.markdown("## Lecture automatique & recommandations")

message = (
    f"Le système compte **{nb_users} utilisateurs** répartis sur "
    f"**{nb_profils} profils distincts**. "
)

if part_top_profil > 60:
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