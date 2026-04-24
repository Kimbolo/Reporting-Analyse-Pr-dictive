import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

profil_counts = df["ID_PROFIL"].value_counts()

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    x=profil_counts.index.astype(str),
    y=profil_counts.values,
    palette="Blues",
    ax=ax
)
ax.set_title("Nombre d’utilisateurs par profil", fontsize=13, weight="bold")
ax.set_xlabel("Profil")
ax.set_ylabel("Nombre d’utilisateurs")
ax.grid(axis="y", linestyle="--", alpha=0.6)

st.pyplot(fig)

st.info(
    "Cette répartition permet d’identifier les profils dominants "
    "et ceux potentiellement sous‑représentés."
)

# ==================================================
# EMPLOYÉS VS NON‑EMPLOYÉS
# ==================================================
st.markdown("## Employés vs non‑employés")

emp_counts = df["EMPLOYE"].value_counts().rename(
    {0: "Non employé", 1: "Employé"}
)

fig, ax = plt.subplots(figsize=(6, 4))
ax.pie(
    emp_counts.values,
    labels=emp_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#ff9999", "#66b3ff"]
)
ax.set_title("Statut des utilisateurs")

st.pyplot(fig)

st.info(
    "Un fort déséquilibre entre employés et non‑employés peut poser "
    "des questions de sécurité ou de gouvernance."
)

# ==================================================
# ANALYSE DE CONCENTRATION (RISQUE)
# ==================================================
st.markdown("## Analyse de concentration des rôles")

part_top_profil = profil_counts.iloc[0] / nb_users * 100

st.metric(
    "Part du profil dominant",
    f"{part_top_profil:.1f} %"
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
profils_dispo = sorted(df["ID_PROFIL"].unique())
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

df_filtre = df[df["ID_PROFIL"].isin(profil_filtre)]

if statut_filtre == "Employé":
    df_filtre = df_filtre[df_filtre["EMPLOYE"] == 1]
elif statut_filtre == "Non employé":
    df_filtre = df_filtre[df_filtre["EMPLOYE"] == 0]

st.dataframe(df_filtre, use_container_width=True)

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