import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================================
# TITRE
# ==========================================================
st.title("Machine Learning – Anticipation & décisions")

# ==========================================================
# SÉCURITÉ DES DONNÉES PARTAGÉES
# ==========================================================
required_keys = ["facture_f", "stock"]
if not all(k in st.session_state for k in required_keys):
    st.warning("Veuillez d’abord charger les données dans l’onglet App.")
    st.stop()

facture = st.session_state["facture_f"].copy()
stock = st.session_state["stock"].copy()

facture["DATE_CREATION"] = pd.to_datetime(
    facture["DATE_CREATION"], errors="coerce"
)

# ==========================================================
# CHOIX DU CAS ML
# ==========================================================
cas_ml = st.radio(
    "Cas d’usage Machine Learning",
    [
        "Prévision des ventes",
        "Risque de rupture de stock",
        "Détection d’anomalies (pertes)",
        "ML Utilisateurs (admin)"
    ],
    horizontal=True
)

# ==========================================================
# 🔮 1. PRÉVISION DES VENTES
# ==========================================================
if cas_ml == "Prévision des ventes":
    st.subheader("Prévision des ventes")

    facture["MOIS"] = facture["DATE_CREATION"].dt.month
    facture["ANNEE"] = facture["DATE_CREATION"].dt.year

    ca_mensuel = (
        facture
        .groupby(["ANNEE", "MOIS"])["MONTANT_NET"]
        .sum()
        .reset_index()
    )

    st.dataframe(ca_mensuel.tail(12))

    X = ca_mensuel[["MOIS"]]
    y = ca_mensuel["MONTANT_NET"]

    model = LinearRegression()
    model.fit(X, y)

    mois_futur = pd.DataFrame({"MOIS": range(1, 13)})
    ca_prevu = model.predict(mois_futur)

    fig, ax = plt.subplots()
    ax.plot(ca_mensuel["MOIS"], ca_mensuel["MONTANT_NET"], label="Historique")
    ax.plot(mois_futur["MOIS"], ca_prevu, linestyle="--", label="Prévision")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Chiffre d’affaires")
    ax.set_title("Prévision du chiffre d’affaires")
    ax.legend()
    st.pyplot(fig)

    st.info(
        "Cette prévision permet d’anticiper les ventes futures afin "
        "d’adapter la production, le stock et la trésorerie."
    )

# ==========================================================
# 📦 2. RISQUE DE RUPTURE DE STOCK
# ==========================================================
elif cas_ml == "Risque de rupture de stock":
    st.subheader("Risque de rupture de stock")

    seuil = stock["QUANTITE"].quantile(0.2)
    stock["RISQUE_RUPTURE"] = stock["QUANTITE"] < seuil

    colonnes = ["QUANTITE", "RISQUE_RUPTURE"]
    if "ID_PRODUIT" in stock.columns:
        colonnes.insert(0, "ID_PRODUIT")

    st.dataframe(stock[colonnes])

    st.warning(
        "Les produits marqués comme à risque doivent être "
        "réapprovisionnés ou planifiés en production."
    )

# ==========================================================
# ⚠️ 3. DÉTECTION D’ANOMALIES (PERTES)
# ==========================================================
elif cas_ml == "Détection d’anomalies (pertes)":
    st.subheader("Détection d’anomalies – pertes & avaries")

    perte_cols = [
        c for c in facture.columns
        if "PERD" in c.upper() or "AVARIE" in c.upper()
    ]

    if not perte_cols:
        st.info("Aucune donnée de pertes disponible pour l’analyse.")
    else:
        pertes = facture[perte_cols].fillna(0)

        model = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        facture["ANOMALIE"] = model.fit_predict(pertes)

        anomalies = facture[facture["ANOMALIE"] == -1]

        st.dataframe(anomalies[perte_cols])

        st.warning(
            f"{len(anomalies)} lignes anormales détectées. "
            "Elles doivent être analysées en priorité."
        )

# ==========================================================
# 👥 4. ML UTILISATEURS (ADMIN)
# ==========================================================
elif cas_ml == "ML Utilisateurs (admin)":
    st.subheader("Classification utilisateurs")

    from db import get_data

    df = get_data("SELECT * FROM utilisateur")
    df = df[["ID_PROFIL", "EMPLOYE"]].dropna()

    X = df[["ID_PROFIL"]]
    y = df["EMPLOYE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.metric("Accuracy du modèle", f"{acc:.2f}")

    df["prediction_employe"] = model.predict(X)
    st.dataframe(df)

    st.info(
        "Ce modèle est un cas administratif illustratif, "
        "sans impact direct sur les décisions business."
    )

# ==========================================================
# 🔔 ALERTES AUTOMATIQUES
# ==========================================================
st.divider()
st.title("Alertes & recommandations automatiques")

alertes = []

facture["MOIS"] = facture["DATE_CREATION"].dt.month
ca_mensuel = facture.groupby("MOIS")["MONTANT_NET"].sum()

if len(ca_mensuel) >= 3:
    if ca_mensuel.iloc[-1] < ca_mensuel.mean() * 0.8:
        alertes.append("🔮 Alerte ventes : baisse significative détectée.")

seuil_stock = stock["QUANTITE"].quantile(0.15)
if (stock["QUANTITE"] < seuil_stock).any():
    alertes.append("Alerte stock : risque de rupture détecté.")

perte_cols = [
    c for c in facture.columns
    if "PERD" in c.upper() or "AVARIE" in c.upper()
]
if perte_cols and facture[perte_cols].sum().sum() > 0:
    alertes.append("Alerte pertes : pertes ou avaries détectées.")

if alertes:
    for alerte in alertes:
        st.error(alerte)
else:
    st.success("Aucune alerte critique détectée.")