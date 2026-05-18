import streamlit as st
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from email.message import EmailMessage
import smtplib

# ==================================================
# CONFIG GLOBAL
# ==================================================
REPORT_DIR = "reports"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "..", "assets", "logo.jpg")

os.makedirs(REPORT_DIR, exist_ok=True)

# ==================================================
# TITRE APP
# ==================================================
st.title("Exports & Rapports exécutifs")
st.caption("Génération automatique de rapports PDF")

# ==================================================
# SÉCURITÉ SESSION
# ==================================================
required = ["facture_f", "paiement_facture", "annee"]
missing = [k for k in required if k not in st.session_state]

if missing:
    st.warning("Veuillez d’abord configurer les filtres dans l’onglet App.")
    st.stop()

facture = st.session_state["facture_f"].copy()
paiement = st.session_state["paiement_facture"].copy()
annee = st.session_state["annee"]

# ==================================================
# KPIs FINANCE
# ==================================================
ca_total = facture["MONTANT_NET"].sum()
enc_total = paiement["MONTANT"].sum()
taux_enc = (enc_total / ca_total * 100) if ca_total > 0 else 0

# ==================================================
# OUTILS GRAPHIQUES
# ==================================================
def generate_dashboard_graphes():
    paths = {}

    # Graphe 1 — CA mensuel
    facture["MOIS"] = facture["DATE_CREATION"].dt.month
    ca = facture.groupby("MOIS")["MONTANT_NET"].sum()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ca.index, ca.values, marker="o")
    ax.set_title("Chiffre d’affaires mensuel")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    path1 = f"{REPORT_DIR}/ca_mensuel.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    paths["ca_mensuel"] = path1

    # Graphe 2 — Encaissements mensuels
    paiement["MOIS"] = paiement["DATE_PAIEMENT"].dt.month
    enc = paiement.groupby("MOIS")["MONTANT"].sum()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(enc.index, enc.values, marker="o", color="green")
    ax.set_title("Encaissements mensuels")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    path2 = f"{REPORT_DIR}/encaissements_mensuels.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    paths["encaissements_mensuels"] = path2

    # Graphe 3 — Taux d’encaissement cumulé
    facture["MOIS"] = facture["DATE_CREATION"].dt.month
    paiement["MOIS"] = paiement["DATE_PAIEMENT"].dt.month

    ca_m = facture.groupby("MOIS")["MONTANT_NET"].sum().sort_index()
    enc_m = paiement.groupby("MOIS")["MONTANT"].sum().sort_index()

    df_cumul = pd.DataFrame({
        "CA": ca_m,
        "Encaissements": enc_m
    }).fillna(0)

    df_cumul["Taux_encaissement"] = (
        df_cumul["Encaissements"].cumsum()
        / df_cumul["CA"].cumsum()
        * 100
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(
        df_cumul.index,
        df_cumul["Taux_encaissement"],
        marker="o",
        color="purple"
    )
    ax.set_title("Taux d’encaissement cumulé (%)")
    ax.set_ylabel("%")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    path3 = f"{REPORT_DIR}/taux_encaissement_cumule.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)

    paths["taux_encaissement_cumule"] = path3

    return paths

def generate_analytics_graphes():
    paths = {}

    enc = paiement.groupby(paiement["DATE_PAIEMENT"].dt.month)["MONTANT"].sum()
    ca = facture.groupby(facture["DATE_CREATION"].dt.month)["MONTANT_NET"].sum()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ca.index, ca.values, label="CA")
    ax.plot(enc.index, enc.values, label="Encaissements")
    ax.legend()
    ax.set_title("CA vs Encaissements")
    ax.grid(True)

    path = f"{REPORT_DIR}/ca_vs_enc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["ca_vs_enc"] = path

    return paths

# ==================================================
# GÉNÉRATION PDF
# ==================================================
def generate_pdf(sections):
    file_name = f"{REPORT_DIR}/rapport_direction_{annee}.pdf"
    c = canvas.Canvas(file_name, pagesize=A4)
    width, height = A4
    y = height - 60

    # ===============================
    # EN-TÊTE OFFICIEL
    # ===============================

    if os.path.exists(LOGO_PATH):
        c.drawImage(
            LOGO_PATH,
            40,                 # marge gauche
            height - 120,       # bandeau haut
            width=140,
            height=70,
            preserveAspectRatio=True
        )

    # --- RAISON SOCIALE À DROITE
    x_right = width - 40          # marge droite
    y_header = height - 60        # point de départ vertical

    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(x_right, y_header, "N'NAM AGRO INDUSTRIE")

    c.setFont("Helvetica", 9)
    c.drawRightString(x_right, y_header - 14, "SIEGE SOCIAL : 2ème Etage Immeuble SAAR – Hippodrome")
    c.drawRightString(x_right, y_header - 26, "BP : 7073 Yaoundé, Cameroun")
    c.drawRightString(x_right, y_header - 38, "Tél : (237) 650 912 171 / 697 097 671")
    c.drawRightString(x_right, y_header - 50, "CC : M052217409491J | RC : RC/YAO/2024/M/459")
    c.drawRightString(x_right, y_header - 62, f"Fait à Yaoundé, le {datetime.now().strftime('%d %B %Y')}")

    # --- LIGNE DE SÉPARATION
    c.line(40, height - 135, width - 40, height - 135)
    y = height - 170


    c.setFont("Helvetica-Bold", 20)
    c.drawString(60, y, "RAPPORT DE PERFORMANCE")
    y -= 35

    c.setFont("Helvetica", 12)
    c.drawString(60, y, f"Période analysée : {annee}")
    y -= 30

    # ==================================================
    # SECTION APP
    # ==================================================
    if "App" in sections:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(60, y, "CONTEXTE GÉNÉRAL")
        y -= 20

        c.setFont("Helvetica", 11)
        c.drawString(60, y, "• Données filtrées via l’application (année, mois, validation).")
        y -= 14
        c.drawString(60, y, "• Le rapport reflète exactement les données visibles à l’écran.")
        y -= 30

    # ==================================================
    # SECTION DASHBOARD
    # ==================================================
    if "Dashboard" in sections:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(60, y, "PERFORMANCE GLOBALE")
        y -= 20

        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"• Chiffre d’affaires : {ca_total:,.0f} FCFA")
        y -= 14
        c.drawString(60, y, f"• Encaissements : {enc_total:,.0f} FCFA")
        y -= 14
        c.drawString(60, y, f"• Taux d’encaissement : {taux_enc:.1f} %")

    # CRÉATION DES GRAPHES
        graphes = generate_dashboard_graphes()

        
        graph_width = 240
        graph_height = 140

        left_x = 60
        right_x = left_x + graph_width + 25

        top_row_y = 360
        bottom_row_y = 190

    # ── LIGNE 1 : GRAPHE A / GRAPHE B ──
    if "ca_mensuel" in graphes:
        c.drawImage(
            graphes["ca_mensuel"],
            left_x,
            top_row_y,
            width=graph_width,
            height=graph_height,
            preserveAspectRatio=True
        )

    if "encaissements_mensuels" in graphes:
        c.drawImage(
            graphes["encaissements_mensuels"],
            right_x,
            top_row_y,
            width=graph_width,
            height=graph_height,
            preserveAspectRatio=True
        )

    # ── LIGNE 2 : GRAPHE C / TEXTE ──
    if "taux_encaissement_cumule" in graphes:
        c.drawImage(
            graphes["taux_encaissement_cumule"],
            left_x,
            bottom_row_y,
            width=graph_width,
            height=graph_height,
            preserveAspectRatio=True
        )

    # ── TEXTE EXÉCUTIF ──
    text_x = right_x
    text_y = bottom_row_y + graph_height - 10

    c.setFont("Helvetica-Bold", 12)
    c.drawString(text_x, text_y, "Lecture exécutive")

    c.setFont("Helvetica", 10)
    if taux_enc < 80:
        c.drawString(text_x, text_y - 20, "• Le taux d’encaissement est insuffisant.")
        c.drawString(text_x, text_y - 35, "• Risque sur la trésorerie à court terme.")
        c.drawString(text_x, text_y - 50, "• Action recommandée : renforcer le recouvrement.")
    else:
        c.drawString(text_x, text_y - 20, "• Performance de trésorerie satisfaisante.")
        c.drawString(text_x, text_y - 35, "• Bonne conversion des ventes en liquidités.")


    # ==================================================
    # SECTION ANALYTICS
    # ==================================================
    if "Analytics" in sections:

        graphes = generate_analytics_graphes()
        c.drawImage(graphes["ca_vs_enc"], 60, 300, width=470, height=200)

        c.showPage()

    # ==================================================
    # SECTION UTILISATEURS
    # ==================================================

    if "Utilisateurs" in sections:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(60, y, "STRUCTURE & GOUVERNANCE")
        y -= 20

        c.setFont("Helvetica", 11)
        c.drawString(60, y, "• L’organisation et les profils utilisateurs influencent la qualité des opérations.")
        y -= 30

    # ==================================================
    # SECTION ML
    # ==================================================
    if "ML" in sections:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(60, y, "PRÉVISIONS & PERSPECTIVES")
        y -= 20

        c.setFont("Helvetica", 11)
        c.drawString(60, y, "• Les tendances actuelles permettent d’anticiper les risques futurs.")
        y -= 30

    # ===============================
    # FOOTER
    # ===============================
    c.setFont("Helvetica-Oblique", 8)
    c.drawRightString(width - 40, 30, "Usage interne – Document confidentiel")

    c.save()
    return file_name

# ==================================================
# UI STREAMLIT
# ==================================================
st.markdown("## Contenu du rapport")

sections_disponibles = ["App", "Dashboard", "Analytics", "Utilisateurs", "ML"]
sections_selectionnees = st.multiselect(
    "Sections à inclure dans le rapport",
    sections_disponibles,
    default=sections_disponibles
)

if st.button("Générer le rapport PDF"):
    pdf = generate_pdf(sections_selectionnees)
    st.success("Rapport généré avec succès")

    with open(pdf, "rb") as f:
        st.download_button(
            "Télécharger le PDF",
            f,
            file_name=os.path.basename(pdf),
            mime="application/pdf"
        )
# ===============================
# ENVOI AUTOMATIQUE PAR EMAIL
# ===============================
st.markdown("## Envoi automatique")

email = st.text_input(
    "Adresse email du destinataire",
    placeholder="ex: direction@entreprise.com"
)

if st.button("Envoyer le rapport par email"):
    if email:
        pdf = generate_pdf(sections_selectionnees)
        send_email_with_pdf(pdf, email)
        st.success("Rapport envoyé par email avec succès")
    else:
        st.warning("Veuillez saisir une adresse email valide.")