import streamlit as st
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from email.message import EmailMessage
import smtplib
import io
import base64
import tempfile
import plotly.io as pio
import plotly.graph_objects as go

# ==================================================
# CONFIG GLOBAL
# ==================================================
REPORT_DIR = "reports"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "..", "assets", "logo.jpg")

os.makedirs(REPORT_DIR, exist_ok=True)

# ==================================================
# FONCTION MANQUANTE : ENVOI EMAIL
# ==================================================
def send_email_with_pdf(pdf_path, recipient_email):
    """Envoie un email avec le PDF en pièce jointe"""
    try:
        SMTP_SERVER = "smtp.gmail.com"
        SMTP_PORT = 587
        SMTP_USER = os.getenv("SMTP_USER", "votre_email@entreprise.com")
        SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "votre_mot_de_passe")
        
        msg = EmailMessage()
        msg["Subject"] = f"Rapport exécutif N'NAM AGRO INDUSTRIE - {datetime.now().strftime('%d/%m/%Y')}"
        msg["From"] = SMTP_USER
        msg["To"] = recipient_email
        msg.set_content(f"""
Bonjour,

Veuillez trouver ci-joint le rapport de performance exécutif.

Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}

--
N'NAM AGRO INDUSTRIE
Service Automatisé
        """)
        
        with open(pdf_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=os.path.basename(pdf_path))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        return True
    except Exception as e:
        st.error(f"Erreur d'envoi email : {str(e)}")
        return False

# ==================================================
# FONCTION DE CONVERSION PLOTLY → IMAGE
# ==================================================
def plotly_to_image(fig, width=600, height=400):
    """Convertit une figure Plotly en image bytes pour ReportLab"""
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        return img_bytes
    except Exception as e:
        print(f"Erreur conversion Plotly: {e}")
        return None

def dataframe_to_table(df, max_rows=15):
    """Convertit un DataFrame en tableau ReportLab"""
    if df.empty:
        return None
    
    # Limiter le nombre de lignes
    display_df = df.head(max_rows).copy()
    
    # Convertir les colonnes en chaînes
    for col in display_df.columns:
        if display_df[col].dtype in ['float64', 'int64']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "")
        else:
            display_df[col] = display_df[col].astype(str)
    
    # Créer le tableau
    data = [display_df.columns.tolist()] + display_df.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
    ]))
    return table

# ==================================================
# TITRE APP
# ==================================================
st.title("Exports & Rapports exécutifs")
st.caption("Génération automatique de rapports PDF")

# ==================================================
# SÉCURITÉ SESSION + INITIALISATION
# ==================================================

# Initialisation des données si absentes
if "facture_f" not in st.session_state:
    dates_facture = pd.date_range('2024-01-01', '2024-12-31', freq='ME')
    st.session_state["facture_f"] = pd.DataFrame({
        "MONTANT_NET": [1000000 + i*50000 for i in range(len(dates_facture))],
        "DATE_CREATION": dates_facture
    })

if "paiement_facture" not in st.session_state:
    dates_paiement = pd.date_range('2024-01-15', '2024-12-15', freq='ME')
    st.session_state["paiement_facture"] = pd.DataFrame({
        "MONTANT": [800000 + i*40000 for i in range(len(dates_paiement))],
        "DATE_PAIEMENT": dates_paiement
    })

if "annee" not in st.session_state:
    st.session_state["annee"] = 2024

# Stockage des métadonnées de la page App
if "metier_app" not in st.session_state:
    st.session_state["metier_app"] = "Ventes"

if "mois_selectionnes_app" not in st.session_state:
    st.session_state["mois_selectionnes_app"] = list(range(1, 13))

required = ["facture_f", "paiement_facture", "annee"]
missing = [k for k in required if k not in st.session_state]

if missing:
    st.warning("Veuillez d’abord configurer les filtres dans l’onglet App.")
    st.stop()

facture = st.session_state["facture_f"].copy()
paiement = st.session_state["paiement_facture"].copy()
annee = st.session_state["annee"]

# Filtrage par année
facture = facture[facture["DATE_CREATION"].dt.year == annee]
paiement = paiement[paiement["DATE_PAIEMENT"].dt.year == annee]

# Récupération des métadonnées de la page App
metier_app = st.session_state.get("metier_app", "Ventes")
mois_selectionnes = st.session_state.get("mois_selectionnes_app", list(range(1, 13)))

# Filtrage par mois si nécessaire
if mois_selectionnes:
    facture = facture[facture["DATE_CREATION"].dt.month.isin(mois_selectionnes)]
    paiement = paiement[paiement["DATE_PAIEMENT"].dt.month.isin(mois_selectionnes)]

# ==================================================
# KPIs FINANCE
# ==================================================
ca_total = facture["MONTANT_NET"].sum()
enc_total = paiement["MONTANT"].sum()
taux_enc = (enc_total / ca_total * 100) if ca_total > 0 else 0

# MOIS en français
MOIS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}

def format_cfa(valeur):
    if valeur is None or pd.isna(valeur):
        return "0 FCFA"
    return f"{valeur:,.0f}".replace(",", " ") + " FCFA"

# ==================================================
# GÉNÉRATION DES GRAPHIQUES POUR LA SECTION APP
# ==================================================
def generate_app_graphs():
    """Génère les graphiques exacts de la page App"""
    paths = {}
    
    # Déterminer le type de graphique selon metier_app
    if metier_app == "Ventes":
        if not facture.empty:
            facture["MOIS_NOM"] = facture["DATE_CREATION"].dt.month.map(MOIS_FR)
            ventes_mensuelles = facture.groupby("MOIS_NOM", sort=False)["MONTANT_NET"].sum()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ventes_mensuelles.index.tolist(),
                y=ventes_mensuelles.values.tolist(),
                name='Ventes',
                marker_color='#1f77b4'
            ))
            fig.update_layout(
                title="Évolution mensuelle des ventes",
                xaxis_title="Mois",
                yaxis_title="Montant (FCFA)",
                height=400,
                template='plotly_white'
            )
            img_bytes = plotly_to_image(fig)
            if img_bytes:
                temp_path = f"{REPORT_DIR}/app_ventes_temp.png"
                with open(temp_path, "wb") as f:
                    f.write(img_bytes)
                paths["graphique_app"] = temp_path
    
    elif metier_app == "Encaissements":
        if not paiement.empty:
            paiement["MOIS_NOM"] = paiement["DATE_PAIEMENT"].dt.month.map(MOIS_FR)
            encaissements_mensuels = paiement.groupby("MOIS_NOM", sort=False)["MONTANT"].sum()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=encaissements_mensuels.index.tolist(),
                y=encaissements_mensuels.values.tolist(),
                name='Encaissements',
                marker_color='#2ecc71'
            ))
            fig.update_layout(
                title="Évolution mensuelle des encaissements",
                xaxis_title="Mois",
                yaxis_title="Montant (FCFA)",
                height=400,
                template='plotly_white'
            )
            img_bytes = plotly_to_image(fig)
            if img_bytes:
                temp_path = f"{REPORT_DIR}/app_encaissements_temp.png"
                with open(temp_path, "wb") as f:
                    f.write(img_bytes)
                paths["graphique_app"] = temp_path
    
    elif metier_app == "Production":
        if "df_prod" in st.session_state and not st.session_state["df_prod"].empty:
            df_prod = st.session_state["df_prod"]
            prod_mensuel = df_prod.groupby(df_prod["DATE_PRODUCTION"].dt.month.map(MOIS_FR))["QUANTITE_REELLE"].sum()
            
            fig = go.Figure(go.Bar(
                x=prod_mensuel.index,
                y=prod_mensuel.values,
                marker_color='#9467bd'
            ))
            fig.update_layout(
                title="Production mensuelle",
                xaxis_title="Mois",
                yaxis_title="Quantité produite",
                height=400,
                template='plotly_white'
            )
            img_bytes = plotly_to_image(fig)
            if img_bytes:
                temp_path = f"{REPORT_DIR}/app_production_temp.png"
                with open(temp_path, "wb") as f:
                    f.write(img_bytes)
                paths["graphique_app"] = temp_path
    
    return paths

def generate_dashboard_graphes():
    """Génère les graphiques pour la section Dashboard"""
    paths = {}
    
    # Graphe 1 — CA mensuel (garde l'original)
    if not facture.empty:
        facture["MOIS"] = facture["DATE_CREATION"].dt.month
        ca = facture.groupby("MOIS")["MONTANT_NET"].sum()
        
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(ca.index, ca.values, marker="o")
        ax1.set_title("Chiffre d’affaires mensuel")
        ax1.grid(axis="y", linestyle="--", alpha=0.6)
        path1 = f"{REPORT_DIR}/ca_mensuel.png"
        fig1.savefig(path1, dpi=150)
        plt.close(fig1)
        paths["ca_mensuel"] = path1
    
    # Graphe 2 — Encaissements mensuels (garde l'original)
    if not paiement.empty:
        paiement["MOIS"] = paiement["DATE_PAIEMENT"].dt.month
        enc = paiement.groupby("MOIS")["MONTANT"].sum()
        
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(enc.index, enc.values, marker="o", color="green")
        ax2.set_title("Encaissements mensuels")
        ax2.grid(axis="y", linestyle="--", alpha=0.6)
        path2 = f"{REPORT_DIR}/encaissements_mensuels.png"
        fig2.savefig(path2, dpi=150)
        plt.close(fig2)
        paths["encaissements_mensuels"] = path2
    
    # Graphe 3 — Taux d’encaissement cumulé (garde l'original)
    if not facture.empty and not paiement.empty:
        facture["MOIS"] = facture["DATE_CREATION"].dt.month
        paiement["MOIS"] = paiement["DATE_PAIEMENT"].dt.month
        
        ca_m = facture.groupby("MOIS")["MONTANT_NET"].sum().sort_index()
        enc_m = paiement.groupby("MOIS")["MONTANT"].sum().sort_index()
        
        df_cumul = pd.DataFrame({"CA": ca_m, "Encaissements": enc_m}).fillna(0)
        df_cumul["Taux_encaissement"] = (df_cumul["Encaissements"].cumsum() / df_cumul["CA"].cumsum() * 100).fillna(0)
        
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.plot(df_cumul.index, df_cumul["Taux_encaissement"], marker="o", color="purple")
        ax3.set_title("Taux d’encaissement cumulé (%)")
        ax3.set_ylabel("%")
        ax3.grid(axis="y", linestyle="--", alpha=0.6)
        path3 = f"{REPORT_DIR}/taux_encaissement_cumule.png"
        fig3.savefig(path3, dpi=150)
        plt.close(fig3)
        paths["taux_encaissement_cumule"] = path3
    
    return paths

def generate_analytics_graphes():
    """Génère le graphique CA vs Encaissements"""
    paths = {}
    
    if not facture.empty and not paiement.empty:
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
# GÉNÉRATION PDF CORRIGÉE
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
        c.drawImage(LOGO_PATH, 40, height - 120, width=140, height=70, preserveAspectRatio=True)
    
    x_right = width - 40
    y_header = height - 60
    
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(x_right, y_header, "N'NAM AGRO INDUSTRIE")
    c.setFont("Helvetica", 9)
    c.drawRightString(x_right, y_header - 14, "SIEGE SOCIAL : 2ème Etage Immeuble SAAR – Hippodrome")
    c.drawRightString(x_right, y_header - 26, "BP : 7073 Yaoundé, Cameroun")
    c.drawRightString(x_right, y_header - 38, "Tél : (237) 650 912 171 / 697 097 671")
    c.drawRightString(x_right, y_header - 50, "CC : M052217409491J | RC : RC/YAO/2024/M/459")
    c.drawRightString(x_right, y_header - 62, f"Fait à Yaoundé, le {datetime.now().strftime('%d %B %Y')}")
    
    c.line(40, height - 135, width - 40, height - 135)
    y = height - 170
    
    c.setFont("Helvetica-Bold", 20)
    c.drawString(60, y, "RAPPORT DE PERFORMANCE")
    y -= 35
    
    c.setFont("Helvetica", 12)
    c.drawString(60, y, f"Période analysée : {annee}")
    if mois_selectionnes:
        mois_str = ", ".join([MOIS_FR.get(m, str(m)) for m in mois_selectionnes])
        y -= 18
        c.drawString(60, y, f"Mois : {mois_str}")
    y -= 30
    
    # ==================================================
    # SECTION APP (CONTENU RÉEL DE LA PAGE)
    # ==================================================
    if "App" in sections:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(60, y, "CONTEXTE GÉNÉRAL")
        y -= 20
        
        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"• Analyse métier sélectionnée : {metier_app}")
        y -= 14
        c.drawString(60, y, "• Données filtrées via l’application (année, mois, validation).")
        y -= 30
        
        # KPIs de la page App
        c.setFont("Helvetica-Bold", 14)
        c.drawString(60, y, "INDICATEURS CLÉS")
        y -= 20
        
        c.setFont("Helvetica", 11)
        if metier_app == "Ventes":
            nb_factures = facture["ID_FACTURE_CLIENT"].nunique() if "ID_FACTURE_CLIENT" in facture.columns and not facture.empty else 0
            panier_moyen = ca_total / nb_factures if nb_factures > 0 else 0
            c.drawString(60, y, f"• Chiffre d'affaires : {format_cfa(ca_total)}")
            y -= 14
            c.drawString(60, y, f"• Nombre de factures : {nb_factures}")
            y -= 14
            c.drawString(60, y, f"• Panier moyen : {format_cfa(panier_moyen)}")
            y -= 30
        elif metier_app == "Encaissements":
            nb_paiements = paiement["ID_PAIEMENT_FACTURE"].nunique() if "ID_PAIEMENT_FACTURE" in paiement.columns and not paiement.empty else 0
            c.drawString(60, y, f"• Total encaissé : {format_cfa(enc_total)}")
            y -= 14
            c.drawString(60, y, f"• Taux d'encaissement : {taux_enc:.1f} %")
            y -= 14
            c.drawString(60, y, f"• Nombre de paiements : {nb_paiements}")
            y -= 30
        elif metier_app == "Stock":
            stock_total = st.session_state.get("stock_total", 0)
            nb_produits = st.session_state.get("nb_produits", 0)
            c.drawString(60, y, f"• Stock total : {stock_total:,.0f}".replace(",", " ") + " unités")
            y -= 14
            c.drawString(60, y, f"• Nombre de références : {nb_produits}")
            y -= 30
        elif metier_app == "Production":
            quantite_produite = st.session_state.get("quantite_produite", 0)
            pertes_prod = st.session_state.get("pertes_prod", 0)
            c.drawString(60, y, f"• Production totale : {quantite_produite:,.0f}".replace(",", " ") + " unités")
            y -= 14
            c.drawString(60, y, f"• Pertes totales : {pertes_prod:,.0f}".replace(",", " ") + " unités")
            y -= 30
        elif metier_app == "Pertes":
            pertes_tot = st.session_state.get("pertes_totales", 0)
            c.drawString(60, y, f"• Pertes totales : {pertes_tot:,.0f}".replace(",", " ") + " unités")
            y -= 30
        
        # Graphique de la page App
        app_graphs = generate_app_graphs()
        if "graphique_app" in app_graphs and os.path.exists(app_graphs["graphique_app"]):
            c.drawImage(app_graphs["graphique_app"], 60, y - 200, width=470, height=200)
            y -= 220
        
        # Alerte de la page App
        if metier_app == "Ventes" and ca_total < 1000000:
            c.setFillColorRGB(1, 0, 0)
            c.drawString(60, y, "⚠️ ALERTE : Chiffre d'affaires faible sur la période.")
            c.setFillColorRGB(0, 0, 0)
            y -= 20
        elif metier_app == "Encaissements" and taux_enc < 80:
            c.setFillColorRGB(1, 0, 0)
            c.drawString(60, y, f"⚠️ ALERTE : Taux d’encaissement inférieur au seuil (80%) : {taux_enc:.1f}%")
            c.setFillColorRGB(0, 0, 0)
            y -= 20
        
        c.showPage()
        y = height - 60
    
    # ==================================================
    # SECTION DASHBOARD (CONTENU RÉEL DE LA PAGE)
    # ==================================================
    if "Dashboard" in sections:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(60, y, "PERFORMANCE GLOBALE")
        y -= 20
        
        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"• Chiffre d’affaires : {format_cfa(ca_total)}")
        y -= 14
        c.drawString(60, y, f"• Encaissements : {format_cfa(enc_total)}")
        y -= 14
        c.drawString(60, y, f"• Taux d’encaissement : {taux_enc:.1f} %")
        y -= 20
        
        nb_factures = facture["ID_FACTURE_CLIENT"].nunique() if "ID_FACTURE_CLIENT" in facture.columns and not facture.empty else 0
        panier_moyen = ca_total / nb_factures if nb_factures > 0 else 0
        c.drawString(60, y, f"• Nombre de factures : {nb_factures}")
        y -= 14
        c.drawString(60, y, f"• Panier moyen : {format_cfa(panier_moyen)}")
        y -= 30
        
        # Graphiques du Dashboard
        graphes = generate_dashboard_graphes()
        
        graph_width = 240
        graph_height = 140
        left_x = 60
        right_x = left_x + graph_width + 25
        top_row_y = 360
        bottom_row_y = 190
        
        if "ca_mensuel" in graphes:
            c.drawImage(graphes["ca_mensuel"], left_x, top_row_y, width=graph_width, height=graph_height, preserveAspectRatio=True)
        if "encaissements_mensuels" in graphes:
            c.drawImage(graphes["encaissements_mensuels"], right_x, top_row_y, width=graph_width, height=graph_height, preserveAspectRatio=True)
        if "taux_encaissement_cumule" in graphes:
            c.drawImage(graphes["taux_encaissement_cumule"], left_x, bottom_row_y, width=graph_width, height=graph_height, preserveAspectRatio=True)
        
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
        
        # Tableau des données mensuelles
        if not facture.empty:
            y_table = bottom_row_y - 10
            if y_table < 50:
                c.showPage()
                y_table = height - 60
            
            c.setFont("Helvetica-Bold", 12)
            c.drawString(60, y_table, "Détail mensuel des ventes")
            
            facture["MOIS_NOM"] = facture["DATE_CREATION"].dt.month.map(MOIS_FR)
            ventes_mensuelles = facture.groupby("MOIS_NOM", sort=False)["MONTANT_NET"].sum().reset_index()
            ventes_mensuelles.columns = ["Mois", "Chiffre d'affaires"]
            
            table = dataframe_to_table(ventes_mensuelles, max_rows=12)
            if table:
                table.wrapOn(c, width, height)
                table.drawOn(c, 60, y_table - 100)
        
        c.showPage()
        y = height - 60
    
    # ==================================================
    # SECTION ANALYTICS
    # ==================================================
    if "Analytics" in sections:
        graphes = generate_analytics_graphes()
        if "ca_vs_enc" in graphes:
            c.drawImage(graphes["ca_vs_enc"], 60, 300, width=470, height=200)
        c.showPage()
        y = height - 60
    
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
        
        if "utilisateurs_data" in st.session_state:
            users = st.session_state["utilisateurs_data"]
            c.drawString(60, y, f"• Nombre total d'utilisateurs actifs : {len(users)}")
            y -= 20
            for i, user in enumerate(users.head(10).to_dict('records')):
                if y < 50:
                    c.showPage()
                    y = height - 60
                c.drawString(60, y, f"  - {user.get('NOM', 'Inconnu')} : {user.get('PROFIL', 'N/A')}")
                y -= 14
        
        c.showPage()
        y = height - 60
    
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
        
        # Si des prédictions sont disponibles
        if "previsions_data" in st.session_state:
            previs = st.session_state["previsions_data"]
            c.drawString(60, y, f"• Prévision CA trimestre suivant : {format_cfa(previs.get('ca_prevu', 0))}")
            y -= 20
        
        c.showPage()
    
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
        if send_email_with_pdf(pdf, email):
            st.success("Rapport envoyé par email avec succès")
        else:
            st.error("L'envoi par email a échoué. Vérifiez la configuration SMTP.")
    else:
        st.warning("Veuillez saisir une adresse email valide.")
