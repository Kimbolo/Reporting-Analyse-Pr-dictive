import streamlit as st
import os
from datetime import datetime
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from email.message import EmailMessage
import smtplib
from db import get_data

# ==================================================
# CONFIG GLOBAL
# ==================================================
REPORT_DIR = "reports"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "..", "assets", "logo.jpg")

os.makedirs(REPORT_DIR, exist_ok=True)

# ==================================================
# CONSTANTES
# ==================================================
MOIS_FR = {1:"Janvier", 2:"Février", 3:"Mars", 4:"Avril", 5:"Mai", 6:"Juin",
           7:"Juillet", 8:"Août", 9:"Septembre", 10:"Octobre", 11:"Novembre", 12:"Décembre"}
MOIS_ABBR = {1:"Jan", 2:"Fév", 3:"Mar", 4:"Avr", 5:"Mai", 6:"Juin",
             7:"Juil", 8:"Aoû", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Déc"}

PAGES_DISPONIBLES = [
    "Accueil",
    "Pôle Commercial",
    "Pôle Production",
    "Stratégie & Pilotage",
    "Administration"
]

# ==================================================
# FONCTIONS UTILITAIRES
# ==================================================
def format_cfa(valeur):
    if valeur is None or pd.isna(valeur): return "0 FCFA"
    return f"{valeur:,.0f}".replace(",", " ") + " FCFA"

def format_nombre(valeur):
    if valeur is None or pd.isna(valeur): return "0"
    return f"{valeur:,.0f}".replace(",", " ")

def send_email_with_pdf(pdf_path, recipient_email):
    try:
        SMTP_SERVER = "smtp.gmail.com"
        SMTP_PORT = 587
        SMTP_USER = "ton_email@gmail.com"
        SMTP_PASSWORD = "ton_mot_de_passe_application"
        msg = EmailMessage()
        msg["Subject"] = f"Rapport Exécutif N'NAM AGRO INDUSTRIE - {datetime.now().strftime('%d/%m/%Y')}"
        msg["From"] = SMTP_USER
        msg["To"] = recipient_email
        msg.set_content(f"Bonjour,\n\nVeuillez trouver ci-joint le rapport de performance exécutif.\n\nGénéré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}\n\n--\nN'NAM AGRO INDUSTRIE")
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

def dataframe_to_table(df, max_rows=20, col_widths=None):
    if df is None or df.empty: return None
    display_df = df.head(max_rows).copy()
    for col in display_df.columns:
        if display_df[col].dtype in ['float64', 'int64']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "0")
        else:
            display_df[col] = display_df[col].astype(str)
    data = [display_df.columns.tolist()] + display_df.values.tolist()
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    return table

def add_kpi_line(c, label, valeur, x, y):
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, f"{label} :")
    c.setFont("Helvetica", 10)
    c.drawString(x + 130, y, valeur)
    return y - 16

def add_section_title(c, title, y, width):
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0.12, 0.47, 0.71)
    c.drawString(50, y, title)
    c.setStrokeColorRGB(0.12, 0.47, 0.71)
    c.setLineWidth(1)
    c.line(50, y - 4, width - 50, y - 4)
    c.setFillColorRGB(0, 0, 0)
    return y - 22

# ==================================================
# CHARGEMENT AUTO DES DONNÉES DEPUIS LA BDD
# ==================================================
@st.cache_data(ttl=300, show_spinner="Chargement des données pour le rapport...")
def charger_donnees_rapport():
    """Charge toutes les données nécessaires depuis la base"""
    data = {}
    
    # Factures
    facture = get_data("SELECT * FROM facture_client")
    if facture is not None and not facture.empty:
        facture['DATE_CREATION'] = pd.to_datetime(facture['DATE_CREATION'], errors='coerce')
        facture['ANNEE'] = facture['DATE_CREATION'].dt.year
        facture['MOIS'] = facture['DATE_CREATION'].dt.month
        facture['MOIS_NOM'] = facture['MOIS'].map(MOIS_FR)
        data['facture'] = facture
        data['annee'] = facture['ANNEE'].max() if 'ANNEE' in facture.columns else datetime.now().year
    else:
        data['facture'] = pd.DataFrame()
        data['annee'] = datetime.now().year
    
    # Personnes
    personne = get_data("SELECT * FROM personne")
    data['personne'] = personne if personne is not None else pd.DataFrame()
    
    # Stock
    stock = get_data("SELECT * FROM stock")
    if stock is not None and not stock.empty:
        produit = get_data("SELECT * FROM produit")
        if produit is not None and not produit.empty:
            stock = stock.merge(produit[['ID_PRODUIT', 'DESIGNATION']], on='ID_PRODUIT', how='left')
        magasin = get_data("SELECT * FROM magasin")
        if magasin is not None and not magasin.empty:
            stock = stock.merge(magasin[['ID_MAGASIN', 'NOM_MAGASIN']], on='ID_MAGASIN', how='left')
    data['stock'] = stock if stock is not None else pd.DataFrame()
    
    # Lignes factures (ventes produits)
    lignes = get_data("SELECT * FROM ligne_facture_client")
    data['lignes_facture'] = lignes if lignes is not None else pd.DataFrame()
    
    # Production
    prod = get_data("SELECT * FROM conditionnement_production")
    if prod is not None and not prod.empty:
        if 'DATE_PRODUCTION' in prod.columns:
            prod['DATE_PRODUCTION'] = pd.to_datetime(prod['DATE_PRODUCTION'], errors='coerce')
            prod['MOIS'] = prod['DATE_PRODUCTION'].dt.month
            prod['MOIS_NOM'] = prod['MOIS'].map(MOIS_ABBR)
    data['production'] = prod if prod is not None else pd.DataFrame()
    
    return data

# Chargement
data_rapport = charger_donnees_rapport()
facture = data_rapport.get('facture', pd.DataFrame())
stock = data_rapport.get('stock', pd.DataFrame())
lignes = data_rapport.get('lignes_facture', pd.DataFrame())
production = data_rapport.get('production', pd.DataFrame())
annee = data_rapport.get('annee', datetime.now().year)

# ==================================================
# CALCUL DES KPIs DEPUIS LES DONNÉES RÉELLES
# ==================================================
ca_total = facture['MONTANT_NET'].sum() if not facture.empty and 'MONTANT_NET' in facture.columns else 0
nb_clients = facture['ID_PERSONNE'].nunique() if not facture.empty and 'ID_PERSONNE' in facture.columns else 0
nb_commandes = len(facture)
panier_moyen = ca_total / nb_clients if nb_clients > 0 else 0
stock_total = stock['QUANTITE'].sum() if not stock.empty and 'QUANTITE' in stock.columns else 0

# Taux de perte
if not production.empty and 'QUANTITE_REELLE' in production.columns:
    qte_totale = production['QUANTITE_REELLE'].sum()
    pertes_cols = ['PERDE_EN_BOUTEILLE', 'PERDE_EN_CAPSULE', 'PERDE_EN_ETIQUETTE', 'PERDE_EN_CARTON', 'QUANTITE_AVARIE']
    pertes_presentes = [c for c in pertes_cols if c in production.columns]
    if pertes_presentes:
        production['PERTES_TOTALES'] = production[pertes_presentes].fillna(0).sum(axis=1)
        taux_perte = (production['PERTES_TOTALES'].sum() / qte_totale * 100) if qte_totale > 0 else 0
    else:
        taux_perte = 0
else:
    taux_perte = 0

# Données saisons
if not facture.empty and 'MOIS' in facture.columns:
    saison_data = facture.copy()
    saison_data['SAISON'] = saison_data['MOIS'].apply(lambda m: 'Saison des pluies' if m in [4,5,6,7,8,9,10] else 'Saison sèche')
    saison_agg = saison_data.groupby('SAISON')['MONTANT_NET'].agg(['sum', 'mean', 'count']).reset_index()
    saison_agg.columns = ['Saison', 'CA_Total', 'CA_Moyen', 'Nb_Factures']
else:
    saison_agg = pd.DataFrame()

# Coefficients saisonnalité
if not facture.empty and 'MOIS' in facture.columns:
    ca_mois = facture.groupby('MOIS')['MONTANT_NET'].sum()
    ca_moyen = ca_mois.mean()
    coeff_list = []
    for m in range(1, 13):
        coeff = round(ca_mois.get(m, 0) / ca_moyen, 2) if ca_moyen > 0 else 0
        potentiel = 'Fort' if coeff > 1.1 else ('Neutre' if coeff >= 0.9 else 'Faible')
        coeff_list.append({'Mois': MOIS_ABBR[m], 'Coefficient': coeff, 'Potentiel': potentiel})
    coeff_data = pd.DataFrame(coeff_list)
else:
    coeff_data = pd.DataFrame()

# Top produits
if not lignes.empty:
    produits_agg = lignes.groupby('DESIGNATION').agg(
        QUANTITE_VENDUE=('QUANTITE', 'sum'),
        CA_PRODUIT=('PRIX_UNITAIRE', lambda x: (x * lignes.loc[x.index, 'QUANTITE']).sum())
    ).reset_index().sort_values('QUANTITE_VENDUE', ascending=False)
else:
    produits_agg = pd.DataFrame()

# Stocks data
stocks_display = stock[['DESIGNATION', 'NOM_MAGASIN', 'QUANTITE']].copy() if not stock.empty and 'DESIGNATION' in stock.columns else pd.DataFrame()

# MP data (ID_CATEGORIE_PRODUIT IN 1,2,4)
mp_data = stock[stock['ID_CATEGORIE_PRODUIT'].isin([1,2,4])][['DESIGNATION', 'QUANTITE']].copy() if not stock.empty and 'ID_CATEGORIE_PRODUIT' in stock.columns else pd.DataFrame()

# Production vs Ventes
if not facture.empty and 'MOIS' in facture.columns:
    ventes_mens = facture.groupby('MOIS').agg(Nb_Ventes=('MONTANT_NET', 'count')).reset_index()
    ventes_mens['Mois_Nom'] = ventes_mens['MOIS'].map(MOIS_ABBR)
    
    if not production.empty and 'MOIS' in production.columns and 'QUANTITE_REELLE' in production.columns:
        prod_mens = production.groupby('MOIS')['QUANTITE_REELLE'].sum().reset_index()
        prod_mens.columns = ['MOIS', 'Volume_Produit']
        comp = ventes_mens.merge(prod_mens, on='MOIS', how='outer').fillna(0)
        comp['Ecart'] = comp['Volume_Produit'] - comp['Nb_Ventes']
        comp['Taux_Ecart'] = (comp['Ecart'] / comp['Volume_Produit'].replace(0, 1) * 100).round(1)
        comp['Mois_Nom'] = comp['MOIS'].map(MOIS_ABBR)
    else:
        comp = ventes_mens.copy()
        comp['Volume_Produit'] = 0
        comp['Ecart'] = 0
        comp['Taux_Ecart'] = 0
else:
    comp = pd.DataFrame()

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
    y -= 30
    
    # ==================================================
    # SECTION ACCUEIL
    # ==================================================
    if "Accueil" in sections:
        y = add_section_title(c, "VUE D'ENSEMBLE — INDICATEURS CLÉS", y, width)
        y = add_kpi_line(c, "Chiffre d'Affaires", format_cfa(ca_total), 60, y)
        y = add_kpi_line(c, "Clients Actifs", format_nombre(nb_clients), 60, y)
        y = add_kpi_line(c, "Commandes", format_nombre(nb_commandes), 60, y)
        y = add_kpi_line(c, "Panier Moyen/Client", format_cfa(panier_moyen), 60, y)
        y = add_kpi_line(c, "Stock Total", format_nombre(stock_total), 60, y)
        y = add_kpi_line(c, "Taux de Perte", f"{taux_perte:.1f}%", 60, y)
        c.showPage()
        y = height - 170
    
    # ==================================================
    # SECTION PÔLE COMMERCIAL
    # ==================================================
    if "Pôle Commercial" in sections:
        y = add_section_title(c, "PÔLE COMMERCIAL", y, width)
        
        if not saison_agg.empty:
            y = add_section_title(c, "Corrélation Saisons", y, width)
            table = dataframe_to_table(saison_agg, max_rows=6, col_widths=[120, 90, 90, 80])
            if table: table.wrapOn(c, width, height); table.drawOn(c, 55, y - 90); y -= 110
        
        if not coeff_data.empty:
            table = dataframe_to_table(coeff_data, max_rows=12, col_widths=[50, 70, 70])
            if table: table.wrapOn(c, width, height); table.drawOn(c, 55, y - 170); y -= 190
        
        if y < 150: c.showPage(); y = height - 170
        
        if not produits_agg.empty:
            y = add_section_title(c, "Top Produits", y, width)
            table = dataframe_to_table(produits_agg.head(10), max_rows=10, col_widths=[180, 100, 100])
            if table: table.wrapOn(c, width, height); table.drawOn(c, 55, y - 160); y -= 180
        
        c.showPage()
        y = height - 170
    
    # ==================================================
    # SECTION PÔLE PRODUCTION
    # ==================================================
    if "⚙️ Pôle Production" in sections:
        y = add_section_title(c, "PÔLE PRODUCTION", y, width)
        
        if not comp.empty:
            y = add_section_title(c, "Production vs Ventes", y, width)
            table = dataframe_to_table(comp, max_rows=12, col_widths=[55, 90, 90, 70, 70])
            if table: table.wrapOn(c, width, height); table.drawOn(c, 55, y - 180); y -= 200
        
        if y < 150: c.showPage(); y = height - 170
        
        if not stocks_display.empty:
            y = add_section_title(c, "Stocks & Magasins", y, width)
            table = dataframe_to_table(stocks_display.head(20), max_rows=20, col_widths=[160, 100, 90])
            if table: table.wrapOn(c, width, height); table.drawOn(c, 55, y - 280); y -= 300
        
        if y < 150: c.showPage(); y = height - 170
        
        if not mp_data.empty:
            y = add_section_title(c, "Matières Premières", y, width)
            table = dataframe_to_table(mp_data, max_rows=15, col_widths=[200, 100])
            if table: table.wrapOn(c, width, height); table.drawOn(c, 55, y - 200); y -= 220
        
        c.showPage()
        y = height - 170
    
    # ==================================================
    # SECTION ADMINISTRATION
    # ==================================================
    if "Administration" in sections:
        y = add_section_title(c, "ADMINISTRATION", y, width)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(60, y, "Profils utilisateurs :")
        y -= 18
        c.setFont("Helvetica", 10)
        for p in ["Administrateur — Accès complet", "Directeur Commercial — Ventes, clients, cartographie",
                   "Responsable Production — Production, MP, stocks", "Consultant Externe — Lecture seule"]:
            c.drawString(70, y, f"• {p}")
            y -= 14
        c.showPage()
    
    # ===============================
    # FOOTER
    # ===============================
    c.setFont("Helvetica-Oblique", 8)
    c.drawRightString(width - 40, 30, "Usage interne – Document confidentiel")
    c.save()
    return file_name

# ==================================================
# TITRE
# ==================================================
st.title("Rapports & Exports")
st.caption("Génération de rapports PDF exécutifs complets")

# ==================================================
# STATUT DES DONNÉES
# ==================================================
st.markdown("---")
st.markdown("### Données chargées")

col_d1, col_d2, col_d3 = st.columns(3)
with col_d1: st.metric("Factures", f"{len(facture):,}")
with col_d2: st.metric("Produits en stock", f"{len(stock):,}")
with col_d3: st.metric("Lignes de vente", f"{len(lignes):,}")

# ==================================================
# SÉLECTION DES PAGES
# ==================================================
st.markdown("---")
st.markdown("## Pages à inclure")

sections_selectionnees = st.multiselect(
    "Pages",
    PAGES_DISPONIBLES,
    default=PAGES_DISPONIBLES
)

# ==================================================
# GÉNÉRATION ET TÉLÉCHARGEMENT
# ==================================================
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("Générer le rapport PDF", type="primary", use_container_width=True):
        with st.spinner("Génération du rapport en cours..."):
            pdf_path = generate_pdf(sections_selectionnees)
            st.success("Rapport généré avec succès !")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Télécharger le PDF",
                    data=f,
                    file_name=f"rapport_NNAM_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

with col2:
    st.markdown("### Envoi par Email")
    email = st.text_input("Adresse email", placeholder="direction@nnam-agro.com")
    if st.button("Envoyer par email", use_container_width=True):
        if email:
            with st.spinner("Génération et envoi en cours..."):
                pdf_path = generate_pdf(sections_selectionnees)
                if send_email_with_pdf(pdf_path, email):
                    st.success(f"Rapport envoyé à {email}")
                else:
                    st.error("Échec de l'envoi.")
        else:
            st.warning("Veuillez saisir une adresse email.")

st.markdown("---")
st.caption(f"Module Rapports & Exports | N'NAM AGRO INDUSTRIE | {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
