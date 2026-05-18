import pandas as pd
from sklearn.linear_model import LinearRegression

def forecast_sales_2026(file_path):
    # Chargement des ventes
    ventes = pd.read_excel(file_path, sheet_name="facture_client", engine="openpyxl")
    ventes["DATE_CREATION"] = pd.to_datetime(ventes["DATE_CREATION"], errors="coerce")

    # Filtre 2025
    ventes_2025 = ventes[
        (ventes["DATE_CREATION"].dt.year == 2025) &
        (ventes["VALIDER"] == 1)
    ].copy()

    ventes_2025["MOIS"] = ventes_2025["DATE_CREATION"].dt.month

    # CA mensuel
    ca_mensuel = (
        ventes_2025
        .groupby("MOIS")["MONTANT_NET"]
        .sum()
        .reset_index()
    )

    # Modèle
    X = ca_mensuel[["MOIS"]]
    y = ca_mensuel["MONTANT_NET"]

    model = LinearRegression()
    model.fit(X, y)

    # Prévision 2026
    mois_2026 = pd.DataFrame({"MOIS": range(1, 13)})
    mois_2026["CA_PREVU_2026"] = model.predict(mois_2026)

    return ca_mensuel, mois_2026, model
