def compute_kpis(df):
    return {
        "total_utilisateurs": len(df),
        "profils_uniques": df["ID_PROFIL"].nunique(),
        "employes": int(df["EMPLOYE"].sum())
    }