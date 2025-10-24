# modules/data_analyst.py
import pandas as pd
import io
from google import genai
from google.genai import types
import numpy as np

# Fonction principale pour l'analyse des données (Phase 1: Pandas)
def analyze_data_pandas(data_source, is_file: bool = False) -> str:
    """Analyse brute des données avec Pandas et retourne les résultats bruts."""
    
    try:
        if is_file:
            # Pour Streamlit, data_source est un objet UploadedFile.
            df = pd.read_csv(data_source) 
        else:
            # Lecture des données collées (utilisant io.StringIO pour lire la chaîne comme un fichier)
            df = pd.read_csv(io.StringIO(data_source), sep=r'\s*,\s*|;', engine='python', skipinitialspace=True)
            
    except Exception as e:
        return f"Échec de la lecture des données. Erreur: {e}. Assurez-vous que les données sont au format CSV ou tabulé et que les séparateurs sont corrects."

    if df.empty:
        return "Le DataFrame est vide. Veuillez fournir des données valides."

    # --- Collecte des insights bruts ---
    
    insights = []
    
    # 1. Dimensions et Types
    insights.append(f"Dimensions: {len(df)} lignes et {len(df.columns)} colonnes.")
    insights.append(f"Colonnes: {', '.join(df.columns)}.")
    insights.append(f"Types de données:\n{df.dtypes.to_string()}")
    
    # 2. Statistiques descriptives
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        insights.append(f"\nStatistiques descriptives des colonnes numériques:\n{df[numeric_cols].describe().to_string()}")

    # 3. Tendances clés (Exemple: Top 5 des valeurs pour la première colonne catégorielle)
    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        top_col = categorical_cols[0]
        top_values = df[top_col].value_counts().nlargest(5).to_string()
        insights.append(f"\nTop 5 des valeurs pour la colonne '{top_col}':\n{top_values}")

    # 4. Anomalies (Exemple: Valeurs manquantes)
    missing_data = df.isnull().sum()
    if missing_data.any():
        missing_report = missing_data[missing_data > 0].to_string()
        insights.append(f"\nAnomalies: Valeurs manquantes détectées :\n{missing_report}")
        
    return "\n---\n".join(insights)

# Fonction pour la rédaction de l'analyse par Gemini (Phase 2: LLM)
def format_analysis_with_gemini(client: genai.Client, raw_analysis: str, system_prompt: str) -> str:
    """
    Rédige les résultats bruts de Pandas dans le style FREY via l'API Gemini.
    """
    
    model = genai.GenerativeModel("gemini-2.5-flash", api_key=client.api_key)

    full_analysis_prompt = f"""
    {system_prompt}

    En tant que FREY, votre mission est de transformer l'analyse de données brutes suivante en un rapport lisible, pédagogique, et inspirant.
    
    Votre réponse doit :
    1. Décrire les tendances générales (dimensions, statistiques principales).
    2. Identifier un insight clé (anomalie, top valeur, corrélation implicite).
    3. Respecter STRICTEMENT la structure finale : Réponse claire + Résumé + Suggestion.
    
    Voici les résultats bruts de l'analyse Pandas :
    ---
    {raw_analysis}
    ---
    """
    
    config = types.GenerationConfig(
        temperature=0.3 # Faible créativité, l'accent est mis sur la fidélité aux données
    )

    try:
        response = model.generate_content(
            contents=[full_analysis_prompt],
            generation_config=config,
        )
        return response.text.strip()
    
    except Exception as e:
        return f"🚨 ERREUR API GEMINI lors de la rédaction de l'analyse : {e}."