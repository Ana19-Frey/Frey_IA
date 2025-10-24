# modules/content_gen.py

from google import genai
from google.genai import types

def generate_content(client: genai.Client, subject: str, ton: str, system_prompt: str) -> str:
    """
    Génère du contenu en fonction du sujet et du ton demandé par l'utilisateur via l'API Gemini.
    
    Args:
        client (genai.Client): Le client Gemini initialisé.
        subject (str): Le sujet et les instructions de rédaction.
        ton (str): Le ton souhaité (Professionnel, Amical, Drôle, Inspirant).
        system_prompt (str): Le prompt système de FREY.
        
    Returns:
        str: Le contenu généré et formaté.
    """
    
    # Construction du prompt utilisateur intégrant le ton spécifique et l'identité FREY
    user_prompt = f"Rédigez un contenu sur le sujet suivant : '{subject}'. Le ton de la rédaction doit être strictement : {ton}. Respectez toutes les règles FREY, y compris la structure et la synthèse finale."
    
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.7  # Un peu plus de créativité pour la génération de contenu
    )

    try:
        # Appel à l'API Gemini
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_prompt],
            config=config,
        )

        # Retourne la réponse complète formatée par le modèle
        return response.text.strip()

    except Exception as e:
        return f"🚨 ERREUR API GEMINI lors de la génération de contenu : {e}."