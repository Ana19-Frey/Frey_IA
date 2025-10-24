# modules/chatbot.py

from google import genai
from google.genai import types
from typing import Any # Ajout pour l'annotation de type de la mémoire

def process_chatbot_query(chat_session: Any, user_prompt: str) -> dict:
    """
    Traite la requête utilisateur via la session de chat Gemini, qui gère l'historique et le prompt système.
    
    Args:
        chat_session (Any): La session de chat Gemini (objet Chat).
        user_prompt (str): La question soumise par l'utilisateur.
        
    Returns:
        dict: Contient la réponse formatée complète générée par le modèle.
    """
    
    try:
        # Envoi du message à la session de chat
        response = chat_session.send_message(user_prompt)

        # Retourne la réponse complète
        return {
            "reponse_complete": response.text.strip(),
            "success": True
        }

    except Exception as e:
        # Gérer les erreurs de l'API (y compris la surcharge)
        return {
            "reponse_complete": f"🚨 ERREUR API GEMINI : Je n'ai pas pu traiter votre demande. Détails: {e}. (La mémoire est conservée, veuillez réessayer).",
            "success": False
        }