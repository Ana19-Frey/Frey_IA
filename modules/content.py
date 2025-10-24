# modules/content.py (Correction de la fonction generate_content)

from google import genai
from google.genai import types # S'assurer que 'types' est importé au début du fichier

def generate_content(client: genai.Client, subject: str, ton: str, system_prompt: str) -> str:
    """
    ✅ Génère du contenu textuel avec Gemini.
    """

    prompt = f"""
{system_prompt}

🎯 Objectif :
Rédige un texte fluide, pertinent et de qualité professionnelle sur le sujet suivant,
en adoptant un ton **{ton}**.

📝 Sujet : "{subject}"
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(
                 temperature=0.7,
                 system_instruction=system_prompt # Le prompt système est appliqué directement ici
            )
        )

        # ⚠️ SOLUTION : VÉRIFICATION SIMPLE ET ROBUSTE
        if response.text:
            return response.text.strip()
        
        # Si la réponse est vide (souvent à cause d'un filtre de sécurité)
        # On tente d'extraire le message du filtre pour le diagnostic
        try:
             # Ceci fonctionne avec la structure moderne de google-genai
             feedback = response.prompt_feedback.block_reason.name if response.prompt_feedback.block_reason else "Non spécifié"
             return f"⚠️ La génération de contenu a échoué. Réponse vide (Filtre de Sécurité ? Raison: {feedback})"
        except Exception:
             return "⚠️ La génération de contenu a échoué. La réponse de l'API était vide."
        
    except Exception as e:
        return f"🚨 ERREUR API GEMINI lors de la génération : {e}"