import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from typing import Any, Optional
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

# --- ⚠️ IMPORTS DES MODULES PYTHON EXISTANTS ---
# Assurez-vous que ces fichiers existent dans votre dossier 'modules/' 
# et contiennent les fonctions suivantes.
try:
    from modules.chatbot import process_chatbot_query 
    # J'utilise data_analyst.py et content.py basés sur les conventions
    from modules.data_analyst import analyze_data_pandas, format_analysis_with_gemini
    from modules.content import generate_content
except ImportError as e:
    print(f"ATTENTION: Une erreur d'importation de module s'est produite: {e}")
    print("Vérifiez les noms des fichiers et des fonctions dans le dossier 'modules/'.")

# --- 🧠 PROMPT SYSTÉMIQUE FREY ---
# --- 🧠 PROMPT SYSTÉMIQUE FREY ---
FREY_SYSTEM_PROMPT = """
Tu es FREY, une Intelligence Artificielle multifonctionnelle.
Ton style est professionnel, bienveillant, pédagogique et inspirant.
Ton objectif est de simplifier la vie de l’utilisateur, de répondre rapidement et de produire du contenu de qualité.
Tu es clair, fluide, amical, mais précis.

RÈGLES DE RÉPONSE OBLIGATOIRES :
1. Structure : Fournis toujours une réponse claire et détaillée en utilisant des titres, sous-titres, et emojis pertinents. **Utilise fréquemment des sauts de ligne pour créer des micro-paragraphes et rendre le texte très aéré et scannable.**
2. Synthèse : Termine TOUJOURS ta réponse par deux sections en gras :
    - **Résumé :** [Synthèse de ta réponse en une seule phrase.]
    - **Suggestion :** [Une action concrète ou une piste de réflexion basée sur la réponse/l'analyse.]
3. Ambiguïté : Si la question est floue, reformule-la avant de répondre.
4. Contenu : Ne réponds JAMAIS "je ne sais pas". Propose toujours une explication logique ou une piste.
"""

# --- 🔑 SÉCURITÉ ET INITIALISATION GEMINI --- 
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
    
    if not GEMINI_API_KEY:
        print("FATAL: GEMINI_API_KEY non définie. Assurez-vous de la définir dans votre fichier .env ou comme variable d'environnement.")
        gemini_client = None 
    else:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        
except Exception as e:
    print(f"Erreur d'initialisation de l'API Gemini : {e}")
    gemini_client = None


# --- Configuration FastAPI et CORS ---
app = FastAPI(title="FREY IA API")

# Configuration des origines autorisées pour le CORS (essentiel pour React)
origins = [
    "http://localhost",
    "http://localhost:5173",  # Front-End React (Vite)
    "http://127.0.0.1:5173",  # Alternative localhost
    "https://frey-ia-front.vercel.app" ,
    # Ajouter ici l'URL de votre application React en production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modèles de Requête (Schémas Pydantic) ---
class ChatRequest(BaseModel):
    user_prompt: str
    history: Optional[list[Any]] = None

class AnalyzeRequest(BaseModel):
    data_input: str

class ContentRequest(BaseModel):
    subject: str
    ton: str

# --- 🚀 Endpoint 1 : Chatbot (/api/chat) ---
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Service Gemini non disponible : Clé API manquante ou invalide.")
    
    try:
        # NOTE: La gestion de la mémoire est simplifiée ici. Elle doit être implémentée 
        # dans modules/chatbot.py ou en utilisant directement l'historique dans cet endpoint.
        
        # Création d'une nouvelle session de chat pour chaque requête
        chat_session = gemini_client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig()
        )
        # Envoi du message à la session de chat
        response = chat_session.send_message(request.user_prompt)
        
        return {"success": True, "response": response.text.strip()}
        
    except Exception as e:
        print(f"Erreur lors de l'appel Gemini (/api/chat): {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de l'appel Gemini : {str(e)}")


# --- 📊 Endpoint 2 : Analyseur de Données (/api/analyze) ---
@app.post("/api/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    
    if not gemini_client:
         raise HTTPException(status_code=503, detail="Service Gemini non disponible.")
    
    try:
        # 1. Analyse des données (retourne le rapport brut)
        # Note: on passe False pour is_file car l'API reçoit des chaînes de caractères du Front-End React
        analysis_report = analyze_data_pandas(request.data_input, is_file=False) 
        
        # Gérer l'échec de lecture des données avant d'appeler Gemini
        if analysis_report.startswith("Échec de la lecture des données"):
             raise ValueError(analysis_report)
        
        # 2. Formatage du rapport par Gemini
        formatted_report = format_analysis_with_gemini(
            client=gemini_client,
            raw_analysis=analysis_report,
            system_prompt=FREY_SYSTEM_PROMPT  # <-- AJOUTER LE PROMPT SYSTÈME GLOBAL
        )
        
        return {"success": True, "report": formatted_report.strip()}
        
    except Exception as e:
        print(f"Erreur lors de l'analyse de données (/api/analyze): {e}")
        # Renvoie une erreur HTTP 500 ou 400 (pour l'échec de lecture des données)
        if str(e).startswith("Échec de la lecture des données"):
             raise HTTPException(status_code=400, detail=str(e))
        else:
             raise HTTPException(status_code=500, detail=f"Erreur lors du traitement des données : {str(e)}")


# --- ✍️ Endpoint 3 : Générateur de Contenu (/api/generate) ---
@app.post("/api/generate")
async def generate_endpoint(request: ContentRequest):
    
    if not gemini_client:
         raise HTTPException(status_code=503, detail="Service Gemini non disponible.")
    
    try:
        # ⚠️ MODIFICATION ICI : Appel à la fonction de génération de contenu avec le system_prompt
        generated_content = generate_content(
            client=gemini_client,
            subject=request.subject,
            ton=request.ton,
            system_prompt=FREY_SYSTEM_PROMPT # <--- AJOUTER LE PROMPT SYSTÈME GLOBAL
        )
        
        return {"success": True, "content": generated_content.strip()}
        
    except Exception as e:
        print(f"Erreur lors de la génération de contenu (/api/generate): {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de contenu : {str(e)}")

# --- 🔍 Endpoint 4 : Lister les modèles Gemini disponibles (/api/models) ---
@app.get("/api/models")
async def list_models_endpoint():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Service Gemini non disponible : Clé API manquante ou invalide.")
    
    try:
        # Initialiser le client genai pour lister les modèles
        # Note: genai.list_models() n'a pas besoin d'un GenerativeModel
        # mais d'un client configuré avec la clé API.
        # Nous utilisons la clé directement ici.
        models = []
        for m in gemini_client.list_models():
            if "generateContent" in m.supported_generation_methods:
                models.append({"name": m.name, "display_name": m.display_name})
        
        return {"success": True, "models": models}
        
    except Exception as e:
        print(f"Erreur lors de la récupération des modèles Gemini : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des modèles Gemini : {str(e)}")
        
# --- 🏃‍♂️ INSTRUCTION POUR LANCER LE SERVEUR ---
# Lancez le serveur avec la commande dans votre terminal :
# uvicorn api_server:app --reload
