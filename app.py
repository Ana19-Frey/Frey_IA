import streamlit as st
import os
import pandas as pd
import io
from google import genai
from google.genai import types
from typing import Any # Ajout pour l'annotation de type de la mémoire

# Importation des modules
from modules.chatbot import process_chatbot_query
from modules.data_analyst import analyze_data_pandas, format_analysis_with_gemini
from modules.content import generate_content


# --- 🧠 PROMPT SYSTÉMIQUE FREY ---
FREY_SYSTEM_PROMPT = """
Tu es FREY, une Intelligence Artificielle multifonctionnelle.
Ton style est professionnel, bienveillant, pédagogique et inspirant.
Ton objectif est de simplifier la vie de l’utilisateur, de répondre rapidement et de produire du contenu de qualité.
Tu es clair, fluide, amical, mais précis.

RÈGLES DE RÉPONSE OBLIGATOIRES :
1. Structure : Fournis toujours une réponse claire et détaillée en utilisant des titres, sous-titres, et emojis pertinents.
2. Synthèse : Termine TOUJOURS ta réponse par deux sections en gras :
    - **Résumé :** [Synthèse de ta réponse en une seule phrase.]
    - **Suggestion :** [Une action concrète ou une piste de réflexion basée sur la réponse/l'analyse.]
3. Ambiguïté : Si la question est floue, reformule-la avant de répondre.
4. Contenu : Ne réponds JAMAIS "je ne sais pas". Propose toujours une explication logique ou une piste.
"""

# --- 🔑 SÉCURITÉ ET INITIALISATION GEMINI ---
try: 
    if 'GEMINI_API_KEY' not in st.secrets:
        st.error("🚨 ERREUR : La clé 'GEMINI_API_KEY' n'est pas trouvée dans .streamlit/secrets.toml.")
        st.stop()
    
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    
    # 2. Initialisation du Client Gemini
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    
except Exception as e:
    st.error(f"Erreur d'initialisation de l'API Gemini : {e}")
    st.stop()

# --------------------------------------------------------

# --- FONCTION DE RÉINITIALISATION DE LA MÉMOIRE ---
def clear_chat_history():
    """Réinitialise la session de chat dans l'état de session de Streamlit."""
    try:
        st.session_state.chat_session = gemini_client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=FREY_SYSTEM_PROMPT
            )
        )
    except Exception as e:
        st.error(f"Erreur lors de la réinitialisation de la session de chat : {e}")

# --------------------------------------------------------

# --- 🧠 GESTION DE LA MÉMOIRE DU CHATBOT (SESSION) ---
if 'chat_session' not in st.session_state:
    try:
        st.session_state.chat_session = gemini_client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=FREY_SYSTEM_PROMPT
            )
        )
    except Exception as e:
        st.error(f"Erreur lors de la création de la session de chat : {e}")
        st.stop()
        
# --------------------------------------------------------


# --- 🎨 CONFIGURATION ET STYLE FREY ---

# 1. Configuration de la page
st.set_page_config(
    page_title="FREY - IA Multifonction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Injection du style CSS (Amélioré)
st.markdown(
    """
    <style>
    /* Palette: bleu roi #2563eb, blanc cassé #f5f7fa, gris foncé #1e293b */
    body { background-color: #f5f7fa; color: #1e293b; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stApp { background-color: #f5f7fa; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.2rem; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { border-bottom: 2px solid #2563eb; color: #2563eb; }
    
    /* Bouton principal */
    div.stButton > button {
        background-color: #2563eb; color: white; border-radius: 8px; 
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); padding: 10px 20px;
    }
    div.stButton > button:hover { background-color: #1e293b; color: white; }
    
    /* Amélioration de la lisibilité des conteneurs d'information */
    div[data-testid="stTextarea"] label,
    div[data-testid="stSelectbox"] label {
        font-weight: bold; /* Mettre en gras les étiquettes */
        color: #1e293b; /* Couleur du texte principal */
        font-size: 1.1rem;
    }

    /* Style pour la séparation des sections du rapport (HR line) */
    hr {
        border-top: 3px solid #e2e8f0; /* Ligne plus visible */
        margin: 20px 0;
    }

    /* Style pour les messages d'information (st.info) */
    div[data-testid="stAlert"] {
        border-left: 5px solid #2563eb; /* Bordure latérale pour accentuer */
        background-color: #eef2ff; /* Fond léger */
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --- 🚀 LOGIQUE PRINCIPALE DE L'INTERFACE ---

st.title("✨ FREY : Intelligence Artificielle Multifonction")
st.markdown("👋 *Professionnelle, bienveillante et inspirante, je suis prête à vous aider à comprendre, analyser et créer.*")

# Création des trois onglets
tab1, tab2, tab3 = st.tabs(["💬 Chatbot Intelligent", "📊 Analyseur de Données", "✍️ Générateur de Contenu"])

# --- 1. Onglet Chatbot (avec mémoire) ---
with tab1:
    st.header("💬 Chatbot Intelligent (Conversationnel)")
    st.info("Posez-moi des questions en séquence. FREY se souvient du contexte de notre échange.")
    
    # Bouton pour réinitialiser la conversation
    st.button('🗑️ Commencer une nouvelle conversation', on_click=clear_chat_history)
    
    # 1. Affichage de l'historique des messages
    for message in st.session_state.chat_session.get_history():
        
        if message.role == "user":
            role_name = "Utilisateur"
            avatar_icon = "👤"
        elif message.role == "model":
            role_name = "FREY"
            avatar_icon = "🤖"
        else:
            # Ignorer le rôle 'system' qui contient le prompt
            continue

        # Affiche le message (avec le nouvel avatar)
        with st.chat_message(role_name, avatar=avatar_icon):
            st.markdown(message.parts[0].text)

    # 2. Zone de saisie utilisateur
    if chatbot_input := st.chat_input("Dites quelque chose à FREY..."):
        
        # Afficher la requête de l'utilisateur immédiatement
        with st.chat_message("Utilisateur", avatar="👤"):
            st.markdown(chatbot_input)

        # Appel de la fonction Gemini avec la session de chat
        with st.spinner('FREY analyse et prépare la réponse...'):
            result = process_chatbot_query(
                chat_session=st.session_state.chat_session, # Passe la session stockée
                user_prompt=chatbot_input
            )
            
            # Afficher la réponse de FREY
            if result["success"]:
                with st.chat_message("FREY", avatar="🤖"):
                    st.markdown(result["reponse_complete"])
            else:
                 with st.chat_message("FREY", avatar="🤖"):
                    st.error(result["reponse_complete"])

# --- 2. Onglet Analyseur de Données ---
with tab2:
    st.header("📊 Analyseur Automatique de Données")
    st.info("Collez ici des données tabulaires (CSV, tabulé) ou téléchargez un fichier pour une analyse immédiate des tendances et des insights.")
    
    data_input = st.text_area("Coller vos données (CSV, tabulé) :", height=150, key='data_input')
    uploaded_file = st.file_uploader("Ou téléverser un fichier de données (CSV, TXT)", type=["csv", "txt"])

    if st.button("Analyser les Données", key='btn_analyse'):
        if data_input or uploaded_file:
            with st.spinner('FREY analyse et rédige le rapport...'):
                
                source_data = uploaded_file if uploaded_file else data_input
                is_file = uploaded_file is not None

                # Étape 1 : Analyse brute avec Pandas
                raw_analysis = analyze_data_pandas(source_data, is_file=is_file) 
                
                # Vérification des erreurs de lecture de Pandas
                if raw_analysis.startswith("Échec de la lecture des données") or raw_analysis.startswith("Le DataFrame est vide"):
                     st.error(raw_analysis)
                else:
                    # Étape 2 : Rédaction et formatage par Gemini (style FREY)
                    formatted_report = format_analysis_with_gemini(
                        client=gemini_client,
                        raw_analysis=raw_analysis,
                        system_prompt=FREY_SYSTEM_PROMPT
                    )
                    
                    st.markdown("---")
                    st.markdown(formatted_report) # Affiche le rapport complet formaté

        else:
             st.error("Veuillez coller des données ou téléverser un fichier pour commencer l'analyse.")

# --- 3. Onglet Générateur de Contenu ---
with tab3:
    st.header("✍️ Générateur de Contenu Intelligent")
    st.info("Rédigez des textes professionnels, e-mails ou publications. Choisissez le ton souhaité pour un résultat parfait.")
    
    ton_select = st.selectbox(
        'Choisissez le Ton :',
        ('Professionnel', 'Amical', 'Drôle', 'Inspirant')
    )
    
    sujet_input = st.text_area("Sujet et Instructions de Rédaction :", height=150, key='gen_input')
    
    if st.button("Générer le Contenu", key='btn_gen'):
        if sujet_input:
            with st.spinner(f'FREY rédige votre contenu en utilisant le ton {ton_select}...'):
                
                # Appel de la fonction Gemini du module de génération
                generated_text = generate_content(
                    client=gemini_client,
                    subject=sujet_input, 
                    ton=ton_select, 
                    system_prompt=FREY_SYSTEM_PROMPT
                )
                
                st.markdown("---")
                st.markdown(f"### Contenu Généré (Ton : {ton_select}) :")
                st.markdown(generated_text)
                
            
        else:
            st.warning("Veuillez fournir un sujet et des instructions pour la rédaction.")