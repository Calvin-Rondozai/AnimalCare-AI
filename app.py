import streamlit as st
import pandas as pd
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import base64
import re

# Load NLP model more efficiently
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_md")

# Load and preprocess dataset
@st.cache_data
def load_disease_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"Unnamed: 0": "Disease", "Advice/ Prevention": "Advice_Prevention"})
    df.fillna("Information not available", inplace=True)
    return df

# Preprocess data more efficiently
@st.cache_data
def preprocess_disease_data(df):
    nlp = load_nlp_model()
    disease_dict = {}
    symptoms_list = []
    disease_names = []
    symptom_vectors = []
    
    # Create a symptom index for lookup
    symptom_to_diseases = {}

    for _, row in df.iterrows():
        disease = row["Disease"].strip().lower()
        symptoms = row["Symptoms"].strip().lower()
        treatment = row["Treatment"].strip()
        prevention = row["Advice_Prevention"].strip()

        disease_dict[disease] = {
            "symptoms": symptoms,
            "treatment": treatment,
            "prevention": prevention
        }

        # Extract individual symptoms and map them to diseases
        symptom_keywords = extract_symptoms(symptoms)
        for symptom in symptom_keywords:
            if symptom in symptom_to_diseases:
                symptom_to_diseases[symptom].append(disease)
            else:
                symptom_to_diseases[symptom] = [disease]

        symptoms_list.append(symptoms)
        disease_names.append(disease)
        symptom_vectors.append(nlp(symptoms).vector)

    return disease_dict, symptoms_list, disease_names, np.array(symptom_vectors), symptom_to_diseases

# Extract key symptoms from text
def extract_symptoms(symptoms_text):
    # Convert to lowercase and split by common separators
    symptoms = re.split(r'[,;.]', symptoms_text.lower())
    # Clean up and filter
    symptom_keywords = []
    for symptom in symptoms:
        symptom = symptom.strip()
        if len(symptom) > 3:  # Filter out very short tokens
            symptom_keywords.append(symptom)
            # Also add key individual words that might be symptoms
            words = symptom.split()
            for word in words:
                if len(word) > 3 and word not in ['with', 'and', 'the', 'that', 'this', 'have', 'has', 'had']:
                    symptom_keywords.append(word)
    return symptom_keywords

# Vectorization
@st.cache_resource
def create_vectorizer(symptoms_list):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(symptoms_list)
    return vectorizer, tfidf_matrix

# Main application setup
def main():
    st.set_page_config(
        page_title="Animal Disease Helper", 
        page_icon="🐾",
        layout="centered"
    )

    # Function to convert image to base64
    def get_base64_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

    # Load image as base64
    image_path = "icon.jpg" 
    base64_image = get_base64_image(image_path)

    # Custom CSS for styling
    st.markdown("""
    <style>
    .stApp {
        padding-top: 0px;
        background-color: #f0f2f5;  /* WhatsApp-like light background */
        max-width: 500px;
        height: 98vh;
        margin: 0 auto;
        font-family: 'Helvetica Neue', Arial, sans-serif;
       } 
    .top-header {
        position: fixed; /* Fixes the header at the top */
        top: 65px;
        width: 30.5%;
        display: flex;
        align-items: center;
        background-color: #008069;  /* WhatsApp main green */
        color: white;
        bottom-padding: 10px;
        padding: 10px 15px;
        border-radius: 10px 10px 0 0;  /* Rounded top corners */
        z-index: 1000; /* Keeps it above other content */
    }
    .top-header img {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
        object-fit: cover;
    }
    .top-header-text {
        display: flex;
        flex-direction: column;
    }
    .top-header-name {
        font-size: 18px;
        font-weight: 600;
    }
    .top-header-status {
        font-size: 12px;
        color: rgba(255,255,255,0.7);
    }
    .chat-container {
        height: auto;
        max-height: 70vh;
        overflow-y: auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
        margin-top: -10px;  /* Remove extra space */
    }
    .user-message {
        background-color: #dcf8c6;  /* WhatsApp user message green */
        align-self: flex-end;
        margin: 5px 0;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        margin-left: auto;
        color: #333;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .bot-message {
        background-color: #ffffff;
        align-self: flex-start;
        margin: 5px 0;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        color: #333;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .info-section {
        border-left: 3px solid #008069;
        padding-left: 10px;
        margin: 8px 0;
    }
    .info-header {
        font-weight: bold;
        margin-bottom: 3px;
    }
    .disease-match {
        background-color: #e7f3ff;
        border-left: 3px solid #1e88e5;
        padding: 8px 10px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
    }
    .disease-match-title {
        font-weight: bold;
        color: #1e88e5;
    }
    .match-score {
        float: right;
        background-color: #1e88e5;
        color: white;
        border-radius: 10px;
        padding: 1px 8px;
        font-size: 0.8em;
    }
    .disease-button {
        display: inline-block;
        margin: 5px 5px 5px 0;
        padding: 5px 10px;
        background-color: #e0f2f1;
        border: 1px solid #80cbc4;
        border-radius: 15px;
        color: #00796b;
        font-size: 0.9em;
        cursor: pointer;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: black !important;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .stTextInput > div > div > input::placeholder {
        color: #aaa;
        opacity: 0.7;
    }
     button {
        background-color: #008069 !important;  
        color: white !important;
        border-radius: 10px !important;
    }
    button:hover {
        border: 1px solid #555555 !important;    
    }
    
    .stButton {
        position: fixed;
        bottom: 4vh;
        right : 35vw;
        width: 5vw;
    }
    .stTextInput {
        position: fixed;
        bottom: 4vh;
        width: 24vw;
    }
    </style> 
    """, unsafe_allow_html=True)

    # Top header with icon, name, and status
    if base64_image:
        st.markdown(f'''
        <div class="top-header">
            <img src="data:image/jpeg;base64,{base64_image}" alt="Chatbot Icon">
            <div class="top-header-text">
                <div class="top-header-name">Animal Disease Helper</div>
                <div class="top-header-status">Online | VetAssist</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="top-header">
            <div class="top-header-text">
                <div class="top-header-name">Animal Disease Helper</div>
                <div class="top-header-status">Online | VetAssist</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Load data
    nlp = load_nlp_model()
    df = load_disease_data("Animal disease spreadsheet - Sheet1.csv")
    disease_dict, symptoms_list, disease_names, symptom_vectors, symptom_to_diseases = preprocess_disease_data(df)
    vectorizer, tfidf_matrix = create_vectorizer(symptoms_list)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "bot", "text": "🐾 Welcome to VetAssist! I'm here to help you identify and learn about animal diseases.\n\n"
                                    "How can I assist you today?\n\n"
                                    "• Type a disease name to get detailed information\n"
                                    "• Describe symptoms you've observed\n"
                                    "• Type 'help' for more assistance\n"
                                    "• Type 'list' to see all diseases I know about"}
        ]
    
    # Track conversation state for multi-turn interactions
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = {
            "current_disease": None,
            "info_stage": None,  # Can be "overview", "symptoms", "treatment", "prevention"
            "waiting_for_next": False,
            "disease_matches": []  # Store multiple disease matches
        }
    
    # Initialize input state
    if "input_key" not in st.session_state:
        st.session_state.input_key = "initial"

    # Helper functions
    def preprocess_text(text):
        doc = nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop])

    def find_diseases_by_symptoms(user_symptoms):
        """Find all diseases that match the symptoms with similarity scores"""
        user_symptoms_preprocessed = preprocess_text(user_symptoms)
        user_vector = nlp(user_symptoms_preprocessed).vector
        
        # Calculate similarity for all diseases
        similarities = cosine_similarity([user_vector], symptom_vectors)[0]
        
        # Get all diseases with similarity above threshold
        matches = []
        for i, score in enumerate(similarities):
            if score > 0.4:  # Lower threshold to catch more potential matches
                matches.append({
                    "disease": disease_names[i],
                    "score": score,
                    "details": disease_dict[disease_names[i]]
                })
        
        # Also find direct symptom keyword matches
        symptom_keywords = extract_symptoms(user_symptoms)
        direct_matches = set()
        
        for symptom in symptom_keywords:
            if symptom in symptom_to_diseases:
                direct_matches.update(symptom_to_diseases[symptom])
        
        # Add any direct matches that weren't caught by vector similarity
        for disease in direct_matches:
            if not any(m["disease"] == disease for m in matches):
                matches.append({
                    "disease": disease,
                    "score": 0.5,  # Default score for keyword matches
                    "details": disease_dict[disease]
                })
        
        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        return matches[:5]  # Limit to top 5 results

    def format_disease_info(disease, info_type=None):
        """Format disease information with better styling"""
        if disease not in disease_dict:
            return "I don't have information about this disease in my database."
        
        details = disease_dict[disease]
        formatted_text = ""
        
        if info_type == "symptoms" or info_type is None:
            formatted_text += "<div class='info-section'>"
            formatted_text += "<div class='info-header'>🔍 SYMPTOMS:</div>"
            formatted_text += f"{details['symptoms']}"
            formatted_text += "</div>"
        
        if info_type == "treatment" or info_type is None:
            formatted_text += "<div class='info-section'>"
            formatted_text += "<div class='info-header'>💊 TREATMENT:</div>"
            formatted_text += f"{details['treatment']}"
            formatted_text += "</div>"
            
        if info_type == "prevention" or info_type is None:
            formatted_text += "<div class='info-section'>"
            formatted_text += "<div class='info-header'>🛡️ PREVENTION:</div>" 
            formatted_text += f"{details['prevention']}"
            formatted_text += "</div>"
            
        return formatted_text

    def get_disease_overview(disease):
        """Returns just the overview of a disease"""
        return f"🦠 <b>{disease.title()}</b>\n\nI have detailed information about this disease. What would you like to know?\n\n• Symptoms\n• Treatment\n• Prevention\n• All details"

    def format_disease_matches(matches):
        """Format multiple disease matches into an HTML response"""
        if not matches:
            return "I couldn't find any diseases matching those symptoms. Could you provide more details?"
        
        response = "<b>🔍 Based on the symptoms, I've identified these potential matches:</b>\n\n"
        
        # Create clickable disease buttons
        for i, match in enumerate(matches):
            disease = match["disease"]
            score = match["score"]
            confidence = int(score * 100)
            
            response += f"<div class='disease-match'>"
            response += f"<span class='disease-match-title'>{(i+1)}. {disease.title()}</span>"
            response += f"<span class='match-score'>{confidence}%</span><br>"
            
            # Add a brief snippet of symptoms
            symptoms = match["details"]["symptoms"]
            if len(symptoms) > 100:
                symptoms = symptoms[:97] + "..."
            response += f"<small><i>{symptoms}</i></small>"
            response += "</div>"
        
        response += "\n<b>Select a disease number or name to learn more:</b>"
        return response

    def continue_disease_conversation(user_input):
        """Handle conversation flow when discussing a specific disease"""
        disease = st.session_state.conversation_state["current_disease"]
        
        if not disease:
            return None
            
        input_lower = user_input.lower()
        
        # Check what the user wants to know about the current disease
        if "symptom" in input_lower:
            return f"<b>🦠 {disease.title()} - Symptoms</b>\n\n{format_disease_info(disease, 'symptoms')}"
        
        elif "treat" in input_lower:
            return f"<b>🦠 {disease.title()} - Treatment</b>\n\n{format_disease_info(disease, 'treatment')}"
            
        elif "prevent" in input_lower or "advice" in input_lower:
            return f"<b>🦠 {disease.title()} - Prevention & Advice</b>\n\n{format_disease_info(disease, 'prevention')}"
            
        elif "all" in input_lower or "detail" in input_lower or "everything" in input_lower:
            return f"<b>🦠 {disease.title()} - Complete Information</b>\n\n{format_disease_info(disease)}"
            
        # If the input isn't related to the current disease, return None to exit disease conversation mode
        return None

    def handle_disease_selection(user_input):
        """Handle when user selects from multiple disease options"""
        disease_matches = st.session_state.conversation_state["disease_matches"]
        
        if not disease_matches:
            return None
            
        input_lower = user_input.lower()
        
        # Check if user entered a number corresponding to a disease
        if input_lower.isdigit():
            index = int(input_lower) - 1
            if 0 <= index < len(disease_matches):
                selected_disease = disease_matches[index]["disease"]
                st.session_state.conversation_state["current_disease"] = selected_disease
                return get_disease_overview(selected_disease)
        
        # Check if user entered a disease name
        for match in disease_matches:
            if match["disease"] in input_lower:
                selected_disease = match["disease"]
                st.session_state.conversation_state["current_disease"] = selected_disease
                return get_disease_overview(selected_disease)

        return None

    def handle_question(user_input):
        user_input = user_input.lower()

        # First check if user is selecting from a list of diseases
        if st.session_state.conversation_state["disease_matches"]:
            selection_response = handle_disease_selection(user_input)
            if selection_response:
                return selection_response
            # If it wasn't a selection, clear the matches and continue
            st.session_state.conversation_state["disease_matches"] = []

        # Check if we're in a disease conversation
        if st.session_state.conversation_state["current_disease"]:
            disease_response = continue_disease_conversation(user_input)
            if disease_response:
                return disease_response
            else:
                # Reset the disease conversation state if the user seems to be asking about something else
                st.session_state.conversation_state["current_disease"] = None

        # Quick response for common queries
        quick_responses = {
            "hello": "👋 Hello! I'm your animal health assistant. How can I help you today?",
            "hi": "👋 Hi there! I can help identify animal diseases and provide information about symptoms, treatments, and prevention. What do you need?",
            "hey": "👋 Hey! Looking for information about an animal disease? I'm here to help!",
            "help": (
                "🐾 <b>VetAssist Guide:</b>\n\n"
                "Here's how I can help you:\n\n"
                "• <b>Find disease by name:</b> Just type the disease name\n"
                "• <b>Find disease by symptoms:</b> Describe what you're seeing\n"
                "• <b>Get specific info:</b> Ask for symptoms, treatment, or prevention\n"
                "• <b>Browse diseases:</b> Type 'list diseases'\n\n"
                "Example questions:\n"
                "- \"What are the symptoms of foot and mouth disease?\"\n"
                "- \"My cow has fever and drooling\"\n"
                "- \"How to prevent swine fever?\""
            )
        }

        # Check for quick responses first
        for key, response in quick_responses.items():
            if key == user_input or (len(user_input) < 10 and key in user_input):
                return response

        # List diseases command
        if "list" in user_input and ("disease" in user_input or len(user_input) < 10):
            # Format the disease list in columns for better readability
            disease_list = sorted([name.title() for name in disease_names])
            formatted_list = "<b>🦠 Available Diseases:</b>\n\n"
            
            # Create bullet points for diseases
            for disease in disease_list:
                formatted_list += f"• {disease}\n"
                
            return formatted_list

        # Check for specific disease mentions
        for disease in disease_names:
            if disease in user_input:
                # Set the conversation state to track this disease
                st.session_state.conversation_state["current_disease"] = disease
                
                # Check if they're asking for specific information
                if "symptom" in user_input:
                    return f"<b>🦠 {disease.title()} - Symptoms</b>\n\n{format_disease_info(disease, 'symptoms')}"
                elif "treat" in user_input:
                    return f"<b>🦠 {disease.title()} - Treatment</b>\n\n{format_disease_info(disease, 'treatment')}"
                elif "prevent" in user_input or "advice" in user_input:
                    return f"<b>🦠 {disease.title()} - Prevention</b>\n\n{format_disease_info(disease, 'prevention')}"
                else:
                    # Return disease overview and prompt for more specific questions
                    return get_disease_overview(disease)

        # Symptom-based disease identification - IMPROVED SECTION
        # Find all potential disease matches
        disease_matches = find_diseases_by_symptoms(user_input)
        
        if disease_matches:
            # Store matches in session state for follow-up selection
            st.session_state.conversation_state["disease_matches"] = disease_matches
            return format_disease_matches(disease_matches)

        return "🤔 I'm not sure I understand. Could you provide more details about the symptoms you're observing? Or ask about a specific disease by name. Type 'help' if you need guidance."

    def send_message():
        user_input = st.session_state.user_input.strip()
        
        if user_input:
            st.session_state.messages.append({"role": "user", "text": user_input})
            
            with st.spinner('Analyzing...'):
                start_time = time.time()
                response = handle_question(user_input)
                processing_time = time.time() - start_time
            
            st.session_state.messages.append({"role": "bot", "text": response})
            
            # Clear the input field
            st.session_state.user_input = ""
            
            # Use st.rerun() instead of experimental_rerun
            st.rerun()

    # Chat display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        message_class = "user-message" if msg["role"] == "user" else "bot-message"
        st.markdown(f'<div class="{message_class}">{msg["text"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # User input with send button in one row
    col1, col2 = st.columns([0.9, 0.1])  # Adjust column widths

    with col1:
        st.text_input(
            "Type your message...", 
            key="user_input", 
            placeholder="Ask about an animal disease...",
            on_change=send_message,
            label_visibility="collapsed"  # Hide the label for a cleaner look
        )

    with col2:
        st.button("➤", key="send_button", on_click=send_message, help="Send Message", 
                  use_container_width=True)  # Make button fill its container

if __name__ == "__main__":
    main()