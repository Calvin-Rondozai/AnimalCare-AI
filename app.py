import streamlit as st
import pandas as pd
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import base64

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

        symptoms_list.append(symptoms)
        disease_names.append(disease)
        symptom_vectors.append(nlp(symptoms).vector)

    return disease_dict, symptoms_list, disease_names, np.array(symptom_vectors)

# Vectorization
@st.cache_resource
def create_vectorizer(symptoms_list):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(symptoms_list)
    return vectorizer, tfidf_matrix

# Main application setup
def main():
    st.set_page_config(
        page_title="Animal Disease Chatbot", 
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
    image_path = "C:/Users/CALVIN/Downloads/hello/chatbot/icon.jpg"  # Adjust path as necessary
    base64_image = get_base64_image(image_path)

    # Custom CSS for styling
    st.markdown("""
    <style>
    .stApp {
        padding-top: 0px;
        background-color: #f0f2f5;  /* WhatsApp-like light background */
        max-width: 500px;
        max-height: 98vh;
        margin: 0 auto;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        
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
    .stButton > button {
        background-color: #008069 !important;  /* WhatsApp green */
        color: white !important;
        border-radius: 10px;
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
                <div class="top-header-status">Online</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="top-header">
            <div class="top-header-text">
                <div class="top-header-name">Animal Disease Helper</div>
                <div class="top-header-status">Online</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Load data
    nlp = load_nlp_model()
    df = load_disease_data("Animal disease spreadsheet - Sheet1.csv")
    disease_dict, symptoms_list, disease_names, symptom_vectors = preprocess_disease_data(df)
    vectorizer, tfidf_matrix = create_vectorizer(symptoms_list)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "bot", "text": "👋 Hello! How can I help you with animal diseases today?"}
        ]
    
    # Initialize input state
    if "input_key" not in st.session_state:
        st.session_state.input_key = "initial"

    # Helper functions
    def preprocess_text(text):
        doc = nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop])

    def find_diseases_by_symptom(symptom):
        """Find all diseases that match a given symptom"""
        matching_diseases = []
        symptom = symptom.lower()
        
        for disease, details in disease_dict.items():
            if symptom in details['symptoms'].lower():
                matching_diseases.append(disease)
        
        return matching_diseases

    def handle_question(user_input):
        user_input = user_input.lower().strip()

        # Quick responses with bye and thank you added
        quick_responses = {
            "hello": "👋 Hello! How can I assist you with animal diseases? Type 'help' for Help!",
            "hi": "👋 Hi there! Ready to help with animal health. Type 'help' for Help!",
            "hey": "👋 Hey! What animal health question can I answer today? Type 'help' for Help!",
            "help": (
                "📚 How to ask questions:\n\n"
                "• Ask about a disease by name\n\n"
                "• Request symptoms, treatment, or prevention\n\n"
                "• Describe a symptom\n\n"
                "• Type 'list diseases' to see all available diseases."
            ),
            "bye": "👋 Goodbye! Stay healthy and take good care of your animals. Feel free to come back anytime you need help!",
            "thank you": "😊 You're welcome! I'm always here to help you with animal health information.",
            "thanks": "😊 Happy to help! Is there anything else you'd like to know about animal diseases?",
            "goodbye": "👋 Goodbye! Wishing you and your animals good health. Take care!"
        }

        # Check for quick responses first
        for key, response in quick_responses.items():
            if key in user_input:
                return [response]

        # List diseases command
        if user_input in ["list diseases", "list"]:
            return [", ".join(disease_names).title()]

        # Specific disease queries with detailed parsing
        for disease in disease_names:
            if disease in user_input:
                details = disease_dict[disease]
                
                # Specific query parsing
                if "symptoms" in user_input:
                    return [
                        f"🤧 Symptoms of {disease.title()}: {details['symptoms']}",
                        "🤔 What else would you like to know about animal health?"
                    ]
                
                if "treatment" in user_input:
                    return [
                        f"💊 Treatment for {disease.title()}: {details['treatment']}",
                        "🤔 What else would you like to know about animal health?"
                    ]
                
                if "prevention" in user_input:
                    return [
                        f"🚫 Prevention for {disease.title()}: {details['prevention']}",
                        "🤔 What else would you like to know about animal health?"
                    ]
                
                # Default full details as separate messages if no specific query
                return [
                    f"🦠 {disease.title()} Overview",
                    f"🤧 Symptoms: {details['symptoms']}",
                    f"💊 Treatment: {details['treatment']}",
                    f"🚫 Prevention: {details['prevention']}",
                    "🤔 What else would you like to know about animal health?"
                ]

        # Symptom search
        symptom_diseases = find_diseases_by_symptom(user_input)
        if symptom_diseases:
            return [
                "\n".join([f"🦠 Disease found matching '{user_input}': {disease.title()}" for disease in symptom_diseases]),
                "🤔 What else would you like to know about animal health?"
            ]

        # Disease identification from symptoms
        disease, details = identify_disease(user_input)
        if disease:
            return [
                f"🦠 {disease.title()} detected!",
                f"🤧 Symptoms: {details['symptoms']}",
                f"💊 Treatment: {details['treatment']}",
                f"🚫 Prevention: {details['prevention']}",
                "🤔 What else would you like to know about animal health?"
            ]

        return ["🤖 Sorry, I couldn't find relevant information. Please describe symptoms or ask about a specific disease."]

    def identify_disease(user_symptoms):
        user_symptoms = preprocess_text(user_symptoms)
        user_vector = nlp(user_symptoms).vector
        similarities = cosine_similarity([user_vector], symptom_vectors)
        best_match_idx = np.argmax(similarities[0])

        if similarities[0][best_match_idx] > 0.5:
            matched_disease = disease_names[best_match_idx]
            return matched_disease, disease_dict[matched_disease]
        return None, None

    def send_message():
        user_input = st.session_state.user_input.strip()
        
        if user_input:
            st.session_state.messages.append({"role": "user", "text": user_input})
            
            with st.spinner('Analyzing...'):
                start_time = time.time()
                responses = handle_question(user_input)
                processing_time = time.time() - start_time
            
            # Add each response as a separate bot message
            for response in responses:
                st.session_state.messages.append({"role": "bot", "text": response})
            
            st.session_state.user_input = ""

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
            placeholder="Type your message...",
            on_change=send_message,
            label_visibility="collapsed"  # Hide the label for a cleaner look
        )

    with col2:
        st.button("➤", key="send_button", on_click=send_message, help="Send Message", 
                  use_container_width=True)  # Make button fill its container

if __name__ == "__main__":
    main()