import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import pickle
import numpy as np

# Configure the Gemini API
api_key = "AIzaSyDd53Xo70X7szIhXV2INe9FEhch3VJ_OGM"
genai.configure(api_key=api_key)

# Load datasets
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")

# Load your prediction model
svc = pickle.load(open('svc.pkl', 'rb'))

# Helper functions
def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].values
    desc = " ".join(desc)

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.flatten().tolist()

    med = medications[medications['Disease'] == dis]['Medication'].values.tolist()
    die = diets[diets['Disease'] == dis]['Diet'].values.tolist()

    return desc, pre, med, die

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_input))
    for item in patient_symptoms:
        input_vector[symptoms_input[item]] = 1
    return diseases_input[svc.predict([input_vector])[0]]

def generate_gemini_response(disease, desc, precautions, medications, diet):
    prompt = f"""
    The predicted disease is: {disease}.
    Description: {desc}
    Precautions: {', '.join(precautions)}
    Medications: {', '.join(medications)}
    Diet: {', '.join(diet)}.

    Provide detailed recommendations and advice.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.result

# Streamlit app
st.title("Disease Prediction with Gemini AI")

# Sidebar with developer information
st.sidebar.title("Developer Information")

# Add circular photo (replace 'your_photo.jpg' with your actual file name)
circular_photo = """
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        ">
        <img src="me.jpg"" style="
            width: 150px;
            height: 150px;
            border-radius: 50%;
            object-fit: cover;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        ">
    </div>
"""
st.sidebar.markdown(circular_photo, unsafe_allow_html=True)

# Developer Name
st.sidebar.subheader("Atharv More")

# Horizontal social media icons
social_icons = """
<div style="
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-top: 20px;">
    <a href="https://github.com/iatharvmore" target="_blank">
        <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" alt="GitHub">
    </a>
    <a href="https://x.com/atharvwxyz" target="_blank">
        <img src="https://img.icons8.com/ios-glyphs/30/000000/twitter--v1.png" alt="Twitter">
    </a>
    <a href="https://www.linkedin.com/in/atharv-more-0498b524b/" target="_blank">
        <img src="https://img.icons8.com/ios-glyphs/30/000000/linkedin-circled--v1.png" alt="LinkedIn">
    </a>
</div>
"""
st.sidebar.markdown(social_icons, unsafe_allow_html=True)
# Main content: Input symptoms
symptoms_input = st.text_input("Enter your symptoms (comma-separated):")

if symptoms_input:
    user_symptoms = [s.strip() for s in symptoms_input.split(",")]

    try:
        predicted_disease = get_predicted_value(user_symptoms)
        desc, precautions, medications, diet = helper(predicted_disease)

        # Use Gemini AI to generate detailed response
        gemini_response = generate_gemini_response(predicted_disease, desc, precautions, medications, diet)

        # Display results
        st.write("### Predicted Disease")
        st.write(predicted_disease)

        st.write("### AI-Generated Recommendations")
        st.write(gemini_response)

    except Exception as e:
        st.error(f"Error: {e}")
