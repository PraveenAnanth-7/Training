import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
import pickle
import base64
from tensorflow.keras.models import load_model

model = load_model('plant-disease.h5')

class_indices = {'Healthy': 0, 'Powdery': 1, 'Rust': 2}
index_to_class = {v: k for k, v in class_indices.items()}

with open("history.pkl", "rb") as f:
    history = pickle.load(f)

def add_background_image(image_file):
    with open(image_file, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()
    css_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-position: bottom center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css_style, unsafe_allow_html=True)

st.set_page_config(page_title="Plant Disease Recognition", layout="centered")
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📤 Upload Image", "📈 Model Performance"])

# Home Page
if page == "🏠 Home":
    add_background_image('images bc.jpeg')
    st.title("🌿 Plant Disease Recognition")
    st.markdown("""
    Welcome to the **Plant Disease Recognition App!** 🌱

    Upload a leaf image, and this app will tell you whether your plant is:
    - Healthy 🌿
    - Affected by Rust 🍂
    - Struggling with Powdery Mildew 🥀
    """)

elif page == "📤 Upload Image":
    st.title("📤 Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button('Predict'):
            prediction = model.predict(img_array)[0]  # Flattened prediction
            predicted_index = np.argmax(prediction)
            predicted_label = index_to_class[predicted_index]
            confidence = prediction[predicted_index]

            # Show all class probabilities
            st.subheader("🔍 Prediction Probabilities:")
            for i, prob in enumerate(prediction):
                st.write(f"- **{index_to_class[i]}**: {prob * 100:.2f}%")

            # Final prediction result
            st.success(f"🎯 Predicted Class: **{predicted_label}** ({confidence * 100:.2f}% confidence)")

# Model Performance Page
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Accuracy Over Epochs")
    with col2:
        st.subheader("📉 Loss Over Epochs")

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(y=history['acc'], mode='lines+markers', name='Train Accuracy', line=dict(color='green')))
    fig_acc.add_trace(go.Scatter(y=history['val_acc'], mode='lines+markers', name='Val Accuracy', line=dict(color='cyan')))
    fig_acc.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        plot_bgcolor='white',
        title="Model Accuracy"
    )

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history['loss'], mode='lines+markers', name='Train Loss', line=dict(color='red')))
    fig_loss.add_trace(go.Scatter(y=history['val_loss'], mode='lines+markers', name='Val Loss', line=dict(color='orange')))
    fig_loss.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        plot_bgcolor='white',
        title="Model Loss"
    )
    st.plotly_chart(fig_acc, use_container_width=True)
    st.plotly_chart(fig_loss, use_container_width=True)
