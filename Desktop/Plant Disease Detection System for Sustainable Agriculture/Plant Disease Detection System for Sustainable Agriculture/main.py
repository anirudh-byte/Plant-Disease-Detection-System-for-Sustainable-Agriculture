import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Load the model only once
@st.cache_resource
def load_model():
    try:
        # Ensure the model is correctly loaded
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function for model prediction
def model_prediction(model, test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Set page config
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Main Page
if app_mode == "HOME":
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>🌿 Plant Disease Detection System 🌿</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Empowering sustainable agriculture with AI and technology.")
    
    # Banner Image
    banner_image = Image.open("Diseases.png")
    st.image(banner_image, use_column_width=True, caption="Identify plant diseases with precision")

    # Add Spacer
    st.markdown("<br>", unsafe_allow_html=True)

    # Information Section
    st.markdown(
        """
        <div style="text-align: center;">
            <h3 style="color: #2E8B57;">Why Use This System?</h3>
            <p style="font-size: 18px;">🌟 Detect plant diseases early to ensure healthy crops.<br>
            🌟 Save time and resources with AI-driven insights.<br>
            🌟 Promote sustainable farming practices.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Buttons Section
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("📖 Learn More", help="Learn about plant diseases and solutions.")
    with col2:
        st.button("📷 Upload Image", help="Proceed to disease recognition by uploading an image.")
    with col3:
        st.button("🛠 Explore Features", help="Explore the features of this system.")

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    
    # Load model once
    model = load_model()
    if model is None:
        st.error("Model loading failed. Please check the model file.")
    else:
        # Upload Image
        test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
        
        if test_image is not None:
            # Show Image when the button is pressed
            if st.button("Show Image"):
                st.image(test_image, width=400, use_container_width=True)

            # Predict Button
            if st.button("Predict"):
                st.snow()
                st.write("Our Prediction")
                result_index = model_prediction(model, test_image)

                # Reading Class Labels
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                
                # Display result
                result_label = class_name[result_index]
                st.success(f"Model is Predicting: It's a **{result_label}**")
                
                # Display whether the plant is healthy or diseased
                if "healthy" in result_label.lower():
                    st.markdown("<h4 style='color: green; text-align: center;'>The plant is healthy!</h4>", unsafe_allow_html=True)
                else:
                    st.markdown("<h4 style='color: red; text-align: center;'>The plant is diseased. Please take appropriate action!</h4>", unsafe_allow_html=True)

    # Footer Animation
    st.markdown("---")
    placeholder = st.empty()

    def animate_footer():
        for _ in range(3):
            for char in "🌱 Growing Better Agriculture... 🌾":
                placeholder.markdown(
                    f"<h4 style='text-align: center; color: #4CAF50;'>{char}</h4>",
                    unsafe_allow_html=True,
                )
                time.sleep(0.1)

    animate_footer()

    st.markdown(
        "<h5 style='text-align: center; color: gray;'>© 2025 Plant Disease Detection System</h5>",
        unsafe_allow_html=True,
    )

