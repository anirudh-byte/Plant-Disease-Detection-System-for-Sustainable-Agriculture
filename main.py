
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os

# --- Configuration ---
# (CLASS_NAMES, TREATMENT_RECOMMENDATIONS, HEALTHY_PLANT_ADVICE, DEFAULT_TREATMENT_ADVICE remain the same)
CLASS_NAMES = [
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
TREATMENT_RECOMMENDATIONS = {
    "Apple_scab": "Apply fungicides containing sulfur or copper. Prune affected branches and improve air circulation. Ensure good sanitation by removing fallen leaves.",
    "Black_rot": "Remove infected fruit and leaves promptly. Apply fungicides containing copper or myclobutanil during the growing season. Prune for air circulation.",
    "Cedar_apple_rust": "Apply fungicides containing myclobutanil or propiconazole. Remove nearby cedar trees if possible, or apply preventative sprays during spore release periods.",
    "Powdery_mildew": "Apply sulfur-based fungicides or horticultural oils. Increase air circulation around plants. Avoid overhead watering, especially late in the day.",
    "Cercospora_leaf_spot": "Apply fungicides containing chlorothalonil or azoxystrobin. Practice crop rotation and remove infected debris. Improve air flow.",
    "Common_rust": "Apply fungicides containing azoxystrobin or propiconazole, especially during early stages. Plant resistant varieties if available. Manage nitrogen levels.",
    "Northern_Leaf_Blight": "Apply fungicides containing pyraclostrobin or azoxystrobin preventatively or at first sign. Practice crop rotation and tillage to bury residue. Choose resistant hybrids.",
    "Haunglongbing_(Citrus_greening)": "There is no cure. Remove and destroy infected trees immediately to prevent spread. Control the Asian citrus psyllid vector with insecticides and biological controls.",
    "Bacterial_spot": "Apply copper-based bactericides preventatively. Avoid overhead irrigation to reduce spread. Prune infected parts during dry weather. Plant resistant varieties.",
    "Early_blight": "Apply fungicides containing chlorothalonil or copper. Remove affected lower leaves. Mulch around plants. Practice crop rotation. Ensure proper plant spacing.",
    "Late_blight": "Apply fungicides containing chlorothalonil or mancozeb, especially during cool, wet weather. Remove and destroy infected plants immediately. Ensure good drainage and air circulation.",
    "Leaf_Mold": "Improve air circulation significantly (pruning, spacing). Lower humidity. Apply fungicides containing chlorothalonil or copper if severe.",
    "Septoria_leaf_spot": "Apply fungicides containing chlorothalonil or copper. Practice crop rotation (3 years). Remove infected leaves and plant debris. Mulch.",
    "Spider_mites": "Use strong water sprays to dislodge mites. Apply insecticidal soap, neem oil, or specific miticides. Encourage natural predators. Increase humidity.",
    "Target_Spot": "Apply fungicides containing azoxystrobin or difenoconazole. Improve air circulation. Remove infected leaves and debris. Avoid overhead watering.",
    "Yellow_Leaf_Curl_Virus": "Control whitefly vectors using insecticides or reflective mulch. Remove and destroy infected plants immediately. Plant resistant varieties.",
    "Tomato_mosaic_virus": "No cure. Remove and destroy infected plants. Disinfect tools frequently. Wash hands after handling plants. Control weeds that may host the virus.",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides containing mancozeb or copper. Prune to improve air circulation. Remove infected leaves and canes during dormancy.",
    "Esca_(Black_Measles)": "No effective chemical control. Practice good sanitation. Remove and destroy infected vines or affected parts. Maintain vine health through proper irrigation and fertilization.",
    "Leaf_scorch": "Ensure adequate watering during dry periods. Improve soil drainage. Apply fungicides containing captan or myclobutanil if fungal pathogens are suspected. Manage nutrient levels."
}
HEALTHY_PLANT_ADVICE = """
Your plant appears healthy! Continue with good care practices:
<ul>
    <li>Provide appropriate sunlight for the species.</li>
    <li>Water according to the plant's needs (check soil moisture).</li>
    <li>Ensure good soil drainage.</li>
    <li>Fertilize appropriately during the growing season.</li>
    <li>Maintain good air circulation around the plant.</li>
    <li>Regularly monitor for any early signs of pests or diseases.</li>
</ul>
"""
DEFAULT_TREATMENT_ADVICE = "Consult with a local agricultural extension service or plant pathologist for specific treatment recommendations tailored to your region and conditions."

# --- Model Loading ---
@st.cache_resource
def load_model(model_path="trained_plant_disease_model.keras"):
    """Loads the Keras model with error handling."""
    try:
        if not os.path.exists(model_path):
            st.error(f"Error: Model file not found at {model_path}")
            return None
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Prediction Function ---
def model_prediction(model, test_image):
    """ Performs prediction, returns index and confidence. """
    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        # input_arr = input_arr / 255.0 # Uncomment if needed
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = model.predict(input_arr)
        result_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        return result_index, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# --- Styling Functions ---
def set_bg_hack():
    """Sets the background image using CSS. VERY HIGH OPACITY, NO BLUR."""
    st.markdown("""
    <style>
    .stApp {
        background-image: url('https://storage.googleapis.com/agtech-background/agtech_image.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    /* Sidebar: VERY High Opacity, NO BLUR */
    .stSidebar > div:first-child {
        background-color: rgba(255, 255, 255, 255); /* White with 98% opacity */
        /* backdrop-filter: blur(...) REMOVED */
        border-right: 1px solid rgba(0, 0, 0, 0.1);
    }
    /* Main Content Area: VERY High Opacity, NO BLUR */
     .main .block-container {
        background-color: rgba(255, 255, 255, 0.98); /* White with 98% opacity */
        border-radius: 10px;
        padding: 2rem;
         margin-top: 1rem;
         /* backdrop-filter: blur(...) REMOVED */
         box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
         border: 1px solid rgba(0, 0, 0, 0.05);
    }
    /* Ensure headers in main content block are dark */
     .main .block-container h1,
     .main .block-container h2,
     .main .block-container h3,
     .main .block-container h4 {
          color: #333;
     }
     .sub-header { /* Titles inside main block */
         color: #00FF7F !important; /* Dark Green */
     }
    </style>
    """, unsafe_allow_html=True)

def local_css():
    """Applies custom CSS styles with media queries for responsiveness."""
    st.markdown("""
    <style>
    /* Default styles for larger screens (laptops) */
    .main-header {
        font-size: 2.8rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.8);
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1B5E20;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
        border-bottom: 2px solid #1B5E20;
        padding-bottom: 0.3rem;
    }
    .card {
        border: 1px solid #D5D5D5;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        background-color: rgba(255, 255, 255, 0.99);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    .info-text,
    .main .block-container p,
    .main .block-container li {
        font-size: 17px;
        line-height: 1.7;
        color: #333333 !important;
    }
    .card h3, .card h4, .card p, .card li, .card small {
        color: #333333 !important;
    }
    .stSidebar .stMarkdown p,
    .stSidebar .stMarkdown li,
    .stSidebar .stMarkdown ul,
    .stSidebar .stMarkdown ol {
       color: #FFFFFF !important;
        font-size: 15px;
        line-height: 1.6;
    }
    .stSidebar h3 {
      color: #00FF7F !important;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .feature-icon {
        font-size: 26px;
        margin-right: 12px;
        color: #4CAF50;
    }
    .footer {
        text-align: center;
        padding: 25px 0;
        font-size: 14px;
        color: #f0f0f0;
        margin-top: 3rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
        border-top: 1px solid rgba(255, 255, 255, 0.4);
    }
    .highlight {
        background-color: rgba(220, 240, 220, 0.98);
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #4CAF50;
        margin: 15px 0;
    }
    .highlight p {
        color: #333333 !important;
    }
    .disease-result {
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        margin: 20px 0;
        border-radius: 10px;
        border: 1px solid;
    }
    .disease-result.healthy {
        background-color: #E8F5E9;
        color: #1B5E20 !important;
        border-color: #A5D6A7;
    }
    .disease-result.diseased {
        background-color: #FFEBEE;
        color: #B71C1C !important;
        border-color: #EF9A9A;
    }
    div[style*="background-color: rgba(255, 235, 238"] p,
    div[style*="background-color: rgba(255, 235, 238"] small,
    div[style*="background-color: rgba(232, 245, 233"] p,
    div[style*="background-color: rgba(232, 245, 233"] ul,
    div[style*="background-color: rgba(232, 245, 233"] li {
        color: #333333 !important;
    }
    .stTextInput > div > div > input,
    .stFileUploader > div > div > button,
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.98);
        color: #333;
        border: 1px solid #cccccc;
    }
    .stFileUploader > div > label {
        background-color: rgba(248, 248, 248, 0.98);
        padding: 8px 12px;
        border-radius: 5px;
        margin-bottom: 8px;
        display: block;
        border: 1px solid #dddddd;
        color: #37474F;
        font-weight: 500;
    }
    .stButton > button {
       background-color: #A5D6A7;
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #388E3C;
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.3);
    }
    ul, ol {
        padding-left: 20px;
        margin-top: 10px;
    }
    li {
        margin-bottom: 8px;
       color: #C62828 !important;
    }

    /* Media query for screens with a maximum width of 768px (typical for phones) */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.0rem; /* Smaller title for phones */
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.4rem; /* Smaller sub-titles for phones */
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }
        .info-text,
        .main .block-container p,
        .main .block-container li {
            font-size: 16px; /* Slightly smaller text for readability */
            line-height: 1.5;
        }
        .card {
            padding: 15px; /* Less padding on smaller screens */
            margin: 10px 0;
        }
        .stSidebar h3 {
            font-size: 1.2rem; /* Smaller sidebar titles */
        }
        .feature-icon {
            font-size: 20px; /* Smaller icons */
            margin-right: 8px;
        }
        .disease-result {
            font-size: 18px; /* Smaller result text */
            padding: 10px;
            margin: 15px 0;
        }
        .stButton > button {
            padding: 10px 15px; /* Smaller button padding */
            font-size: 14px; /* Smaller button text */
        }
    }

    /* You can add more media queries for different screen sizes if needed */
    /* For example, for tablets (between 769px and 1024px) */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-header {
            font-size: 2.4rem;
        }
        .sub-header {
            font-size: 1.6rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Function ---
def feature_card(icon, title, description):
    """Generates HTML for a feature card."""
    return f"""
    <div class="card">
        <h3><span class="feature-icon">{icon}</span> {title}</h3>
        <p class="info-text">{description}</p>
    </div>
    """

# --- Streamlit App ---
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply background and CSS
set_bg_hack()
local_css()

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px; border-radius: 10px; margin-bottom: 15px;">
        <h2 style="color: #00FF7F; text-align: center;">🌿 GreenLeaf AI</h2>
    </div>
    """, unsafe_allow_html=True)
    app_mode = st.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION", "ABOUT"], label_visibility="collapsed")
    try:
        logo_path = "logo.png" # Optional
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            st.image(logo, use_column_width=True)
        else:
            st.markdown("<h3 style='text-align: center; color: #1B5E20;'>🌱 Sustainable Agriculture Technology</h3>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading logo: {e}")
        st.markdown("<h3 style='text-align: center; color: #1B5E20;'>🌱 Sustainable Agriculture Technology</h3>", unsafe_allow_html=True)

    st.markdown("---")
    # This markdown content should now be clearly visible due to CSS rules
    st.markdown("### 🔍 How it Works")
    st.markdown("""
    1.  Navigate to **DISEASE RECOGNITION**.
    2.  Upload a clear image of a plant leaf.
    3.  Click **Analyze Image**.
    4.  View the AI's diagnosis and recommendations.
    """)
    st.markdown("---")
    st.markdown("Developed for Sustainable Agriculture")

# --- Page Content ---
model = load_model()
if model is None:
    st.error("🚨 Critical Error: The prediction model could not be loaded.")

# --- HOME Page ---
if app_mode == "HOME":
    st.markdown('<h1 class="main-header">🌿 Plant Disease Detection System 🌿</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: white; text-shadow: 1px 1px 2px black;">Leveraging AI for Healthier Crops and Sustainable Farming</p>', unsafe_allow_html=True)

    # Display Home Page Image
    home_image_path = "Diseases.png" # Assumes Diseases.png is in the same directory
    if os.path.exists(home_image_path):
        try:
            image = Image.open(home_image_path)
            st.image(image, caption="AI Enhancing Agricultural Practices", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading home page image: {e}")
    else:
        st.warning(f"Note: Home page image '{home_image_path}' not found in script directory.")

    st.markdown("---", unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Key Metrics</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    stats_data = [("38+", "Disease Classes"), ("97%+", "Est. Accuracy"), ("14+", "Plant Types"), ("~1s", "Avg. Detection")]
    cols = [col1, col2, col3, col4]
    for i, (value, label) in enumerate(stats_data):
        with cols[i]:
            # Using 99% opacity for this card background for max visibility
            st.markdown(f"""<div class="card" style="text-align: center; background-color: rgba(232, 245, 233, 0.99);"> <h2 style="color: #1B5E20; margin-bottom: 5px;">{value}</h2> <p style="color: #333;">{label}</p> </div>""", unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">Why Use This AI-Powered System?</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(feature_card("⏱️", "Early Detection", "Identify potential diseases before they become widespread."), unsafe_allow_html=True)
        st.markdown(feature_card("💡", "Informed Decisions", "Get data-driven insights for targeted treatment."), unsafe_allow_html=True)
    with col2:
        st.markdown(feature_card("📉", "Reduce Losses", "Minimize crop damage and economic losses."), unsafe_allow_html=True)
        st.markdown(feature_card("♻️", "Sustainable Practice", "Promote eco-friendly farming by reducing chemical use."), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: center; padding: 20px;" class="highlight"><h3>Ready to protect your crops?</h3><p>Navigate to the <strong>DISEASE RECOGNITION</strong> page to get started!</p></div>""", unsafe_allow_html=True)


# --- DISEASE RECOGNITION Page ---
elif app_mode == "DISEASE RECOGNITION":
    st.markdown('<h1 class="main-header">🔬 Disease Recognition Center</h1>', unsafe_allow_html=True)
    if model is None:
        st.warning("Model is not available.")
    else:
        st.markdown("""<div class="highlight"><p>Upload a clear, well-lit image of a single plant leaf...</p></div>""", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("### 1. Upload Plant Image")
            test_image = st.file_uploader("Choose Image (.jpg, .png, .jpeg):", type=["jpg", "png", "jpeg"], key="file_uploader")
            predict_btn = False
            if test_image is not None:
                try:
                    st.markdown("#### Image Preview:")
                    img = Image.open(test_image)
                    st.image(img, use_column_width=True)
                    st.markdown("### 2. Analyze")
                    predict_btn = st.button("Analyze Image", key="predict_button")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
        with col2:
            # This content should now be clearly visible due to CSS rules
            st.markdown("### 3. Diagnosis Results")
            if test_image is not None and predict_btn:
                with st.spinner("🧠 Analyzing..."):
                    time.sleep(1)
                    result_index, confidence = model_prediction(model, test_image)
                if result_index is not None and confidence is not None:
                    result_label = CLASS_NAMES[result_index]
                    try:
                        plant_parts = result_label.split("___")
                        plant_name = plant_parts[0].replace("_", " ").strip()
                        condition = plant_parts[1].replace("_", " ").strip()
                    except IndexError:
                        plant_name, condition = "Unknown", result_label
                    is_healthy = "healthy" in condition.lower()
                    status_class = "healthy" if is_healthy else "diseased"
                    # Result Div - text color forced by CSS
                    st.markdown(f"""<div class="disease-result {status_class}"> Condition: {condition} </div>""", unsafe_allow_html=True)
                    # Result Card - text color forced by CSS
                    st.markdown(f"""<div class="card"><h4>Detected Plant: {plant_name}</h4><h4>Diagnosis: {condition}</h4><h4>Confidence: {confidence:.2%}</h4></div>""", unsafe_allow_html=True)
                    if not is_healthy:
                        st.markdown("### Recommended Actions")
                        treatment_key = plant_parts[1].strip() if len(plant_parts)>1 else condition
                        treatment = TREATMENT_RECOMMENDATIONS.get(treatment_key, DEFAULT_TREATMENT_ADVICE)
                        # Treatment Card - text color forced by CSS
                        st.markdown(f"""<div class="card" style="background-color: rgba(255, 235, 238, 0.99);"><p>{treatment}</p><small><i>Disclaimer: Recommendations are informational... Verify with experts.</i></small></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("### Plant Health Status")
                        # Healthy Card - text color forced by CSS
                        st.markdown(f"""<div class="card" style="background-color: rgba(232, 245, 233, 0.99);"><p>{HEALTHY_PLANT_ADVICE}</p></div>""", unsafe_allow_html=True)
                else:
                    st.error("Analysis failed. Please try again.")
            elif not test_image:
                st.info("Upload an image and click 'Analyze Image' to view results.")


# --- ABOUT Page ---
elif app_mode == "ABOUT":
    st.markdown('<h1 class="main-header">ℹ️ About This System</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Our Mission</h2>', unsafe_allow_html=True)
    st.markdown("""<div class="card"><p class="info-text">Empowering users with accessible AI for plant health...</p></div>""", unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">How It Works</h2>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card"><p class="info-text">Utilizes a CNN model trained on leaf images... Recognizes <strong>{len(CLASS_NAMES)} classes</strong>.</p></div>""", unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Supported Plants</h2>', unsafe_allow_html=True)
    plant_list = sorted(list(set([name.split("___")[0].replace("_", " ") for name in CLASS_NAMES])))
    st.markdown('<div class="card">', unsafe_allow_html=True)
    cols = st.columns(3)
    plants_per_col = (len(plant_list) + len(cols) - 1) // len(cols)
    for i, plant in enumerate(plant_list):
        cols[i // plants_per_col].markdown(f"- {plant}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Technology & Data</h2>', unsafe_allow_html=True)
    last_updated_date = time.strftime("%B %Y")
    st.markdown(f"""<div class="card"><p class="info-text"><strong>Model:</strong> CNN<br><strong>Framework:</strong> TensorFlow/Keras<br><strong>Dataset:</strong> PlantVillage (augmented)<br><strong>Input:</strong> 128x128px<br><strong>Accuracy:</strong> ~95%+ (validation)<br><strong>Last Trained:</strong> Approx. {last_updated_date}</p></div>""", unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Tips for Best Results</h2>', unsafe_allow_html=True)
    st.markdown("""<div class="card"><ol class="info-text"><li>Use clear, well-lit images.</li><li>Capture leaf against plain background.</li><li>Ensure symptoms are visible.</li><li>Use individual leaves.</li><li>Avoid shadows/water droplets.</li></ol></div>""", unsafe_allow_html=True)


# --- Footer ---
st.markdown("---")
current_year = time.strftime("%Y")
st.markdown(f"""
<div class="footer">
    <p>© {current_year} Plant Disease Detection System | AI for Sustainable Agriculture</p>
    <p><i>Disclaimer: Preliminary analysis only. Consult local experts.</i></p>
    <p>Contact: support@plantdoctor.ai (Example)</p>
</div>
""", unsafe_allow_html=True)

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

