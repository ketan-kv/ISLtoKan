import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="ISL to Kannada Translator",
    page_icon="🤟",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🤟 Live Translation",
        "📊 Evaluation",
        "⚙️ Settings",
        "ℹ️ About",
    ]
)

# =============================================================================
# HOME PAGE
# =============================================================================

if page == "🏠 Home":

    st.title("🤟 ISL → Kannada Translator")

    st.markdown(
        """
        Welcome to the **ISL → Kannada Translation System**.

        This application will:

        - 📷 Capture live webcam input
        - ✋ Recognize Indian Sign Language gestures
        - 🔤 Translate gestures into Kannada
        - 📈 Display prediction confidence
        - 📊 Evaluate model performance
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project Information")

        st.success("Project Status: Under Development")

        st.write("**Model:** Random Forest")
        st.write("**Input:** MediaPipe Hand Landmarks")
        st.write("**Output:** Kannada Translation")

    with col2:
        st.subheader("Current Features")

        st.checkbox("Application Architecture", value=True, disabled=True)
        st.checkbox("Streamlit Frontend", value=True, disabled=True)
        st.checkbox("Navigation System", value=True, disabled=True)
        st.checkbox("Live Translation", value=False, disabled=True)
        st.checkbox("Evaluation Dashboard", value=False, disabled=True)

# =============================================================================
# LIVE TRANSLATION PAGE
# =============================================================================

elif page == "🤟 Live Translation":

    st.title("🤟 Live Translation")

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:

        st.subheader("📷 Live Camera Feed")

        camera_placeholder = st.empty()

        camera_placeholder.info(
            "Live webcam feed will appear here."
        )

    with right:

        st.subheader("Prediction")

        detected_sign = st.empty()
        kannada_text = st.empty()
        confidence = st.empty()
        fps = st.empty()

        detected_sign.metric(
            "Detected Sign",
            "--"
        )

        kannada_text.metric(
            "Kannada",
            "--"
        )

        confidence.metric(
            "Confidence",
            "0%"
        )

        fps.metric(
            "FPS",
            "0"
        )

    st.markdown("---")

    st.subheader("Prediction History")

    history = st.empty()

    history.info("No predictions yet.")

# =============================================================================
# EVALUATION PAGE
# =============================================================================

elif page == "📊 Evaluation":

    st.title("📊 Evaluation Dashboard")

    st.info("Evaluation dashboard will be implemented here.")

# =============================================================================
# SETTINGS PAGE
# =============================================================================

elif page == "⚙️ Settings":

    st.title("⚙️ Settings")

    st.write("Camera Settings")

    camera_index = st.number_input(
        "Camera Index",
        min_value=0,
        value=0
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.28
    )

    st.success("Settings will be connected to the application later.")

# =============================================================================
# ABOUT PAGE
# =============================================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About")

    st.markdown(
        """
        ## ISL → Kannada Translator

        This project aims to translate **Indian Sign Language (ISL)** into
        **Kannada** using Computer Vision and Machine Learning.

        ### Technology Stack

        - Python
        - OpenCV
        - MediaPipe
        - Scikit-Learn
        - Streamlit

        ### Team

        - AI Model Development
        - Translation Pipeline
        - Software Application & Frontend
        """
    )