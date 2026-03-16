"""
Streamlit UI for Dog vs Cat Classifier.
Run from project root: streamlit run ui/streamlit_app.py
"""
import base64
import io
import logging
import os

import numpy as np
import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Dog vs Cat Classifier",
    layout="wide",
)

logger = logging.getLogger(__name__)

# Deployed API base URL (no trailing slash). All requests use this + path (/predict, /explainability, /health, etc.).
API_BASE_URL = "https://dogs-cats-classifier.onrender.com"
DEFAULT_API_URL = os.environ.get("API_URL", API_BASE_URL)
API_TIMEOUT_SECONDS = 30

st.markdown("""
<style>
    .main { padding: 20px; }
    .header { text-align: center; margin-bottom: 30px; }
    .result-success {
        background-color: #3d3d3d; color: #e8e8e8; padding: 20px;
        border-radius: 8px; margin: 10px 0; border: 1px solid #555;
    }
    .result-unknown {
        background-color: #3d3d3d; color: #e8e8e8; padding: 20px;
        border-radius: 8px; margin: 10px 0; border: 1px solid #555;
    }
    .result-error {
        background-color: #3d3d3d; color: #f0a0a0; padding: 20px;
        border-radius: 8px; margin: 10px 0; border: 1px solid #555;
    }
    .backend-warning {
        background-color: #fff3cd; border: 1px solid #ffc107;
        padding: 16px; border-radius: 8px; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


def check_api_connection(base_url, timeout=5):
    """Check FastAPI backend connectivity. Returns (success, message)."""
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        response = requests.get(health_url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("status", "ok")
        return False, f"Health check returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused. The backend API server is not running."
    except requests.exceptions.Timeout:
        return False, "Request timed out."
    except requests.exceptions.RequestException as e:
        return False, str(e)


def encode_image_to_base64(image):
    """Encode PIL Image to RGB and then to base64 PNG string."""
    if image is None:
        raise ValueError("image is None")
    if hasattr(image, "convert"):
        img = image.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(image)).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii").strip()


def predict_image(api_base_url, image, timeout=API_TIMEOUT_SECONDS):
    """POST /predict. Returns response dict or None."""
    predict_url = f"{api_base_url.rstrip('/')}/predict"
    try:
        image_b64 = encode_image_to_base64(image)
        response = requests.post(
            predict_url,
            json={"image_base64": image_b64},
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.json()
        st.error("API Error: %s - %s" % (response.status_code, response.text[:200]))
        return None
    except requests.exceptions.ConnectionError:
        st.error("The backend API server is not running. Please start FastAPI first.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
        return None
    except Exception as e:
        st.error("Error: %s" % e)
        return None


def get_gradcam(api_base_url, image, timeout=API_TIMEOUT_SECONDS):
    """POST /explainability. Returns result dict or None."""
    url = f"{api_base_url.rstrip('/')}/explainability"
    try:
        try:
            image_b64 = encode_image_to_base64(image)
        except Exception as enc_err:
            st.error("Image encoding failed: %s" % enc_err)
            return None
        if not image_b64:
            st.error("Image encoding result is empty.")
            return None
        response = requests.post(
            url,
            json={"image_base64": image_b64},
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.json()
        try:
            err = response.json()
            msg = err.get("detail", str(err))
        except Exception:
            msg = response.text or f"HTTP {response.status_code}"
        st.error("Grad-CAM request failed: %s" % msg)
        return None
    except requests.exceptions.ConnectionError:
        st.error("The backend API server is not running. Please start FastAPI first.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
        return None
    except Exception as e:
        st.error("Grad-CAM error: %s" % e)
        return None


def fetch_json(api_base_url, path, timeout=10):
    """GET request returning JSON or None."""
    url = f"{api_base_url.rstrip('/')}{path}"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        return None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException):
        return None


def main():
    with st.sidebar:
        st.header("API Settings")
        api_url = st.text_input(
            "Backend API URL",
            value=DEFAULT_API_URL,
            help="FastAPI server base URL (e.g. https://dogs-cats-classifier.onrender.com)",
        )
        st.caption("Changes apply after switching tabs.")

    api_ok, api_message = check_api_connection(api_url)
    if not api_ok:
        st.markdown("""
        <div class="backend-warning">
            <strong>The backend API server is not running.</strong><br/>
            Please start FastAPI first: <code>uvicorn app.main:app --host 0.0.0.0 --port 8000</code>
        </div>
        """, unsafe_allow_html=True)
        st.error(api_message)
        st.info("Current API URL: " + api_url)

    st.markdown("""
    <div class="header">
        <h1>Dog vs Cat Classifier</h1>
        <p>Production-ready AI-powered image classification</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Classify", "Explain", "Performance", "Info"])

    with tab1:
        st.header("Image Classification")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Upload")
            uploaded_file = st.file_uploader(
                "Choose image",
                type=["jpg", "jpeg", "png"],
                help="Upload jpg, jpeg, or png",
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded image", use_container_width=True)
                if st.button("Classify", key="classify"):
                    if not api_ok:
                        st.error("The backend API server is not running. Please start FastAPI first.")
                    else:
                        with st.spinner("Classifying..."):
                            result = predict_image(api_url, image)
                            if result:
                                st.session_state.last_result = result
                                st.session_state.last_image = image
                                st.session_state.pop("gradcam_result", None)
        with col2:
            st.subheader("Result")
            if "last_result" in st.session_state:
                result = st.session_state.last_result
                if result.get("status") == "success":
                    st.markdown(f"""
                    <div class="result-success">
                        <h3>Success</h3>
                        <p><b>Label:</b> {result["label"].upper()}</p>
                        <p><b>Confidence:</b> {result["confidence"]:.2%}</p>
                        <p><b>Time:</b> {result.get("processing_time_ms", 0):.2f} ms</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif result.get("label") == "unknown" or result.get("status") == "unknown_detected":
                    msg = result.get("error_message") or result.get("message") or "This image does not appear to contain a dog or cat."
                    st.markdown(f"""
                    <div class="result-unknown">
                        <h3>Unknown</h3>
                        <p>This image does not appear to contain a dog or cat.</p>
                        <p><b>Message:</b> {msg}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-error">
                        <h3>Validation failed</h3>
                        <p><b>Reason:</b> {result.get("error_message", "Unknown error")}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Upload an image and click Classify.")

    with tab2:
        st.header("Explainability (Grad-CAM)")
        if not api_ok:
            st.warning("The backend API server is not running. Please start FastAPI first.")
        elif "last_result" in st.session_state and st.session_state.last_result.get("label") == "unknown":
            st.warning("Grad-CAM is not available for unknown predictions. Upload a dog or cat image.")
        elif "last_image" in st.session_state:
            if st.button("Generate Grad-CAM"):
                if not api_ok:
                    st.error("The backend API server is not running. Please start FastAPI first.")
                else:
                    with st.spinner("Generating Grad-CAM..."):
                        gradcam_result = get_gradcam(api_url, st.session_state.last_image)
                        if gradcam_result:
                            st.session_state.gradcam_result = gradcam_result
                        else:
                            st.session_state.pop("gradcam_result", None)
            if st.session_state.get("gradcam_result"):
                gradcam_result = st.session_state.gradcam_result
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Original")
                    orig_b64 = gradcam_result.get("original_image_base64")
                    if orig_b64:
                        st.image(Image.open(io.BytesIO(base64.b64decode(orig_b64))), use_container_width=True)
                    else:
                        st.image(st.session_state.last_image, use_container_width=True)
                with col2:
                    st.subheader("Heatmap")
                    heat_b64 = gradcam_result.get("gradcam_image_base64")
                    if heat_b64:
                        st.image(Image.open(io.BytesIO(base64.b64decode(heat_b64))), use_container_width=True)
                    else:
                        st.info("Heatmap not available.")
                with col3:
                    st.subheader("Overlay")
                    over_b64 = gradcam_result.get("overlay_image_base64")
                    if over_b64:
                        st.image(Image.open(io.BytesIO(base64.b64decode(over_b64))), use_container_width=True)
                    else:
                        st.info("Overlay not available.")
                st.success("Prediction: %s | Confidence: %.2f%%" % (gradcam_result.get("label", "").upper(), gradcam_result.get("confidence", 0) * 100))
        else:
            st.info("Classify an image first in the Classify tab.")

    with tab3:
        st.header("Model Performance")
        if not api_ok:
            st.warning("The backend API server is not running. Please start FastAPI first.")
        else:
            info = fetch_json(api_url, "/model-info")
            if info:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", "%.2f%%" % (info["accuracy"] * 100))
                col2.metric("Precision", "%.2f%%" % (info["precision"] * 100))
                col3.metric("Recall", "%.2f%%" % (info["recall"] * 100))
                col4.metric("F1", "%.2f%%" % (info["f1_score"] * 100))
                st.divider()
                col1, col2 = st.columns(2)
                col1.write("**Backbone:** %s" % info["backbone"])
                col2.write("**Classes:** %s" % info["num_classes"])
                col1.write("**Parameters:** %s" % f"{info.get('parameters', 0):,}")
            else:
                st.error("Could not load model info. Ensure the backend is running.")

    with tab4:
        st.header("API Info")
        if not api_ok:
            st.warning("The backend API server is not running. Please start FastAPI first.")
        else:
            health = fetch_json(api_url, "/health")
            if health:
                st.write("**Status:** %s" % health["status"])
                st.write("**Version:** %s" % health["version"])
                st.write("**Device:** %s" % health["device"])
                st.write("**Model loaded:** %s" % ("Yes" if health.get("model_loaded") else "No"))
            else:
                st.error("API connection failed.")
        st.divider()
        st.subheader("Usage")
        st.markdown("""
        1. Start API: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
        2. Classify tab: Upload image and click Classify
        3. Explain tab: Generate Grad-CAM for dog/cat predictions
        4. Performance tab: View model metrics
        """)
        st.caption("Current API URL: " + api_url)


if __name__ == "__main__":
    main()
