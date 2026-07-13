import os

import altair as alt
import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2
from PIL import Image


@st.cache_resource
def get_backend_url() -> str:
    """
    Retrieve the backend service URL from Google Cloud Run or environment variables.
    Returns:
        str: The backend service URL.
    """
    parent = "projects/mlopsdogclassification/locations/europe-west4"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "dogs-bentoml":
            return service.uri
    return st.secrets.get("BACKEND_URL", os.environ.get("BACKEND_URL", ""))


def main() -> None:
    """
    Function to run the Streamlit frontend for the Dog Breed Classifier application.
    This function sets up the Streamlit interface, handles image uploads and camera input,
    sends the image to the backend for classification, and displays the results.
    """
    st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="wide")
    st.title("🐶 Dog Breed Classifier")

    BACKEND_URL = get_backend_url()
    if not BACKEND_URL:
        st.error("Backend service not found. Please check your configuration.")
        st.stop()

    if "camera_open" not in st.session_state:
        st.session_state["camera_open"] = False
    if "image_hash" not in st.session_state:
        st.session_state["image_hash"] = None

    col_input, col_preview = st.columns(2)

    with col_input:
        tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
            )

        with tab_camera:
            label = "Close Camera" if st.session_state["camera_open"] else "Open Camera"
            if st.button(label, use_container_width=True):
                st.session_state["camera_open"] = not st.session_state["camera_open"]
                st.rerun()

            camera_photo = None
            if st.session_state["camera_open"]:
                camera_photo = st.camera_input("Take a photo", label_visibility="collapsed")

        image_source = uploaded_file or camera_photo

        # disclaimer that images will be saved
        st.info("⚠️ Disclaimer: Uploaded images will be saved for monitoring. ")

        # Clear stale prediction when a new image is loaded
        if image_source is not None:
            current_hash = hash(image_source.getvalue())
            if current_hash != st.session_state["image_hash"]:
                st.session_state["image_hash"] = current_hash
                st.session_state.pop("prediction", None)

        if "prediction" in st.session_state:
            response = st.session_state["prediction"]["predictions"]
            top_breed = response["class1"].replace("_", " ").title()
            top_conf = response["confidence1"]
            st.success(f"🥇 **Top Prediction: {top_breed}** &nbsp;&nbsp; {top_conf:.1%}")

            breeds = [response[f"class{i}"].replace("_", " ").title() for i in range(1, 6)]
            confidences = [response[f"confidence{i}"] for i in range(1, 6)]

            chart_df = pd.DataFrame(
                {
                    "Breed": breeds,
                    "Confidence (%)": [c * 100 for c in confidences],
                    "is_top": [i == 0 for i in range(len(breeds))],
                }
            )
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Confidence (%):Q",
                        scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(format=".0f", tickCount=10),
                    ),
                    y=alt.Y("Breed:N", sort="-x"),
                    color=alt.Color(
                        "is_top:N",
                        scale=alt.Scale(domain=[False, True], range=["#0d6e30", "#21c354"]),
                        legend=None,
                    ),
                )
            )
            st.altair_chart(chart, use_container_width=True)

    with col_preview:
        if image_source is not None:
            if st.button("Classify", type="primary", use_container_width=True):
                image_source.seek(0)
                try:
                    with st.spinner("Classifying..."):
                        response = requests.post(
                            f"{BACKEND_URL}/predict", files={"image": image_source}, timeout=180
                        ).json()
                    st.session_state["prediction"] = response
                    st.rerun()
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The service may be starting up — try again in a moment.")
                except Exception as e:
                    st.error(f"Classification failed: {e}")

            image = Image.open(image_source)
            st.image(image, use_container_width=True)
        else:
            st.info("Your image will appear here.")


if __name__ == "__main__":
    main()
