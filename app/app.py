import os
import sys
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizerFast

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from CLIP_HAR_PROJECT.data.preprocessing import get_class_mappings

# Import project modules
from CLIP_HAR_PROJECT.models.clip_model import CLIPLabelRetriever

# Set page configuration
st.set_page_config(
    page_title="HAR Classification",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_model(model_path, model_name, class_names, prompt_template):
    """Load a model from a checkpoint."""
    # Load CLIP model
    model = CLIPLabelRetriever.from_pretrained(
        model_name, labels=class_names, prompt_template=prompt_template
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    return model


@st.cache_resource
def get_model_and_processors(model_path, model_name, class_names, prompt_template):
    """Get model and processors with caching."""
    # Load tokenizer and image processor
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
    image_processor = CLIPImageProcessor.from_pretrained(model_name)

    # Load model
    model = load_model(model_path, model_name, class_names, prompt_template)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, tokenizer, image_processor, device


def process_image(image, image_processor, model, class_names, device):
    """Process an image and get predictions."""
    # Preprocess image
    if isinstance(image, np.ndarray):
        # Convert from OpenCV format to PIL
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Process image
    inputs = image_processor(image, return_tensors="pt").to(device)

    # Get predictions
    with torch.no_grad():
        _, pred_labels, scores = model.predict(inputs["pixel_values"])

    # Get top predictions
    pred_label = pred_labels[0]
    top_scores = torch.from_numpy(scores).topk(5, dim=1)
    top_idxs = top_scores.indices[0].cpu().numpy()
    top_values = top_scores.values[0].cpu().numpy()
    top_classes = [class_names[idx] for idx in top_idxs]

    return {
        "pred_label": pred_label,
        "top_classes": top_classes,
        "top_scores": top_values,
        "all_scores": scores[0],
    }


def run_webcam_prediction(image_processor, model, class_names, device):
    """Run predictions on webcam feed."""
    # Initialize webcam
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open webcam. Please check your webcam connection.")
            return
    except Exception as e:
        st.error(f"Error initializing webcam: {str(e)}")
        return

    # Create placeholders
    webcam_placeholder = st.empty()
    prediction_placeholder = st.empty()
    
    try:
        # Run webcam loop until stop button is pressed
        while st.session_state.webcam_running:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam")
                st.session_state.webcam_running = False
                break

            # Process frame
            try:
                results = process_image(frame, image_processor, model, class_names, device)
                
                # Display frame with prediction
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                webcam_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                # Display prediction
                with prediction_placeholder.container():
                    cols = st.columns(5)
                    for i, (cls, score) in enumerate(
                        zip(results["top_classes"], results["top_scores"])
                    ):
                        with cols[i]:
                            st.metric(label=cls, value=f"{score:.2f}")
            except Exception as e:
                st.error(f"Error processing frame: {str(e)}")
                continue

            # Add a small delay
            cv2.waitKey(10)
        
    finally:
        # Release webcam
        cap.release()
        st.info("Webcam stopped")


def plot_confidence_chart(class_names, scores):
    """Plot a bar chart of confidence scores."""
    fig = px.bar(
        x=class_names,
        y=scores,
        labels={"x": "Action", "y": "Confidence Score"},
        title="Confidence Scores for Each Class",
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def main():
    """Main function for the Streamlit app."""
    st.title("Human Action Recognition")
    st.write("Classify human actions in images using CLIP")

    # Initialize session state for webcam control
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False

    # Sidebar
    st.sidebar.title("Settings")

    # Model settings
    st.sidebar.header("Model Settings")
    model_name = st.sidebar.text_input(
        "CLIP Model Name", value="openai/clip-vit-base-patch16"
    )

    model_path = st.sidebar.text_input("Model Checkpoint Path", value="")

    prompt_template = st.sidebar.text_input(
        "Prompt Template", value="a photo of person/people who is/are {label}"
    )

    # Get class names
    _, _, class_names = get_class_mappings(None)

    # Check if model path exists
    if model_path and not os.path.exists(model_path):
        st.sidebar.error(f"Model checkpoint not found: {model_path}")
        model_path = ""

    # Load model if path provided
    model = None
    if model_path:
        try:
            model, tokenizer, image_processor, device = get_model_and_processors(
                model_path, model_name, class_names, prompt_template
            )
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            model = None

    # Main content
    tab1, tab2, tab3 = st.tabs(["Image Upload", "Webcam", "About"])

    # Tab 1: Image Upload
    with tab1:
        st.header("Upload an Image")

        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Process image if model is loaded
            if model:
                with st.spinner("Analyzing image..."):
                    results = process_image(
                        image, image_processor, model, class_names, device
                    )

                # Display results
                st.subheader("Prediction Results")
                st.write(f"**Predicted Action:** {results['pred_label']}")

                # Display top predictions
                col1, col2 = st.columns([2, 3])

                with col1:
                    st.write("**Top Predictions:**")
                    for cls, score in zip(
                        results["top_classes"], results["top_scores"]
                    ):
                        st.write(f"- {cls}: {score:.4f}")

                with col2:
                    fig = plot_confidence_chart(class_names, results["all_scores"])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Please provide a valid model checkpoint path in the sidebar."
                )

    # Tab 2: Webcam
    with tab2:
        st.header("Webcam Prediction")

        if model:
            # Webcam controls
            col1, col2 = st.columns(2)
            
            with col1:
                start_button = st.button(
                    "Start Webcam", 
                    disabled=st.session_state.webcam_running,
                    use_container_width=True
                )
                
            with col2:
                stop_button = st.button(
                    "Stop Webcam", 
                    disabled=not st.session_state.webcam_running,
                    use_container_width=True
                )
            
            # Handle button clicks
            if start_button:
                st.session_state.webcam_running = True
                st.info("Starting webcam...")
                run_webcam_prediction(image_processor, model, class_names, device)
                
            if stop_button:
                st.session_state.webcam_running = False
                st.info("Stopping webcam...")
                
            # Display webcam status
            st.markdown(f"**Webcam Status**: {'Running' if st.session_state.webcam_running else 'Stopped'}")
            
            # Instructions
            with st.expander("Webcam Instructions", expanded=False):
                st.markdown("""
                1. Click 'Start Webcam' to begin real-time action recognition
                2. Position yourself in the camera frame
                3. Perform different actions to see the predictions
                4. Click 'Stop Webcam' when finished
                
                Note: If the webcam doesn't start, please check your camera permissions.
                """)
        else:
            st.warning("Please provide a valid model checkpoint path in the sidebar.")

    # Tab 3: About
    with tab3:
        st.header("About Human Action Recognition")

        st.write(
            """
        This app uses a CLIP-based model to classify human actions in images.
        The model has been fine-tuned on the Human Action Recognition (HAR) dataset.

        ## Available Actions
        The model can recognize the following actions:
        """
        )

        # Display class names in a grid
        cols = st.columns(3)
        for i, cls in enumerate(class_names):
            cols[i % 3].write(f"- {cls}")

        st.write(
            """
        ## Model Details
        The base model is CLIP (Contrastive Language-Image Pre-training) by OpenAI.
        CLIP is trained to understand images in relation to text descriptions, allowing
        it to perform zero-shot classification of images.

        In this project, we fine-tune CLIP to specifically recognize human actions by
        using text prompts like "a photo of person/people who is/are {action}".
        """
        )


if __name__ == "__main__":
    main()
