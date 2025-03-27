# Streamlit App Tutorial

## What is Streamlit?

Streamlit is an open-source Python library that makes it easy to create custom web apps for machine learning and data science projects. With just a few lines of Python code, you can build interactive applications with:

- Data visualizations
- Model inference interfaces
- Interactive controls
- Custom layouts and designs

No HTML, CSS, or JavaScript knowledge required!

## Why Streamlit for ML Projects?

Streamlit is particularly well-suited for machine learning projects because:

1. **Python-first**: Written entirely in Python, making it accessible to data scientists
2. **Fast prototyping**: Build functional apps in minutes or hours, not days
3. **Interactive elements**: Add sliders, buttons, and other controls easily
4. **Live reloading**: See changes instantly during development
5. **ML library integration**: Works well with PyTorch, TensorFlow, scikit-learn, etc.
6. **Deployment options**: Deploy locally, in Docker, or on cloud platforms

## Streamlit Basics

### Core Concepts

Streamlit scripts run from top to bottom, redrawing the UI whenever:
- The user interacts with a widget
- The script is modified and saved
- A scheduled rerun occurs

### Common Elements

```python
import streamlit as st

# Text and Markdown
st.title("My App")
st.header("A Section")
st.subheader("A Subsection")
st.text("Simple text")
st.markdown("**Bold** and *italic*")

# Data display
st.dataframe(my_dataframe)
st.table(static_table)
st.json(my_json)

# Media
st.image("my_image.jpg")
st.video("my_video.mp4")

# Interactive widgets
name = st.text_input("Enter your name")
age = st.slider("Select your age", 0, 100, 25)
button = st.button("Click me")

if button:
    st.write(f"Hello {name}, you are {age} years old")

# Sidebar
with st.sidebar:
    st.header("Sidebar")
    option = st.selectbox("Choose an option", ["A", "B", "C"])
```

## Streamlit in Our CLIP HAR Project

Our project uses Streamlit to create an interactive interface for the Human Action Recognition model:

### Key Features of Our App

1. **Video Input Options**:
   - Upload video files
   - Use webcam for real-time recognition
   - Provide a YouTube URL

2. **Visualization Components**:
   - Display video with action predictions
   - Show confidence scores for top predictions
   - Visualize prediction history over time

3. **Model Options**:
   - Select different trained models
   - Choose inference backends (PyTorch, ONNX, TensorRT)
   - Adjust confidence thresholds

## How Our App Works

### App Structure

The main app is structured into several components:

```
app/
├── app.py            # Main Streamlit app
├── components/       # Reusable UI components
│   ├── sidebar.py    # Sidebar with model options
│   ├── video.py      # Video display component
│   └── results.py    # Results visualization
├── utils/            # Utility functions
│   ├── inference.py  # Model inference helpers
│   ├── processing.py # Video processing functions
│   └── visuals.py    # Visualization helpers
└── static/           # Static assets
```

### Session State Management

Streamlit's session state helps manage app state between reruns:

```python
# Initialize state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

if 'model' not in st.session_state:
    st.session_state.model = load_default_model()

# Update state
if process_button:
    st.session_state.predictions = run_inference(video, st.session_state.model)
```

### Webcam Integration

Our app uses Streamlit's webcam component with custom processing:

```python
# Webcam capture
if use_webcam:
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        # Process the captured image
        image = Image.open(img_file_buffer)
        predictions = run_inference(image, model)
        
        # Display results
        display_predictions(predictions)
```

### Handling Video Files

Processing uploaded videos:

```python
# Video upload
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(video_file.read())
    
    # Process video
    video_path = temp_file.name
    predictions = process_video(video_path, model)
    
    # Display video with predictions
    show_annotated_video(video_path, predictions)
```

## Key Streamlit Features We Use

### 1. Callbacks

For more complex interactions:

```python
def on_model_change():
    st.session_state.model = load_model(st.session_state.model_name)

model_name = st.selectbox("Select model", 
                          models_list, 
                          on_change=on_model_change,
                          key="model_name")
```

### 2. Layout Options

For organizing content:

```python
col1, col2 = st.columns(2)

with col1:
    st.header("Video Input")
    # Video input widgets

with col2:
    st.header("Predictions")
    # Prediction display
```

### 3. Caching

For performance optimization:

```python
@st.cache_resource
def load_model(model_path):
    """Load model - this will only run once per model path"""
    return CLIPHARModel.from_pretrained(model_path)
```

## Extending the App

### Adding a New Feature

Want to add a new feature to display action statistics? Here's how:

1. Create a new component file (e.g., `components/statistics.py`):

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_action_statistics(predictions):
    """Display statistics about detected actions"""
    if not predictions:
        st.warning("No predictions available for statistics")
        return
    
    # Count actions
    actions = [p['label'] for p in predictions]
    action_counts = pd.Series(actions).value_counts()
    
    # Create chart
    fig, ax = plt.subplots()
    action_counts.plot(kind='bar', ax=ax)
    ax.set_title("Action Frequency")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    
    st.pyplot(fig)
```

2. Import and use it in `app.py`:

```python
from components.statistics import show_action_statistics

# After processing video
if st.session_state.predictions:
    st.header("Action Statistics")
    show_action_statistics(st.session_state.predictions)
```

### Adding a New Model Selection Option

To add a new model format option:

```python
model_format = st.selectbox(
    "Model Format",
    ["PyTorch", "ONNX", "TensorRT", "New Format"],
    key="model_format"
)

if model_format == "New Format":
    # Logic for handling the new format
    model = load_custom_format_model(model_path)
else:
    # Existing logic
    model = load_model(model_path, model_format.lower())
```

## Deploying the Streamlit App

Our Streamlit app can be deployed in several ways:

### 1. Direct Run

```bash
streamlit run app/app.py
```

### 2. Via Docker (as in our project)

```bash
docker-compose up clip-har-app
```

### 3. Cloud Deployment

Our app can be deployed to Streamlit Cloud, Heroku, or other platforms.

## Troubleshooting Common Issues

### App Crashes with Large Videos

Solution: Use chunked processing
```python
def process_large_video(video_path, chunk_size=30):
    """Process video in chunks to avoid memory issues"""
    # Implementation details
```

### Slow Performance

Solutions:
- Use `@st.cache_data` and `@st.cache_resource` decorators
- Optimize model inference with batch processing
- Consider reducing video resolution for processing

### Webcam Not Working

Solutions:
- Check browser permissions
- Ensure webcam is properly initialized with session state
- Add proper error handling for webcam failures

## Further Learning

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Components Gallery](https://streamlit.io/components)
- [Streamlit for Data Scientists](https://www.streamlit.io/for/data-scientists)
- [Deploying Streamlit Apps](https://docs.streamlit.io/knowledge-base/deploy)

In the next steps, try extending our app with new visualizations or additional input options! 