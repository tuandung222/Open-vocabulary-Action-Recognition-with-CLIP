# CLIP HAR System Architecture

This document provides a comprehensive overview of the CLIP HAR (Human Action Recognition) project architecture, detailing the various components and their interactions.

## System Overview

The CLIP HAR system is designed as a modular, layered architecture to support the complete lifecycle of developing, training, evaluating, and deploying human action recognition models based on CLIP (Contrastive Language-Image Pre-training).

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIP HAR Architecture                          │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│  Data Layer │ Model Layer │   Training  │ Evaluation  │    MLOps    │ Deployment  │
│             │             │    Layer    │    Layer    │    Layer    │    Layer    │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
       │             │             │             │             │             │
       ▼             ▼             ▼             ▼             ▼             ▼
┌─────────────┐┌─────────────┐┌─────────────┐┌─────────────┐┌─────────────┐┌─────────────┐
│ Dataset     ││ CLIP Model  ││ Distributed ││ Metrics     ││ Experiment  ││ Model       │
│ Loading     ││ Integration ││ Training    ││ Calculation ││ Tracking    ││ Export      │
├─────────────┤├─────────────┤├─────────────┤├─────────────┤├─────────────┤├─────────────┤
│ Preprocessing││ Custom     ││ Training    ││ Confusion   ││ DVC         ││ ONNX        │
│             ││ Layers      ││ Loop        ││ Matrix      ││ Integration ││ Format      │
├─────────────┤├─────────────┤├─────────────┤├─────────────┤├─────────────┤├─────────────┤
│ Augmentation││ Classifier  ││ Model       ││ Evaluation  ││ Model       ││ TensorRT    │
│             ││ Head        ││ Checkpoints ││ Reports     ││ Registry    ││ Format      │
└─────────────┘└─────────────┘└─────────────┘└─────────────┘└─────────────┘└─────────────┘
                                                                                │
                                                                                ▼
                                                                         ┌─────────────┐
                                                                         │ Application │
                                                                         │    Layer    │
                                                                         └──────┬──────┘
                                                                                │
                                                                                ▼
                                                                         ┌─────────────┐
                                                                         │  Streamlit  │
                                                                         │     UI      │
                                                                         ├─────────────┤
                                                                         │  FastAPI    │
                                                                         │  Endpoints  │
                                                                         ├─────────────┤
                                                                         │   Python    │
                                                                         │   Client    │
                                                                         └─────────────┘
```

## Layer Details

### 1. Data Layer

The data layer manages dataset acquisition, preprocessing, and augmentation.

```
┌──────────────────────────────────┐
│         Data Layer Flow          │
└──────────────────────────────────┘
          │
┌─────────▼─────────┐
│   Dataset Loader  │
└─────────┬─────────┘
          │
          │  • load_dataset(dataset_name)
          │  • split_dataset(train_ratio, val_ratio)
          │  • create_data_loaders(batch_size)
          │
┌─────────▼─────────┐
│   Preprocessing   │
└─────────┬─────────┘
          │
          │  • resize_images(size)
          │  • normalize_images(mean, std)
          │  • tokenize_text(text_prompts)
          │
┌─────────▼─────────┐
│   Augmentation    │
└─────────┬─────────┘
          │
          │  • apply_augmentations(image, strength)
          │  • random_crop(), random_flip()
          │  • color_jitter()
          │
          ▼
    To Model Layer
```

Key components:
- **Dataset Loader**: Handles loading and splitting datasets
- **Preprocessing**: Responsible for image resizing, normalization, and text tokenization
- **Augmentation**: Implements various data augmentation techniques

### 2. Model Layer

The model layer defines the neural network architecture.

```
┌──────────────────────────────────┐
│        Model Architecture        │
└──────────────────────────────────┘
                │
        ┌───────▼───────┐
        │   CLIP Model  │
        └───────┬───────┘
                │  • image_encoder
                │  • text_encoder
                │  • extract_features(images)
                │
        ┌───────▼───────┐
        │ Custom Layers │
        └───────┬───────┘
                │  • attention_module
                │  • feature_fusion
                │  • temporal_modeling
                │
        ┌───────▼───────┐
        │Classifier Head│
        └───────┬───────┘
                │  • fc_layers
                │  • dropout
                │  • classification output
                │
                ▼
        To Training Layer
```

Key components:
- **CLIP Model**: Provides pre-trained image and text encoders
- **Custom Layers**: Extends CLIP with task-specific functionality
- **Classifier Head**: Maps features to action classes

### 3. Training Layer

The training layer orchestrates model training processes.

```
┌───────────────────────────────────────────────────────────┐
│                  Training Process Flow                     │
└───────────────────────────────────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │   Initialize  │
                    │    Training   │
                    └───────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
     ┌────────▼────────┐        ┌────────▼────────┐
     │ Single GPU Mode │        │Distributed Mode │
     └────────┬────────┘        └────────┬────────┘
              │                           │
              │                  ┌────────▼────────┐
              │                  │  Setup DDP/FSDP  │
              │                  └────────┬────────┘
              │                           │
              └───────────┬───────────────┘
                          │
                ┌─────────▼─────────┐
                │   Training Loop   │◄────┐
                └─────────┬─────────┘     │
                          │               │
                ┌─────────▼─────────┐     │
                │    Forward Pass   │     │
                └─────────┬─────────┘     │
                          │               │
                ┌─────────▼─────────┐     │
                │ Loss Calculation  │     │
                └─────────┬─────────┘     │
                          │               │
                ┌─────────▼─────────┐     │
                │   Backward Pass   │     │
                └─────────┬─────────┘     │
                          │               │
                ┌─────────▼─────────┐     │
                │  Optimizer Step   │     │
                └─────────┬─────────┘     │
                          │               │
                ┌─────────▼─────────┐     │
                │   End of Epoch?   │     │
                └─────────┬─────────┘     │
                          │               │
                          │ Yes           │ No
                          ▼               │
                ┌─────────────────┐       │
                │   Validation    │       │
                └─────────┬───────┘       │
                          │               │
                ┌─────────▼─────────┐     │
                │  Early Stopping?  │     │
                └─────────┬─────────┘     │
                          │               │
              ┌───────────┴────────┐      │
              │                    │      │
              │ No                 │ Yes  │
              ▼                    ▼      │
        ┌───────────┐       ┌───────────┐ │
        │ Continue  ├───────►    End    │ │
        │ Training  │       │  Training │ │
        └───────────┘       └───────────┘ │
              │                           │
              └───────────────────────────┘
```

Key components:
- **Distributed Training**: Support for DistributedDataParallel (DDP) and FullyShardedDataParallel (FSDP)
- **Training Loop**: Manages iterations, epochs, and validation
- **Optimization**: Implements learning rate scheduling and gradient clipping

### 4. Evaluation Layer

The evaluation layer assesses model performance.

```
┌────────────────────────────────────────────────────┐
│              Evaluation Workflow                    │
└────────────────────────────────────────────────────┘
              │
      ┌───────▼────────┐
      │   Load Model   │
      └───────┬────────┘
              │
      ┌───────▼────────┐
      │Prepare Dataset │
      └───────┬────────┘
              │
      ┌───────▼────────┐
      │ Run Inference  │
      └───────┬────────┘
              │
┌─────────────┼─────────────┐
│             │             │
│    ┌────────▼─────────┐   │
│    │Calculate Metrics │   │
│    └────────┬─────────┘   │
│             │             │
│    ┌────────▼─────────┐   │
│    │  Generate Conf.  │   │
│    │     Matrix       │   │
│    └────────┬─────────┘   │
│             │             │
│    ┌────────▼─────────┐   │
│    │     Create       │   │
│    │  Visualizations  │   │
│    └────────┬─────────┘   │
│             │             │
└─────────────┼─────────────┘
              │
      ┌───────▼────────┐
      │Produce Reports │
      └───────┬────────┘
              │
              ▼
        To MLOps Layer
```

Key components:
- **Metrics Calculation**: Computes accuracy, precision, recall, F1 score
- **Confusion Matrix**: Visualizes class prediction performance
- **Reporting**: Generates comprehensive evaluation reports

### 5. MLOps Layer

The MLOps layer handles experiment tracking and model management.

```
┌──────────────────────────────────────────────────────────┐
│                     MLOps Layer                           │
└──────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
┌────────▼─────────┐┌──────▼───────┐┌────────▼─────────┐
│    Experiment    ││    Version   ││  Model Registry  │
│     Tracking     ││    Control   ││                  │
└────────┬─────────┘└──────┬───────┘└────────┬─────────┘
         │                 │                 │
    ┌────▼────┐        ┌───▼───┐        ┌────▼─────┐
    │ MLflow  │        │  DVC  │        │HuggingFace│
    └────┬────┘        └───┬───┘        │   Hub     │
         │                 │            └────┬─────┘
    ┌────▼────┐            │                 │
    │Weights & │           │            ┌────▼─────┐
    │ Biases   │           │            │  Local   │
    └────┬────┘            │            │ Registry │
         │                 │            └────┬─────┘
         └────────┐   ┌────┘                 │
                  │   │                      │
                  ▼   ▼                      ▼
             ┌─────────────┐           ┌──────────┐
             │  Tracking   │───────────►  Model   │
             │  Database   │           │ Storage  │
             └─────────────┘           └──────────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                     To Deployment Layer
```

Key components:
- **Unified Tracking**: Supports both MLflow and Weights & Biases
- **DVC Integration**: Manages dataset and model versioning
- **HuggingFace Hub**: Facilitates model sharing and distribution

### 6. Deployment Layer

The deployment layer prepares models for production use.

```
┌──────────────────────────────────────────────────────────────┐
│                     Deployment Layer                          │
└──────────────────────────────────────────────────────────────┘
                             │
                     ┌───────▼───────┐
                     │ Trained Model │
                     └───────┬───────┘
                             │
         ┌─────────────────────────────────────┐
         │                                     │
┌────────▼─────────┐   ┌────────────┐   ┌──────▼───────┐
│  Export Formats  │   │   Model    │   │  Inference   │
│                  │   │  Adapters  │   │   Server     │
└────────┬─────────┘   └──────┬─────┘   └──────┬───────┘
         │                    │                │
    ┌────┴───────────┐        │                │
    │                │        │                │
┌───▼────┐      ┌────▼───┐    │          ┌─────▼─────┐
│ PyTorch │      │ ONNX   │    │          │  FastAPI  │
└───┬────┘      └────┬───┘    │          │   Server  │
    │                │        │          └─────┬─────┘
┌───▼────┐      ┌────▼───┐    │                │
│TorchScrip     │TensorRT│    │                │
└───┬────┘      └────┬───┘    │                │
    │                │        │                │
    └────────┬───────┘        │                │
             ▼                │                │
      ┌─────────────┐         │                │
      │  Optimized  │         │                │
      │    Model    ├─────────┘                │
      └──────┬──────┘                          │
             │                                 │
             └─────────────┬──────────────────┘
                           │
                           ▼
                    To Application Layer
```

Key components:
- **Model Export**: Converts models to various formats (ONNX, TensorRT)
- **Inference Server**: FastAPI-based API for model serving
- **Adapters**: Unified interface for different model formats

### 7. Application Layer

The application layer provides user interfaces.

```
┌────────────────────────────────────────────────────┐
│               Application Layer                     │
└────────────────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼───────────┐      ┌──────────▼─────────┐
│    User Interfaces  │      │   API Components   │
└─────────┬───────────┘      └──────────┬─────────┘
          │                             │
    ┌─────┴─────┐                ┌──────┴───────┐
    │           │                │              │
┌───▼───┐  ┌────▼───┐       ┌────▼─────┐  ┌─────▼────┐
│Stream-│  │Jupyter │       │  REST    │  │ Python   │
│lit UI │  │Notebook│       │  API     │  │ Client   │
└───────┘  └────────┘       └──────────┘  └──────────┘
    │           │                │              │
    │           │                └───────┬──────┘
    │           │                        │
    └───────────┼────────────────────────┘
                │
        ┌───────▼─────────┐
        │    End Users    │
        └─────────────────┘
```

Key components:
- **Streamlit App**: Interactive web interface for model testing
- **REST API**: Endpoints for model inference
- **Python Client**: Programmatic access to model functionality

## Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                               Data Flow                                         │
└────────────────────────────────────────────────────────────────────────────────┘

┌─────┐          ┌─────┐          ┌─────┐          ┌─────┐          ┌──────────┐
│User │          │App  │          │API  │          │Model│          │DataStore │
└──┬──┘          └──┬──┘          └──┬──┘          └──┬──┘          └────┬─────┘
   │                │                │                │                   │
   │ Upload Image   │                │                │                   │
   │───────────────>│                │                │                   │
   │                │                │                │                   │
   │                │Request Prediction                │                   │
   │                │───────────────>│                │                   │
   │                │                │                │                   │
   │                │                │ Run Inference  │                   │
   │                │                │───────────────>│                   │
   │                │                │                │                   │
   │                │                │ Return Predictions                 │
   │                │                │<───────────────│                   │
   │                │                │                │                   │
   │                │ Return Results │                │                   │
   │                │<───────────────│                │                   │
   │                │                │                │                   │
   │ Display Results│                │                │                   │
   │<───────────────│                │                │                   │
   │                │                │                │                   │
   │                │                │ Record Prediction (Optional)       │
   │                │                │───────────────────────────────────>│
   │                │                │                │                   │

```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Deployment Architecture                          │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────────┐    ┌────────────────────────┐  ┌──────────────────────┐
│ Development Environment│    │ Training Infrastructure │  │Deployment Infrastructure│
└───────────┬────────────┘    └───────────┬────────────┘  └───────────┬──────────┘
            │                             │                           │
    ┌───────▼────────┐          ┌─────────▼───────┐         ┌────────▼─────────┐
    │  Development   │          │    Training     │         │    Inference     │
    │    Machine     │          │    Container    │         │    Container     │
    └───────┬────────┘          └─────────┬───────┘         └─────────┬────────┘
            │                             │                           │
    ┌───────▼────────┐          ┌─────────▼───────┐      ┌────────────┴───────────┐
    │Git Repository  │          │    GPU Cluster   │      │                        │
    └───────┬────────┘          └─────────────────┘      │                        │
            │                                          ┌──▼───────┐      ┌────────▼─┐
    ┌───────▼────────┐                                 │API Server│      │Streamlit │
    │CI/CD Pipeline  │◄────────────────────────────────┘          │      │  Server  │
    └───────┬────────┘                                 └──────────┘      └──────────┘
            │
            │                   ┌─────────────────────┐
            └──────────────────►│      Storage        │
                                └──────────┬──────────┘
                                           │
                                ┌──────────┴──────────┐
                                │                     │
                          ┌─────▼────┐         ┌──────▼─────┐
                          │   DVC    │         │   Model    │
                          │  Storage │         │  Registry  │
                          └──────────┘         └────────────┘
```

## Module Dependencies

```
┌──────────────────────────────────────────────────┐
│              Module Dependencies                  │
└──────────────────────────────────────────────────┘

┌─────────┐     ┌─────────┐     ┌───────────┐
│   app   │────►│  mlops  │────►│ evaluation│
└────┬────┘     └────┬────┘     └─────┬─────┘
     │               │                │
     │               │                │
     │          ┌────▼────┐      ┌────▼────┐
     └─────────►│ models  │◄─────┤training │
                └────┬────┘      └────┬────┘
                     │                │
                     │                │
                     ▼                │
                ┌────────┐            │
                │  data  │◄───────────┘
                └────────┘

```

## Scalability and Performance

The system is designed for scalability and performance:

1. **Distributed Training**: Support for multi-GPU and multi-node training
2. **Model Optimization**: Export to optimized formats for inference 
3. **Containerization**: Docker containers for consistent deployment
4. **API Design**: Asynchronous endpoint processing for high throughput

## Security Considerations

The system implements several security measures:

1. **Authentication**: API access control
2. **Data Privacy**: Input data sanitization
3. **Model Protection**: Access control for model artifacts
4. **Container Security**: Minimal base images and dependency scanning
