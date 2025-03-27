# CLIP HAR System Architecture

This document provides a comprehensive overview of the CLIP HAR (Human Action Recognition) project architecture, detailing the various components and their interactions.

## System Overview

The CLIP HAR system is designed as a modular, layered architecture to support the complete lifecycle of developing, training, evaluating, and deploying human action recognition models based on CLIP (Contrastive Language-Image Pre-training).

```mermaid
graph TD
    subgraph "Data Layer"
        D[Dataset Loader] --> P[Preprocessing]
        P --> A[Augmentation]
    end
    
    subgraph "Model Layer"
        CLIP[CLIP Model] --> Custom[Custom Layers]
        Custom --> Classifier[Classifier Head]
    end
    
    subgraph "Training Layer"
        DDP[DistributedDataParallel] --> FSDP[FullyShardedDataParallel]
        FSDP --> Training[Training Loop]
    end
    
    subgraph "Evaluation Layer"
        Metrics[Metrics Calculation] --> ConfMatrix[Confusion Matrix]
        ConfMatrix --> Reporting[Evaluation Reports]
    end
    
    subgraph "MLOps Layer"
        MLflow[MLflow] --> W[Weights & Biases]
        W --> DVC[Data Version Control]
        DVC --> HF[HuggingFace Hub]
    end
    
    subgraph "Deployment Layer"
        Export[Model Export] --> ONNX[ONNX Format]
        Export --> TRT[TensorRT]
        ONNX --> API[FastAPI]
        TRT --> API
    end
    
    subgraph "Application Layer"
        API --> Streamlit[Streamlit UI]
        API --> Client[Python Client]
    end
    
    A --> CLIP
    Classifier --> DDP
    Training --> Metrics
    Reporting --> MLflow
    HF --> Export
```

## Layer Details

### 1. Data Layer

The data layer manages dataset acquisition, preprocessing, and augmentation.

```mermaid
classDiagram
    class DatasetLoader {
        +load_dataset(dataset_name)
        +split_dataset(train_ratio, val_ratio)
        +create_data_loaders(batch_size)
    }
    
    class Preprocessing {
        +resize_images(size)
        +normalize_images(mean, std)
        +tokenize_text(text_prompts)
    }
    
    class Augmentation {
        +apply_augmentations(image, strength)
        +random_crop()
        +random_flip()
        +color_jitter()
    }
    
    DatasetLoader --> Preprocessing
    Preprocessing --> Augmentation
```

Key components:
- **Dataset Loader**: Handles loading and splitting datasets
- **Preprocessing**: Responsible for image resizing, normalization, and text tokenization
- **Augmentation**: Implements various data augmentation techniques

### 2. Model Layer

The model layer defines the neural network architecture.

```mermaid
classDiagram
    class CLIPModel {
        +image_encoder
        +text_encoder
        +forward(images, text)
        +extract_features(images)
    }
    
    class CustomLayers {
        +attention_module
        +feature_fusion
        +forward(clip_features)
    }
    
    class ClassifierHead {
        +fc_layers
        +dropout
        +forward(features)
    }
    
    CLIPModel --> CustomLayers
    CustomLayers --> ClassifierHead
```

Key components:
- **CLIP Model**: Provides pre-trained image and text encoders
- **Custom Layers**: Extends CLIP with task-specific functionality
- **Classifier Head**: Maps features to action classes

### 3. Training Layer

The training layer orchestrates model training processes.

```mermaid
flowchart TD
    A[Initialize Training] --> B{Distributed?}
    B -->|Yes| C[Setup Distributed]
    B -->|No| D[Single GPU]
    C --> E[Initialize DDP/FSDP]
    D --> F[Training Loop]
    E --> F
    F --> G[Forward Pass]
    G --> H[Loss Calculation]
    H --> I[Backward Pass]
    I --> J[Optimizer Step]
    J --> K{End of Epoch?}
    K -->|No| G
    K -->|Yes| L[Validation]
    L --> M{Early Stopping?}
    M -->|Yes| N[End Training]
    M -->|No| F
```

Key components:
- **Distributed Training**: Support for DistributedDataParallel (DDP) and FullyShardedDataParallel (FSDP)
- **Training Loop**: Manages iterations, epochs, and validation
- **Optimization**: Implements learning rate scheduling and gradient clipping

### 4. Evaluation Layer

The evaluation layer assesses model performance.

```mermaid
flowchart LR
    A[Load Model] --> B[Prepare Dataset]
    B --> C[Run Inference]
    C --> D[Calculate Metrics]
    D --> E[Generate Confusion Matrix]
    E --> F[Create Visualizations]
    F --> G[Produce Reports]
```

Key components:
- **Metrics Calculation**: Computes accuracy, precision, recall, F1 score
- **Confusion Matrix**: Visualizes class prediction performance
- **Reporting**: Generates comprehensive evaluation reports

### 5. MLOps Layer

The MLOps layer handles experiment tracking and model management.

```mermaid
graph TD
    subgraph "Experiment Tracking"
        MLflow[MLflow] --- Wandb[Weights & Biases]
        MLflow --- Multi[Multi Tracker]
        Wandb --- Multi
    end
    
    subgraph "Version Control"
        DVC[Data Version Control] --- Git[Git]
    end
    
    subgraph "Model Registry"
        HF[HuggingFace Hub] --- Local[Local Registry]
    end
    
    Multi --> DVC
    DVC --> HF
```

Key components:
- **Unified Tracking**: Supports both MLflow and Weights & Biases
- **DVC Integration**: Manages dataset and model versioning
- **HuggingFace Hub**: Facilitates model sharing and distribution

### 6. Deployment Layer

The deployment layer prepares models for production use.

```mermaid
flowchart TD
    A[Trained Model] --> B{Export Format}
    B --> C[PyTorch]
    B --> D[ONNX]
    B --> E[TorchScript]
    B --> F[TensorRT]
    
    C --> G[Model Adapter]
    D --> G
    E --> G
    F --> G
    
    G --> H[FastAPI Server]
    H --> I[REST API]
    
    I --> J[Python Client]
    I --> K[Web Application]
    I --> L[Mobile Application]
```

Key components:
- **Model Export**: Converts models to various formats (ONNX, TensorRT)
- **Inference Server**: FastAPI-based API for model serving
- **Adapters**: Unified interface for different model formats

### 7. Application Layer

The application layer provides user interfaces.

```mermaid
graph LR
    subgraph "Interfaces"
        A[Streamlit App] --- B[REST API]
        B --- C[Python Client]
    end
    
    subgraph "Features"
        D[Image Upload] --- A
        E[Webcam Interface] --- A
        F[Results Visualization] --- A
    end
```

Key components:
- **Streamlit App**: Interactive web interface for model testing
- **REST API**: Endpoints for model inference
- **Python Client**: Programmatic access to model functionality

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant App
    participant API
    participant Model
    participant DataStore
    
    User->>App: Upload Image
    App->>API: Request Prediction
    API->>Model: Run Inference
    Model->>API: Return Predictions
    API->>App: Return Results
    App->>User: Display Results
    
    opt Log Result
        API->>DataStore: Record Prediction
    end
```

## Deployment Architecture

```mermaid
graph TD
    subgraph "Development Environment"
        A[Development Machine] --> B[Git Repository]
        B --> C[CI/CD Pipeline]
    end
    
    subgraph "Training Infrastructure"
        C --> D[Training Container]
        D --> E[GPU Cluster]
    end
    
    subgraph "Deployment Infrastructure"
        C --> F[Inference Container]
        F --> G[API Server]
        F --> H[Streamlit Server]
    end
    
    subgraph "Storage"
        I[DVC Storage] --> D
        J[Model Registry] --> F
    end
```

## Module Dependencies

```mermaid
graph TD
    app --> mlops
    app --> models
    
    mlops --> models
    mlops --> data
    mlops --> evaluation
    
    deployment --> models
    deployment --> mlops
    
    evaluation --> models
    evaluation --> data
    
    pipeline --> training
    pipeline --> evaluation
    pipeline --> mlops
    
    training --> models
    training --> data
    
    models --> data
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
