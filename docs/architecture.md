# CLIP HAR System Architecture

## Overview

The CLIP HAR system provides an end-to-end solution for Human Action Recognition using OpenAI's CLIP (Contrastive Language-Image Pre-training) model. By leveraging CLIP's powerful multi-modal capabilities, our system can recognize human actions from images with high accuracy in both zero-shot and fine-tuned settings.

## System Architecture

```mermaid
flowchart TD
    classDef primary fill:#4285F4,stroke:#333,stroke-width:1px,color:white
    classDef secondary fill:#34A853,stroke:#333,stroke-width:1px,color:white
    classDef tertiary fill:#FBBC05,stroke:#333,stroke-width:1px,color:white
    classDef interface fill:#EA4335,stroke:#333,stroke-width:1px,color:white
    
    subgraph Data["Data Platform"]
        DVC[DVC Storage]
        Preprocess[Preprocessing]
        Augment[Augmentation]
        Dataset[Dataset Creation]
        
        DVC --> Preprocess
        Preprocess --> Augment
        Augment --> Dataset
    end
    
    subgraph Model["Model Architecture"]
        CLIP[CLIP Base Model]
        Custom[Custom Adaptation Layer]
        ZeroShot[Zero-Shot Classifier]
        FineTuned[Fine-Tuned Classifier]
        
        CLIP --> Custom
        Custom --> ZeroShot
        Custom --> FineTuned
    end
    
    subgraph Training["Training Platform"]
        Single[Single GPU]
        DDP[Distributed Data Parallel]
        FSDP[Fully Sharded Data Parallel]
        Checkpoint[Checkpoint Management]
        
        Single --> Checkpoint
        DDP --> Checkpoint
        FSDP --> Checkpoint
    end
    
    subgraph MLOps["MLOps Platform"]
        MLflow[MLflow Tracking]
        WandB[Weights & Biases]
        HF[HuggingFace Hub]
        Pipelines[Automated Pipelines]
        
        MLflow --- WandB
        MLflow --> HF
        WandB --> HF
        Pipelines --> HF
    end
    
    subgraph Eval["Evaluation System"]
        Metrics[Performance Metrics]
        Confusion[Confusion Matrix]
        ClassAnalysis[Per-Class Analysis]
        
        Metrics --> Confusion
        Metrics --> ClassAnalysis
    end
    
    subgraph Deploy["Deployment Platform"]
        ONNX[ONNX Export]
        TensorRT[TensorRT Export]
        API[REST API]
        Docker[Docker Containers]
        
        ONNX --> API
        TensorRT --> API
        API --> Docker
    end
    
    subgraph UI["User Interfaces"]
        Streamlit[Streamlit App]
        REST[REST API Client]
        
        Streamlit --> REST
    end
    
    Dataset --> Model
    Model --> Training
    Training --> Eval
    Eval --> Deploy
    Deploy --> UI
    
    MLOps --> Data
    MLOps --> Model
    MLOps --> Training
    MLOps --> Eval
    MLOps --> Deploy
    
    class Data,Model,Training,MLOps,Eval,Deploy,UI primary
    class DVC,CLIP,DDP,MLflow,Metrics,ONNX,Streamlit secondary
    class Preprocess,Custom,FSDP,WandB,Confusion,API,REST tertiary
    class Augment,ZeroShot,Checkpoint,HF,ClassAnalysis,Docker interface
```

## Component Details

### 1. Data Platform

The Data Platform manages dataset versioning, preprocessing, and augmentation, ensuring consistent data quality throughout the machine learning lifecycle.

```mermaid
flowchart LR
    classDef primary fill:#4285F4,stroke:#333,stroke-width:1px,color:white
    classDef secondary fill:#34A853,stroke:#333,stroke-width:1px,color:white
    classDef tertiary fill:#FBBC05,stroke:#333,stroke-width:1px,color:white
    
    raw[Raw Images] --> dvc[(DVC Storage)]
    dvc --> validate{Validation}
    validate -->|Pass| preprocess[Preprocessing]
    validate -->|Fail| reject[Rejection Queue]
    
    preprocess --> augment[Augmentation]
    augment --> split{Train/Test Split}
    split --> train[Training Set]
    split --> test[Test Set]
    
    config[Configuration] --> preprocess
    config --> augment
    config --> split
    
    class raw,dvc,preprocess,augment,split,train,test primary
    class validate,config secondary
    class reject tertiary
```

**Key Features:**
- **Data Version Control**: Tracks dataset changes using DVC
- **Automated Validation**: Ensures data quality and consistency
- **Configurable Preprocessing**: Custom pipelines for different datasets
- **Advanced Augmentation**: Improves model robustness and generalization

### 2. Model Architecture

The Model Architecture leverages CLIP's powerful multi-modal representation capabilities for human action recognition.

```mermaid
flowchart TD
    classDef primary fill:#4285F4,stroke:#333,stroke-width:1px,color:white
    classDef secondary fill:#34A853,stroke:#333,stroke-width:1px,color:white
    classDef tertiary fill:#FBBC05,stroke:#333,stroke-width:1px,color:white
    
    input[Input Image] --> visionEncoder[Vision Encoder]
    prompts[Text Prompts] --> textEncoder[Text Encoder]
    
    visionEncoder --> imageFeatures[Image Features]
    textEncoder --> textFeatures[Text Features]
    
    imageFeatures --> similarity[Similarity Calculation]
    textFeatures --> similarity
    
    similarity --> logits[Classification Logits]
    logits --> output[Action Prediction]
    
    class input,visionEncoder,imageFeatures,similarity,logits,output primary
    class prompts,textEncoder,textFeatures secondary
```

**Key Features:**
- **CLIP Integration**: Utilizes CLIP's vision and text encoders
- **Zero-Shot Capability**: Classifies actions without labeled examples
- **Custom Prompting**: Optimized text prompts for action recognition
- **Fine-Tuning Options**: Parameter-efficient tuning strategies

### 3. Training Platform

The Training Platform orchestrates model training across various hardware configurations, from single GPUs to distributed clusters.

```mermaid
flowchart TD
    classDef primary fill:#4285F4,stroke:#333,stroke-width:1px,color:white
    classDef secondary fill:#34A853,stroke:#333,stroke-width:1px,color:white
    classDef tertiary fill:#FBBC05,stroke:#333,stroke-width:1px,color:white
    
    config[Training Config] --> mode{Training Mode}
    
    mode -->|Single GPU| single[Single GPU Training]
    mode -->|Multi-GPU| distributed[Distributed Training]
    
    distributed -->|DDP| ddp[DistributedDataParallel]
    distributed -->|FSDP| fsdp[FullyShardedDataParallel]
    
    single --> train[Training Loop]
    ddp --> train
    fsdp --> train
    
    train --> validate[Validation]
    validate --> earlyStop{Early Stopping}
    
    earlyStop -->|Yes| endTraining[End Training]
    earlyStop -->|No| train
    
    validate --> checkpoint[Checkpointing]
    checkpoint --> bestModel[Best Model]
    
    class config,mode,train,validate,checkpoint,bestModel primary
    class single,distributed,ddp,fsdp secondary
    class earlyStop,endTraining tertiary
```

**Key Features:**
- **Flexible Deployment**: Single-GPU, DDP, and FSDP training modes
- **Mixed Precision**: Accelerated training with FP16/BF16
- **Automated Checkpointing**: Saves best models based on validation metrics
- **Early Stopping**: Prevents overfitting and saves compute resources

### 4. MLOps Platform

The MLOps Platform integrates experiment tracking, model versioning, and pipeline automation for a seamless development workflow.

```mermaid
flowchart TD
    classDef primary fill:#4285F4,stroke:#333,stroke-width:1px,color:white
    classDef secondary fill:#34A853,stroke:#333,stroke-width:1px,color:white
    classDef tertiary fill:#FBBC05,stroke:#333,stroke-width:1px,color:white
    
    experiment[Experiment Config] --> tracking{Tracking System}
    
    tracking -->|Local| mlflow[MLflow Server]
    tracking -->|Cloud| wandb[Weights & Biases]
    
    mlflow --> unified[Unified Tracking API]
    wandb --> unified
    
    unified --> metrics[Metrics Dashboard]
    unified --> artifacts[Artifact Storage]
    
    artifacts --> modelRegistry[Model Registry]
    modelRegistry --> huggingface[HuggingFace Hub]
    
    pipeline[Automated Pipeline] --> dvc[DVC Pipeline]
    dvc --> cicd[CI/CD Integration]
    
    class experiment,tracking,unified,modelRegistry,pipeline primary
    class mlflow,wandb,metrics,artifacts,huggingface secondary
    class dvc,cicd tertiary
```

**Key Features:**
- **Dual Tracking**: Simultaneous MLflow and W&B experiment tracking
- **Unified API**: Consistent interface for multiple tracking backends
- **Automated Pipelines**: End-to-end training and evaluation workflows
- **HuggingFace Integration**: Seamless model sharing and collaboration

### 5. Evaluation System

The Evaluation System provides comprehensive model assessment with detailed metrics and visualizations.

```mermaid
flowchart LR
    classDef primary fill:#4285F4,stroke:#333,stroke-width:1px,color:white
    classDef secondary fill:#34A853,stroke:#333,stroke-width:1px,color:white
    classDef tertiary fill:#FBBC05,stroke:#333,stroke-width:1px,color:white
    
    model[Trained Model] --> inference[Inference]
    testSet[Test Dataset] --> inference
    
    inference --> predictions[Predictions]
    predictions --> metrics[Metrics Calculation]
    
    metrics --> accuracy[Accuracy]
    metrics --> precision[Precision/Recall]
    metrics --> f1[F1 Score]
    
    metrics --> confusion[Confusion Matrix]
    metrics --> perClass[Per-Class Analysis]
    metrics --> examples[Example Predictions]
    
    confusion --> report[Evaluation Report]
    perClass --> report
    examples --> report
    
    class model,testSet,inference,predictions,metrics,report primary
    class accuracy,precision,f1,confusion secondary
    class perClass,examples tertiary
```

**Key Features:**
- **Comprehensive Metrics**: Accuracy, precision, recall, F1 score
- **Visual Analysis**: Confusion matrices and performance plots
- **Per-Class Breakdown**: Detailed analysis by action category
- **Example Visualization**: Success and failure case examination

### 6. Deployment Platform

The Deployment Platform optimizes models for production and serves them through various interfaces.

```mermaid
flowchart TD
    classDef primary fill:#4285F4,stroke:#333,stroke-width:1px,color:white
    classDef secondary fill:#34A853,stroke:#333,stroke-width:1px,color:white
    classDef tertiary fill:#FBBC05,stroke:#333,stroke-width:1px,color:white
    
    model[Trained Model] --> export{Export Format}
    
    export -->|PyTorch| pytorch[PyTorch Model]
    export -->|ONNX| onnx[ONNX Model]
    export -->|TensorRT| tensorrt[TensorRT Engine]
    
    pytorch --> api[FastAPI Service]
    onnx --> api
    tensorrt --> api
    
    api --> docker[Docker Container]
    docker --> deployment{Deployment}
    
    deployment -->|Local| local[Local Serving]
    deployment -->|Cloud| cloud[Cloud Deployment]
    
    class model,export,api,docker,deployment primary
    class pytorch,onnx,tensorrt secondary
    class local,cloud tertiary
```

**Key Features:**
- **Multiple Export Formats**: PyTorch, ONNX, and TensorRT
- **Fast Inference API**: High-performance serving with FastAPI
- **Containerization**: Docker images for consistent deployment
- **Deployment Options**: Local and cloud deployment strategies

### 7. User Interfaces

The User Interfaces provide interactive ways to interact with the model.

```mermaid
flowchart LR
    classDef primary fill:#4285F4,stroke:#333,stroke-width:1px,color:white
    classDef secondary fill:#34A853,stroke:#333,stroke-width:1px,color:white
    classDef tertiary fill:#FBBC05,stroke:#333,stroke-width:1px,color:white
    
    user[End User] --> interface{Interface}
    
    interface -->|Web| streamlit[Streamlit App]
    interface -->|API| rest[REST API]
    interface -->|Batch| batch[Batch Processing]
    
    streamlit --> imageUpload[Image Upload]
    streamlit --> webcam[Webcam Feed]
    streamlit --> visualization[Result Visualization]
    
    rest --> clientApp[Client Application]
    rest --> curl[cURL/Requests]
    
    batch --> folderProcess[Folder Processing]
    batch --> exportResults[Results Export]
    
    class user,interface,streamlit,rest,batch primary
    class imageUpload,webcam,visualization,clientApp secondary
    class curl,folderProcess,exportResults tertiary
```

**Key Features:**
- **Streamlit Application**: Interactive web interface for model testing
- **REST API**: Programmatic access for integration
- **Real-time Processing**: Live webcam action recognition
- **Batch Processing**: Process multiple images at once

## Technology Stack

- **Framework**: PyTorch, HuggingFace Transformers
- **Training**: Distributed training with DDP and FSDP
- **Experiment Tracking**: MLflow and Weights & Biases
- **Data Management**: DVC for versioning
- **Deployment**: Docker, FastAPI, ONNX Runtime
- **UI**: Streamlit

## Implementation Benefits

- **Modular Design**: Components can be used independently or together
- **Scalable Training**: From single GPU to multi-node clusters
- **Production-Ready**: Optimized for deployment in various environments
- **Extensible**: Easy to adapt for new action classes or domains
- **Comprehensive MLOps**: Full lifecycle management from data to deployment 