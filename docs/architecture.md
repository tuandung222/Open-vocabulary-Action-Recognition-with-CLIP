# CLIP HAR System Architecture

This document provides a comprehensive overview of the CLIP HAR (Human Action Recognition) system architecture from an MLOps perspective.

## System Overview

The CLIP HAR system implements a modern MLOps architecture that spans the entire ML lifecycle from data preparation to production deployment, centered around the CLIP (Contrastive Language-Image Pre-training) model for human action recognition.

### High-Level Architecture

```mermaid
flowchart TD
    subgraph "CLIP HAR End-to-End Platform"
    
    DataLayer["Data Layer"] --> ModelLayer["Model Layer"]
    ModelLayer --> TrainingLayer["Training Layer"]
    TrainingLayer --> EvalLayer["Evaluation Layer"]
    EvalLayer --> MLOpsLayer["MLOps Layer"]
    MLOpsLayer --> DeployLayer["Deployment Layer"]
    DeployLayer --> AppLayer["Application Layer"]
    
    classDef layerClass fill:#f9f9f9,stroke:#333,stroke-width:1px;
    class DataLayer,ModelLayer,TrainingLayer,EvalLayer,MLOpsLayer,DeployLayer,AppLayer layerClass;
    end
```

## Layered Architecture

### 1. Data Layer

The data layer handles dataset management, version control, preprocessing, and augmentation pipelines.

```mermaid
flowchart TB
    subgraph "Data Layer"
    direction TB
    
    DVC["Data Version Control<br>(DVC)"] --> DataLoad["Dataset Loading"]
    DataLoad --> Preprocess["Preprocessing Pipeline"]
    Preprocess --> Augment["Augmentation Strategies"]
    Augment --> Batching["Batch Generation"]
    
    DataConfig["Configuration Management"] --> DataLoad
    DataConfig --> Preprocess
    DataConfig --> Augment
    
    subgraph "Data Validation"
    Cleaning["Data Cleaning"] --- QC["Quality Checks"]
    QC --- Balancing["Class Balancing"]
    end
    
    Preprocess --> Cleaning
    
    classDef mainNode fill:#e1f5fe,stroke:#01579b,stroke-width:1px;
    classDef configNode fill:#fff9c4,stroke:#fbc02d,stroke-width:1px;
    classDef validationNode fill:#f8bbd0,stroke:#880e4f,stroke-width:1px;
    
    class DVC,DataLoad,Preprocess,Augment,Batching mainNode;
    class DataConfig configNode;
    class Cleaning,QC,Balancing validationNode;
    end
```

### 2. Model Layer

The model layer defines the neural network architecture, incorporating CLIP's vision and language components.

```mermaid
flowchart TB
    subgraph "Model Layer"
    direction TB
    
    Input["Input Handler"] --> VisionText["CLIP Architecture"]
    
    subgraph "CLIP Architecture"
    direction TB
    ImageEncoder["Vision Encoder<br>(ViT-B/16)"] --- TextEncoder["Text Encoder<br>(Transformer)"]
    ImageEncoder --> VisualFeatures["Visual Features"]
    TextEncoder --> TextFeatures["Text Features"]
    VisualFeatures --> Similarity["Similarity Calculation"]
    TextFeatures --> Similarity
    end
    
    Similarity --> CustomLayers["Custom Layers"]
    CustomLayers --> ClassifierHead["Classification Head"]
    ClassifierHead --> Output["Output Handler"]
    
    ModelRegistry["Model Registry<br>& Versioning"] -.-> VisionText
    CustomConfig["Model Configuration"] -.-> CustomLayers
    CustomConfig -.-> ClassifierHead
    
    classDef clipNode fill:#bbdefb,stroke:#1565c0,stroke-width:1px;
    classDef featureNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef configNode fill:#fff9c4,stroke:#fbc02d,stroke-width:1px;
    classDef ioNode fill:#e1bee7,stroke:#6a1b9a,stroke-width:1px;
    
    class ImageEncoder,TextEncoder,Similarity clipNode;
    class VisualFeatures,TextFeatures featureNode;
    class ModelRegistry,CustomConfig configNode;
    class Input,Output ioNode;
    end
```

### 3. Training Layer

The training layer orchestrates model training with support for distributed execution and experiment tracking.

```mermaid
flowchart TB
    subgraph "Training Layer"
    direction TB
    
    Config["Training Configuration"] --> TrainInit["Training Initialization"]
    
    TrainInit --> TrainingMode{"Training Mode"}
    TrainingMode -->|Single GPU| SingleGPU["Single GPU Training"]
    TrainingMode -->|Distributed| DistSetup["Distributed Setup"]
    DistSetup --> DDP["DistributedDataParallel (DDP)"]
    DistSetup --> FSDP["FullyShardedDataParallel (FSDP)"]
    
    SingleGPU --> TrainLoop["Training Loop"]
    DDP --> TrainLoop
    FSDP --> TrainLoop
    
    subgraph "Optimization"
    Optimizer["Optimizer (AdamW)"] --> LRScheduler["LR Scheduler"]
    LRScheduler --> GradClip["Gradient Clipping"]
    end
    
    Optimization --> TrainLoop
    
    TrainLoop --> Validation["Validation"]
    Validation --> Checkpoint["Checkpointing"]
    Checkpoint --> EarlyStop{"Early Stopping?"}
    EarlyStop -->|Yes| EndTraining["End Training"]
    EarlyStop -->|No| TrainLoop
    
    Metrics["Metrics Calculation"] --> Validation
    
    ExpTracking["Experiment Tracking"] -.-> TrainLoop
    ExpTracking -.-> Validation
    ExpTracking -.-> Metrics
    
    classDef setupNode fill:#b2dfdb,stroke:#00695c,stroke-width:1px;
    classDef loopNode fill:#f0f4c3,stroke:#9e9d24,stroke-width:1px;
    classDef decisionNode fill:#ffccbc,stroke:#bf360c,stroke-width:1px;
    classDef trackingNode fill:#d1c4e9,stroke:#4527a0,stroke-width:1px;
    
    class TrainInit,SingleGPU,DistSetup,DDP,FSDP setupNode;
    class TrainLoop,Validation,Checkpoint loopNode;
    class TrainingMode,EarlyStop decisionNode;
    class Metrics,ExpTracking trackingNode;
    end
```

### 4. Evaluation Layer

The evaluation layer assesses model performance with comprehensive metrics and visualizations.

```mermaid
flowchart TB
    subgraph "Evaluation Layer"
    direction TB
    
    ModelLoad["Load Model"] --> TestDataset["Prepare Test Dataset"]
    TestDataset --> Inference["Run Inference"]
    
    Inference --> MetricsCalc["Calculate Metrics"]
    
    subgraph "Performance Analysis"
    MetricsCalc --> Accuracy["Accuracy"]
    MetricsCalc --> Precision["Precision/Recall"]
    MetricsCalc --> F1["F1 Score"]
    MetricsCalc --> ConfMatrix["Confusion Matrix"]
    end
    
    subgraph "Visualizations"
    PerClassViz["Per-Class Performance"] --- PredViz["Prediction Examples"]
    PredViz --- ROCCurve["ROC Curves"]
    end
    
    PerformanceAnalysis --> Visualizations
    
    Report["Generate Evaluation Report"] --> ExportResults["Export Results"]
    
    Visualizations --> Report
    MetricsCalc --> Report
    
    classDef prepNode fill:#c5cae9,stroke:#283593,stroke-width:1px;
    classDef metricsNode fill:#b3e5fc,stroke:#0277bd,stroke-width:1px;
    classDef vizNode fill:#dcedc8,stroke:#558b2f,stroke-width:1px;
    classDef reportNode fill:#e1bee7,stroke:#6a1b9a,stroke-width:1px;
    
    class ModelLoad,TestDataset,Inference prepNode;
    class MetricsCalc,Accuracy,Precision,F1,ConfMatrix metricsNode;
    class PerClassViz,PredViz,ROCCurve vizNode;
    class Report,ExportResults reportNode;
    end
```

### 5. MLOps Layer

The MLOps layer implements experiment tracking, model versioning, and pipeline automation.

```mermaid
flowchart TB
    subgraph "MLOps Layer"
    direction LR
    
    subgraph "Experiment Tracking"
    MLflow["MLflow<br>Tracking Server"] --- WandB["Weights & Biases"]
    MLflow --- Tracker["Unified Tracker"]
    WandB --- Tracker
    end
    
    subgraph "Pipeline Management"
    DVC["Data Version<br>Control (DVC)"] --- Pipeline["Pipeline<br>Orchestration"]
    Pipeline --- CI["CI/CD<br>Integration"]
    end
    
    subgraph "Model Registry"
    VersionControl["Model<br>Versioning"] --- Compare["Model<br>Comparison"]
    Compare --- Lineage["Model<br>Lineage"]
    end
    
    subgraph "Artifact Management"
    CheckpointMgmt["Checkpoint<br>Management"] --- Staging["Staging<br>Environment"]
    Staging --- Production["Production<br>Deployment"]
    end
    
    ExperimentTracking --> PipelineManagement
    PipelineManagement --> ModelRegistry
    ModelRegistry --> ArtifactManagement
    
    HuggingFace["HuggingFace<br>Hub Integration"] -.-> ModelRegistry
    
    classDef trackingNode fill:#bbdefb,stroke:#1565c0,stroke-width:1px;
    classDef pipelineNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef registryNode fill:#ffecb3,stroke:#ff8f00,stroke-width:1px;
    classDef artifactNode fill:#e1bee7,stroke:#6a1b9a,stroke-width:1px;
    classDef integrationNode fill:#d7ccc8,stroke:#4e342e,stroke-width:1px;
    
    class MLflow,WandB,Tracker trackingNode;
    class DVC,Pipeline,CI pipelineNode;
    class VersionControl,Compare,Lineage registryNode;
    class CheckpointMgmt,Staging,Production artifactNode;
    class HuggingFace integrationNode;
    end
```

### 6. Deployment Layer

The deployment layer handles model export, optimization, and serving infrastructure.

```mermaid
flowchart TB
    subgraph "Deployment Layer"
    direction TB
    
    ModelExport["Model Export"] --> FormatConversion["Format Conversion"]
    
    subgraph "Model Formats"
    PyTorch["PyTorch<br>Format"] --- ONNX["ONNX<br>Format"]
    ONNX --- TorchScript["TorchScript<br>Format"]
    ONNX --- TensorRT["TensorRT<br>Format"]
    end
    
    FormatConversion --> ModelFormats
    
    ModelFormats --> OptimizeModel["Model Optimization"]
    OptimizeModel --> Benchmark["Performance Benchmarking"]
    
    Benchmark --> ServingInfra["Serving Infrastructure"]
    
    subgraph "Serving Options"
    RestAPI["REST API<br>(FastAPI)"] --- Docker["Docker<br>Containerization"]
    Docker --- K8s["Kubernetes<br>Deployment"]
    end
    
    ServingInfra --> ServingOptions
    
    classDef exportNode fill:#b2dfdb,stroke:#00695c,stroke-width:1px;
    classDef formatNode fill:#c5cae9,stroke:#283593,stroke-width:1px;
    classDef optimizeNode fill:#ffe0b2,stroke:#e65100,stroke-width:1px;
    classDef infraNode fill:#f5f5f5,stroke:#424242,stroke-width:1px;
    
    class ModelExport,FormatConversion exportNode;
    class PyTorch,ONNX,TorchScript,TensorRT formatNode;
    class OptimizeModel,Benchmark optimizeNode;
    class ServingInfra,RestAPI,Docker,K8s infraNode;
    end
```

### 7. Application Layer

The application layer provides user interfaces and API clients for model interaction.

```mermaid
flowchart TB
    subgraph "Application Layer"
    direction TB
    
    subgraph "User Interfaces"
    StreamlitUI["Streamlit UI"] --- WebcamApp["Webcam App"]
    WebcamApp --- BatchUI["Batch Processing UI"]
    end
    
    subgraph "API Services"
    FastAPI["FastAPI Service"] --- Endpoints["REST Endpoints"]
    Endpoints --- APIDoc["API Documentation<br>(Swagger)"]
    end
    
    subgraph "Client Libraries"
    PythonClient["Python Client"] --- RESTClient["REST Client"]
    RESTClient --- CLITool["Command Line Tool"]
    end
    
    ModelServing["Model Serving"] --> UserInterfaces
    ModelServing --> APIServices
    
    UserInterfaces --> Feedback["User Feedback"]
    APIServices --> ClientLibraries
    
    Feedback --> ModelMonitoring["Model Monitoring"]
    ClientLibraries --> ModelMonitoring
    
    classDef uiNode fill:#e1bee7,stroke:#6a1b9a,stroke-width:1px;
    classDef apiNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef clientNode fill:#bbdefb,stroke:#1565c0,stroke-width:1px;
    classDef monitorNode fill:#ffccbc,stroke:#bf360c,stroke-width:1px;
    
    class StreamlitUI,WebcamApp,BatchUI uiNode;
    class FastAPI,Endpoints,APIDoc apiNode;
    class PythonClient,RESTClient,CLITool clientNode;
    class ModelServing,Feedback,ModelMonitoring monitorNode;
    end
```

## Technology Stack

The CLIP HAR project leverages a comprehensive technology stack for MLOps:

```mermaid
graph TD
    subgraph "Technology Stack"
    
    subgraph "Programming Languages"
    Python["Python 3.8+"]
    end
    
    subgraph "Machine Learning"
    PyTorch["PyTorch"] --- HF["Hugging Face<br>Transformers"]
    HF --- CLIP["CLIP Model"]
    end
    
    subgraph "MLOps"
    MLflow["MLflow"] --- WandB["Weights &<br>Biases"]
    WandB --- DVC["DVC"]
    DVC --- HFHub["HuggingFace Hub"]
    end
    
    subgraph "Deployment"
    Docker["Docker"] --- FastAPI["FastAPI"]
    FastAPI --- ONNX["ONNX Runtime"]
    ONNX --- TensorRT["TensorRT"]
    Docker --- Kubernetes["Kubernetes"]
    end
    
    subgraph "UI & Visualization"
    Streamlit["Streamlit"] --- Plotly["Plotly"]
    Plotly --- Matplotlib["Matplotlib"]
    end
    
    Python --> MachineLeaning
    Python --> MLOps
    Python --> Deployment
    Python --> UI&Visualization
    
    classDef langNode fill:#bbdefb,stroke:#1565c0,stroke-width:1px;
    classDef mlNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef mlopsNode fill:#ffe0b2,stroke:#e65100,stroke-width:1px;
    classDef deployNode fill:#e1bee7,stroke:#6a1b9a,stroke-width:1px;
    classDef uiNode fill:#d1c4e9,stroke:#4527a0,stroke-width:1px;
    
    class Python langNode;
    class PyTorch,HF,CLIP mlNode;
    class MLflow,WandB,DVC,HFHub mlopsNode;
    class Docker,FastAPI,ONNX,TensorRT,Kubernetes deployNode;
    class Streamlit,Plotly,Matplotlib uiNode;
    end
```

## Data Flow

The following diagram illustrates the complete data flow through the system:

```mermaid
flowchart LR
    subgraph "CLIP HAR Data Flow"
    
    RawData[("Raw Image<br>Data")] --> DVC["DVC<br>Versioning"]
    DVC --> Preprocessing["Preprocessing<br>Pipeline"]
    
    Preprocessing --> TrainingData[("Processed<br>Training Data")]
    Preprocessing --> ValData[("Validation<br>Data")]
    Preprocessing --> TestData[("Test<br>Data")]
    
    TextPrompts[("Action Class<br>Text Prompts")] --> TrainingPhase
    
    TrainingData --> TrainingPhase["Training<br>Phase"]
    ValData --> TrainingPhase
    
    TrainingPhase --> ModelRegistry["Model<br>Registry"]
    
    TrainingPhase -.-> MLflow["MLflow<br>Tracking"]
    TrainingPhase -.-> WandB["Weights & Biases<br>Tracking"]
    
    ModelRegistry --> ModelExport["Model<br>Export"]
    TestData --> Evaluation["Evaluation<br>Phase"]
    ModelRegistry --> Evaluation
    
    Evaluation -.-> MLflow
    Evaluation -.-> WandB
    
    ModelExport --> ONNXFormat["ONNX<br>Format"]
    ModelExport --> TorchScriptFormat["TorchScript<br>Format"]
    ModelExport --> TensorRTFormat["TensorRT<br>Format"]
    
    ONNXFormat --> DeploymentPhase["Deployment<br>Phase"]
    TorchScriptFormat --> DeploymentPhase
    TensorRTFormat --> DeploymentPhase
    
    DeploymentPhase --> FastAPI["FastAPI<br>Service"]
    DeploymentPhase --> StreamlitApp["Streamlit<br>Application"]
    
    NewData[("New Image<br>Data")] --> FastAPI
    NewData --> StreamlitApp
    
    FastAPI --> Predictions[("Action<br>Predictions")]
    StreamlitApp --> Predictions
    
    classDef dataNode fill:#bbdefb,stroke:#1565c0,stroke-width:1px,stroke-dasharray: 5 5;
    classDef processNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef modelNode fill:#ffecb3,stroke:#ff8f00,stroke-width:1px;
    classDef trackingNode fill:#e1bee7,stroke:#6a1b9a,stroke-width:1px;
    classDef deployNode fill:#f5f5f5,stroke:#424242,stroke-width:1px;
    classDef predictNode fill:#ffccbc,stroke:#bf360c,stroke-width:1px;
    
    class RawData,TrainingData,ValData,TestData,TextPrompts,NewData,Predictions dataNode;
    class DVC,Preprocessing,TrainingPhase,Evaluation processNode;
    class ModelRegistry,ModelExport,ONNXFormat,TorchScriptFormat,TensorRTFormat modelNode;
    class MLflow,WandB trackingNode;
    class DeploymentPhase,FastAPI,StreamlitApp deployNode;
    end
```

## Deployment Architecture

The deployment architecture shows how the system components are deployed in production:

```mermaid
flowchart TB
    subgraph "CLIP HAR Deployment Architecture"
    
    subgraph "Model Training Infrastructure"
    GPU["GPU Cluster"] --- TrainingEnv["Training<br>Environment"]
    TrainingEnv --- ModelRegistry["Model<br>Registry"]
    end
    
    subgraph "MLOps Platform"
    MLflow["MLflow<br>Server"] --- WandB["Weights & Biases"]
    WandB --- DVC["DVC<br>Storage"]
    end
    
    subgraph "Containerization"
    DockerRegistry["Docker<br>Registry"] --- TrainImage["Training<br>Container"]
    DockerRegistry --- InferenceImage["Inference<br>Container"]
    DockerRegistry --- AppImage["App<br>Container"]
    end
    
    subgraph "Kubernetes Cluster"
    APIServer["Inference API<br>Server"] --- StreamlitServer["Streamlit<br>Server"]
    StreamlitServer --- MLflowServer["MLflow<br>Server"]
    
    subgraph "Scalability"
    APIAutoscaler["API<br>Autoscaler"] --- APIReplicas["API<br>Replicas"]
    end
    
    APIServer --> APIAutoscaler
    end
    
    subgraph "Client Applications"
    WebUI["Web<br>Interface"] --- Mobile["Mobile<br>App"]
    Mobile --- PythonClient["Python<br>Client"]
    end
    
    ModelTrainingInfrastructure --> MLOpsPlatform
    ModelRegistry --> Containerization
    Containerization --> KubernetesCluster
    KubernetesCluster --> ClientApplications
    
    LoadBalancer["Load<br>Balancer"] --> KubernetesCluster
    
    classDef infraNode fill:#bbdefb,stroke:#1565c0,stroke-width:1px;
    classDef mlopsNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef containerNode fill:#ffecb3,stroke:#ff8f00,stroke-width:1px;
    classDef k8sNode fill:#e1bee7,stroke:#6a1b9a,stroke-width:1px;
    classDef clientNode fill:#d1c4e9,stroke:#4527a0,stroke-width:1px;
    classDef networkNode fill:#f5f5f5,stroke:#424242,stroke-width:1px;
    
    class GPU,TrainingEnv,ModelRegistry infraNode;
    class MLflow,WandB,DVC mlopsNode;
    class DockerRegistry,TrainImage,InferenceImage,AppImage containerNode;
    class APIServer,StreamlitServer,MLflowServer,APIAutoscaler,APIReplicas k8sNode;
    class WebUI,Mobile,PythonClient clientNode;
    class LoadBalancer networkNode;
    end
```

## CLIP HAR Model Architecture

The specific architecture of the CLIP HAR model:

```mermaid
flowchart TB
    subgraph "CLIP HAR Model Architecture"
    
    subgraph "Input"
    ImageInput["Image Input"] --- TextPrompts["Text Prompts<br>(Action Classes)"]
    end
    
    subgraph "CLIP Encoders"
    ImageInput --> VisionEncoder["Vision Transformer<br>(ViT-B/16)"]
    TextPrompts --> TextEncoder["Text Encoder<br>(Transformer)"]
    
    VisionEncoder --> ImageEmbedding["Image<br>Embedding"]
    TextEncoder --> TextEmbedding["Text<br>Embeddings"]
    end
    
    subgraph "Similarity Calculation"
    ImageEmbedding --> Normalize1["L2<br>Normalization"]
    TextEmbedding --> Normalize2["L2<br>Normalization"]
    
    Normalize1 --> CosineSim["Cosine<br>Similarity"]
    Normalize2 --> CosineSim
    end
    
    subgraph "Classification"
    CosineSim --> LogitScale["Temperature<br>Scaling"]
    LogitScale --> Softmax["Softmax"]
    Softmax --> Predictions["Action<br>Predictions"]
    end
    
    classDef inputNode fill:#bbdefb,stroke:#1565c0,stroke-width:1px;
    classDef encoderNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px;
    classDef embeddingNode fill:#ffe0b2,stroke:#e65100,stroke-width:1px;
    classDef simNode fill:#e1bee7,stroke:#6a1b9a,stroke-width:1px;
    classDef classNode fill:#d1c4e9,stroke:#4527a0,stroke-width:1px;
    
    class ImageInput,TextPrompts inputNode;
    class VisionEncoder,TextEncoder encoderNode;
    class ImageEmbedding,TextEmbedding embeddingNode;
    class Normalize1,Normalize2,CosineSim simNode;
    class LogitScale,Softmax,Predictions classNode;
    end
```

## Complete System Integration

The integration of all components in the CLIP HAR project:

```
+---------------------+     +------------------------+     +----------------+
|  Data Management    |     | Training Infrastructure|     |  CI/CD Pipeline|
|                     |     |                        |     |                |
| +-------+  +------+ |     | +------+  +---------+ |     | +-----+  +---+ |
| |Raw Data|->|DVC   |=|=====>|Dev   |--|Training | |     | |Git  |->|CI | |
| +-------+  |Version| |     | |Env  |  |Env      | |     | |Repo |  |Flow| |
|            |Control|=|=====>+------+  +---------+ |     | +-----+  +---+ |
|            +------+ |     |            |          |     |            |   |
|               |     |     |            |          |     |            v   |
|               v     |     |            v          |     |        +------+|
|          +--------+ |     |         +--------+    |     |        |Tests | |
|          |Data    | |     |         |Experiment|  |     |        +------+|
|          |Process | |     |         |Tracking  |  |     |            |   |
|          +--------+ |     |         +--------+    |     |            v   |
+---------------------+     +------------------------+     |     +---------+|
        |                            |                     |     |Build    ||
        |                            |                     |     |Images   ||
        |                            |                     |     +---------+|
        |                            |                     +----------------+
        |                            |                             |
        |                            |                             |
        v                            v                             v
+---------------------+     +------------------------+     +----------------+
|  Production Env     |<----|Monitoring & Observ.    |     |                |
|                     |     |                        |     |                |
| +-------+  +------+ |     | +------+  +---------+ |     |                |
| |K8s    |--|Load  | |     | |Model |--|Service  | |     |                |
| |Cluster|  |Balancer| |     | |Metrics|  |Metrics  | |     |                |
| +-------+  +------+ |     | +------+  +---------+ |     |                |
|     |         |     |     |               |       |     |                |
|     v         v     |     |               v       |     |                |
| +--------------+    |     |         +---------+   |     |                |
| |API Endpoints |    |     |         |Alerting |   |     |                |
| +--------------+    |     |         |System   |   |     |                |
|     ^               |     |         +---------+   |     |                |
+-----|---------------+     +------------------------+     |                |
      |                                 |                  |                |
      |                                 +------------------|----------------+
      v                                                    |
+------------+                                             |
|  End Users |                                             |
+------------+                                             |
                                                           |
     +-----------------------------------------------------|
     |
     v
Feedback Loop
```

## CI/CD Pipeline

The CI/CD pipeline automates testing, building, and deployment processes for the CLIP HAR project.

```
                         +----------------+
                         |  Code Changes  |
                         +-------+--------+
                                 |
                                 v
                        +------------------+
                        | GitHub Repository|
                        +--------+---------+
                           /            \
                          /              \
                         v                v
               +----------------+  +-------------------+
               | Pull Request   |  | Push to Main      |
               | Event          |  | Event             |
               +-------+--------+  +--------+----------+
                       |                     |
                       v                     |
               +----------------+            |
               | Code Quality   |            |
               | Checks         |            |
               +-------+--------+            |
                       |                     |
                       v                     |
               +----------------+            |
               | Unit Tests     |            |
               +-------+--------+            |
                       |                     |
                       v                     v
               +----------------+    +-------------------+
               | Integration    |--->| Build Workflow    |
               | Tests          |    +--------+----------+
               +----------------+             |
                                        /-----+-----\
                                       /             \
                                      v               v
                         +----------------+  +------------------+
                         | Build Training |  | Build App        |
                         | Container      |  | Container        |
                         +-------+--------+  +--------+---------+
                                 |                     |
                                 v                     v
                         +----------------+  +------------------+
                         | Push to        |  | Push to          |
                         | Registry       |  | Registry         |
                         +-------+--------+  +--------+---------+
                                 |                     |
                                 \---------+-----------/
                                           |
                                           v
                                 +-------------------+
                                 | Deploy to         |
                                 | Development       |
                                 +--------+----------+
                                          |
                                          v
                                 +-------------------+
                                 | Evaluate          |
                                 | Performance       |
                                 +--------+----------+
                                     /        \
                                    /          \
                                   v            v
                        +-----------------+  +------------------+
                        | Deploy to       |  | Notify           |
                        | Production      |  | Team             |
                        +-----------------+  +------------------+
```

## Kubernetes Deployment Workflow

The Kubernetes deployment workflow for the CLIP HAR system:

```
+--------------------------------------+
|     Kubernetes Deployment Workflow   |
+--------------------------------------+
         |
         v
+------------------+     +------------------+     +------------------+
| Infrastructure   |     | Storage          |     | Deployments      |
|                  |     |                  |     |                  |
| K8s Cluster      |     | Persistent       |     | Inference API    |
| CLIP HAR Namespace|---->| Volumes         |---->| Deployment       |
| ConfigMaps       |     | PV Claims        |     | Streamlit UI     |
| & Secrets        |     | Model Storage    |     | MLflow Deployment|
+------------------+     +------------------+     +------------------+
                                                         |
                                                  /------+------\
                                                 /               \
                                                v                 v
                                +------------------+     +------------------+
                                | Ingress &        |     | Scaling &        |
                                | Networking       |     | Updates          |
                                |                  |     |                  |
                                | Ingress Controller|    | Horizontal Pod   |
                                | TLS Termination  |     | Autoscaler       |
                                | Path-based       |     | Rolling Updates  |
                                | Routing          |     | Health Checks    |
                                +------------------+     +------------------+
                                                                |
                                                                v
                                                        +------------------+
                                                        | Monitoring       |
                                                        |                  |
                                                        | Prometheus       |
                                                        | Grafana          |
                                                        | Alert Manager    |
                                                        +------------------+
```

## Model Performance Optimization Pipeline

The model optimization pipeline improves inference performance:

```
                      +---------------------------+
                      | Trained PyTorch Model     |
                      +-----------+---------------+
                                  |
                                  v
                      +--------------------------+
                      | Export to ONNX           |
                      +-----------+--------------+
                                  |
                                  v
                      +--------------------------+
                      | ONNX Optimization        |
                      +-----------+--------------+
                                  |
                                  v
                      +--------------------------+
                      | ONNX Quantization        |
                      +-----------+--------------+
                                  |
                                  v
                      +--------------------------+
                      | TensorRT Conversion      |------------+
                      +-----------+--------------+            |
                                  |                           |
                                  v                           v
                      +--------------------------+   +-------------------+
                      | TensorRT Calibration     |   | CPU Deployment    |
                      +-----------+--------------+   +--------+----------+
                                  |                           |
                                  v                           |
                      +--------------------------+            |
                      | GPU Deployment           |            |
                      +-----------+--------------+            |
                                  |                           |
                                  \-----------+---------------/
                                              |
                                              v
                                  +--------------------------+
                                  | Benchmarking             |
                                  |                          |
                                  | Latency Testing          |
                                  | Throughput Testing       |
                                  | Memory Usage             |
                                  | Power Efficiency         |
                                  +-----------+--------------+
                                              |
                                              v
                                  +--------------------------+
                                  | Optimization Report      |
                                  +--------------------------+
```

## Conclusion

The CLIP HAR system architecture provides a comprehensive ML platform for human action recognition using CLIP, with full MLOps capabilities for experiment tracking, model versioning, and deployment. The modular design enables flexibility, scalability, and maintainability across the entire ML lifecycle.
