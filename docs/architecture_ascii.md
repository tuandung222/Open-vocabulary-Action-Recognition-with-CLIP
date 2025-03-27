# CLIP HAR System Architecture

This document provides a comprehensive overview of the CLIP HAR (Human Action Recognition) system architecture from an MLOps perspective.

## System Overview

The CLIP HAR system implements a modern MLOps architecture that spans the entire ML lifecycle from data preparation to production deployment, centered around the CLIP (Contrastive Language-Image Pre-training) model for human action recognition.

### High-Level Architecture

```
+----------------+     +---------------+     +----------------+     +----------------+
| Data Layer     | --> | Model Layer   | --> | Training Layer | --> | Evaluation     |
|                |     |               |     |                |     | Layer          |
+----------------+     +---------------+     +----------------+     +----------------+
                                                                          |
         +-----------------------------------------------------------------+
         |
         v
+----------------+     +---------------+     +----------------+
| MLOps Layer    | --> | Deployment    | --> | Application    |
|                |     | Layer         |     | Layer          |
+----------------+     +---------------+     +----------------+
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

## CLIP HAR Model Architecture

The specific architecture of the CLIP HAR model:

```
+--------------+                  +---------------------+
| Image Input  | ---------------- | Text Prompts        |
|              |                  | (Action Classes)    |
+--------------+                  +---------------------+
      |                                   |
      v                                   v
+--------------+                  +---------------------+
| Vision       |                  | Text                |
| Encoder      |                  | Encoder             |
| (ViT-B/16)   |                  | (Transformer)       |
+--------------+                  +---------------------+
      |                                   |
      v                                   v
+--------------+                  +---------------------+
| Image        |                  | Text                |
| Embedding    |                  | Embeddings          |
+--------------+                  +---------------------+
      |                                   |
      v                                   v
+--------------+                  +---------------------+
| L2           |                  | L2                  |
| Normalization|                  | Normalization       |
+--------------+                  +---------------------+
      |                                   |
      \-----------------+------------------/
                        |
                        v
               +------------------+
               | Cosine           |
               | Similarity       |
               +--------+---------+
                        |
                        v
               +------------------+
               | Temperature      |
               | Scaling          |
               +--------+---------+
                        |
                        v
               +------------------+
               | Softmax          |
               +--------+---------+
                        |
                        v
               +------------------+
               | Action           |
               | Predictions      |
               +------------------+
```

## Conclusion

The CLIP HAR system architecture provides a comprehensive ML platform for human action recognition using CLIP, with full MLOps capabilities for experiment tracking, model versioning, and deployment. The modular design enables flexibility, scalability, and maintainability across the entire ML lifecycle. 