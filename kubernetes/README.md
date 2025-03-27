# Kubernetes Deployment for CLIP HAR

This directory contains Kubernetes configuration files for deploying the CLIP HAR system in a production environment.

## Overview

The CLIP HAR Kubernetes deployment consists of two main components:

1. **CLIP HAR Inference Service** - API service for model inference
2. **Monitoring Stack** - Prometheus, Elasticsearch, Kibana, and Grafana for monitoring and observability

## Prerequisites

- Kubernetes cluster (v1.19+)
- kubectl configured to connect to your cluster
- NVIDIA GPU operators installed (for GPU support)
- Storage class configured for persistent volumes

## Deployment

### Quick Start

Deploy the entire stack with a single command:

```bash
kubectl apply -f ./
```

This will deploy both the inference service and the monitoring stack.

### Component-specific Deployment

To deploy components individually:

```bash
# Deploy only the inference service
kubectl apply -f clip-har-inference.yaml

# Deploy only the monitoring stack
kubectl apply -f clip-har-monitoring.yaml
```

## Inference Service Configuration

The inference service is configured via a ConfigMap. You can modify the configuration by editing the `clip-har-inference-config` ConfigMap in the `clip-har-inference.yaml` file.

Key configuration options:

- Model type (PyTorch, ONNX, TensorRT)
- Batch size
- Worker threads
- Monitoring options

## Scaling

The service is configured with a Horizontal Pod Autoscaler (HPA) that automatically scales the number of pods based on CPU and memory utilization.

To adjust scaling parameters:

```bash
kubectl edit hpa clip-har-inference-hpa
```

## Monitoring Access

After deployment, you can access the monitoring dashboards:

```bash
# Prometheus
kubectl port-forward svc/prometheus-server 9090:9090

# Kibana
kubectl port-forward svc/kibana 5601:5601

# Grafana
kubectl port-forward svc/grafana 3000:3000
```

### Grafana Login

- URL: http://localhost:3000
- Username: admin
- Password: admin123 (This should be changed in production)

## Production Considerations

For production deployments:

1. **Security**: Replace hardcoded passwords with Kubernetes secrets
2. **Networking**: Configure Ingress resources for secure external access
3. **GPU Allocation**: Adjust GPU requests based on your hardware capabilities
4. **Resource Limits**: Tune resource requests and limits based on performance testing
5. **Storage**: Configure appropriate storage classes for your environment

## Troubleshooting

### Common Issues

1. **Pod Pending State**: Check if PVCs are being provisioned correctly
   ```bash
   kubectl get pvc
   ```

2. **GPU Not Available**: Verify GPU operator installation
   ```bash
   kubectl get pods -n gpu-operator-resources
   ```

3. **Service Unavailable**: Check pod logs
   ```bash
   kubectl logs -l app=clip-har,component=inference
   ```

## Upgrading

To upgrade the deployment:

1. Update the image tag in the deployment file
2. Apply the updated configuration
   ```bash
   kubectl apply -f clip-har-inference.yaml
   ```

The deployment is configured with a rolling update strategy, ensuring zero-downtime upgrades. 