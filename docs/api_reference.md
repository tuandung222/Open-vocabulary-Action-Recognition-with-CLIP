# Inference API Reference

This document provides a detailed reference for the CLIP HAR Inference API endpoints, parameters, and response formats.

## API Overview

The CLIP HAR Inference API is built with FastAPI and provides endpoints for human action recognition on images. The API supports multiple input formats and provides detailed classification results.

**Base URL:** `http://localhost:8000` (default)

## Endpoints

### GET `/`

Get service information.

#### Response

```json
{
  "name": "CLIP HAR Inference Service",
  "model_name": "clip_har_model.pt",
  "model_type": "pytorch",
  "device": "cuda",
  "class_names": ["calling", "clapping", "cycling", ...]
}
```

### GET `/health`

Health check endpoint.

#### Response

```json
{
  "status": "healthy"
}
```

### POST `/predict`

Run inference on an image provided as a base64 string or URL.

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_data` | string | No¹ | Base64-encoded image data |
| `image_url` | string | No¹ | URL of an image |
| `top_k` | integer | No | Number of top predictions to return (default: 5) |

¹ Either `image_data` or `image_url` must be provided.

#### Example Request

```json
{
  "image_data": "iVBORw0KGgoAAAANSUhEUgAAA...",
  "top_k": 3
}
```

OR

```json
{
  "image_url": "https://example.com/image.jpg",
  "top_k": 3
}
```

#### Response

```json
{
  "predictions": [
    {
      "rank": 1,
      "class_id": 5,
      "class_name": "dancing",
      "score": 0.9532
    },
    {
      "rank": 2,
      "class_id": 2,
      "class_name": "clapping",
      "score": 0.0321
    },
    {
      "rank": 3,
      "class_id": 12,
      "class_name": "running",
      "score": 0.0087
    }
  ],
  "inference_time": 0.0456,
  "model_name": "clip_har_model.pt"
}
```

### POST `/predict/image`

Run inference on an uploaded image file.

#### Request Form

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Image file to analyze |
| `top_k` | integer | No | Number of top predictions to return (default: 5) |

#### Example Request

```bash
curl -X POST "http://localhost:8000/predict/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "top_k=3"
```

#### Response

```json
{
  "predictions": [
    {
      "rank": 1,
      "class_id": 5,
      "class_name": "dancing",
      "score": 0.9532
    },
    {
      "rank": 2,
      "class_id": 2,
      "class_name": "clapping",
      "score": 0.0321
    },
    {
      "rank": 3,
      "class_id": 12,
      "class_name": "running",
      "score": 0.0087
    }
  ],
  "inference_time": 0.0456,
  "model_name": "clip_har_model.pt"
}
```

## Response Format

All prediction endpoints return a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | array | List of prediction objects |
| `inference_time` | number | Time taken for inference in seconds |
| `model_name` | string | Name of the model used for inference |

Each prediction object in the `predictions` array contains:

| Field | Type | Description |
|-------|------|-------------|
| `rank` | integer | Rank of the prediction (1 = top prediction) |
| `class_id` | integer | ID of the predicted class |
| `class_name` | string | Name of the predicted class |
| `score` | number | Confidence score (0-1) for the prediction |

## Error Responses

### 400 Bad Request

Returned when the request is invalid, such as when neither `image_data` nor `image_url` is provided.

```json
{
  "detail": "No input provided. Please provide either image_data or image_url."
}
```

### 500 Internal Server Error

Returned when there is an error processing the request.

```json
{
  "detail": "Error during inference: Internal server error"
}
```

## Using the API

### Python Client

The project provides a Python client for easy integration:

```python
from CLIP_HAR_PROJECT.mlops.inference_serving import InferenceClient

# Initialize client
client = InferenceClient(url="http://localhost:8000")

# Predict from image file
result = client.predict_from_image_path("path/to/image.jpg")
print(f"Top prediction: {result['predictions'][0]['class_name']}")
print(f"Confidence: {result['predictions'][0]['score']:.4f}")

# Predict from URL
result = client.predict_from_image_url("https://example.com/image.jpg")
```

### CURL Examples

#### Predict from URL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg", "top_k": 3}'
```

#### Predict from Base64

```bash
# Convert image to base64
base64_image=$(base64 -w 0 image.jpg)

# Send request
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"image_data\": \"$base64_image\", \"top_k\": 3}"
```

#### Predict from File Upload

```bash
curl -X POST "http://localhost:8000/predict/image" \
  -H "accept: application/json" \
  -F "file=@image.jpg" \
  -F "top_k=3"
```

## API Configuration

The API can be configured with the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to the model file | `/app/models/checkpoints/best_model.pt` |
| `MODEL_TYPE` | Type of model (pytorch, onnx, torchscript) | `pytorch` |
| `DEVICE` | Device to run inference on (cuda, cpu) | `cuda` if available, otherwise `cpu` |
| `PORT` | Port to run the API on | `8000` |
| `HOST` | Host to run the API on | `0.0.0.0` |

## API Authentication

The API currently does not implement authentication. For production use, consider adding authentication mechanisms such as:

- API keys
- OAuth 2.0
- JWT tokens

## Rate Limiting

The API does not implement rate limiting by default. For production use, consider adding rate limiting using:

- FastAPI middleware
- Reverse proxy (Nginx, Traefik)
- API gateway
