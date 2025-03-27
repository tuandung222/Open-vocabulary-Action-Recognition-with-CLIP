# CLIP HAR Model Architecture

This guide provides a detailed explanation of the model architecture used in our CLIP HAR (Human Action Recognition) project.

## Overview

Our model architecture leverages CLIP (Contrastive Language-Image Pre-training) as a foundation, with additional components designed specifically for human action recognition tasks.

```
┌───────────────────────────────────────────────────────────┐
│                 CLIP HAR Model Architecture                │
└───────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
┌────────▼─────────┐┌──────▼───────┐┌────────▼─────────┐
│   Image Encoder  ││ Text Encoder ││  Adapter Layers  │
│    (Vision Tr.)  ││  (Text Tr.)  ││                  │
└────────┬─────────┘└──────┬───────┘└────────┬─────────┘
         │                 │                 │
         └────────┬────────┘                 │
                  │                          │
          ┌───────▼───────┐           ┌──────▼───────┐
          │ Joint Feature │           │ Action-Specific│
          │   Space       │           │   Components  │
          └───────┬───────┘           └──────┬───────┘
                  │                          │
                  └────────────┬─────────────┘
                               │
                       ┌───────▼───────┐
                       │ Classification │
                       │     Head      │
                       └───────────────┘
```

## Component Details

### 1. CLIP Foundation

The core of our model is built on OpenAI's CLIP architecture, which consists of two primary components:

#### Image Encoder (Vision Transformer)

```
┌────────────────────────────────────┐
│        Vision Transformer          │
└────────────────────────────────────┘
              │
┌─────────────▼─────────────┐
│      Image Patchification  │
│  (divide image into patches)│
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│     Linear Projection +    │
│      Position Embedding    │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│                           │
│   Transformer Encoders    │◄─┐
│  (Multi-head Attention +  │  │
│      MLP Blocks)          │  │ x N layers
│                           │  │
└─────────────┬─────────────┘  │
              └─────────────────┘
              │
┌─────────────▼─────────────┐
│    Global Representation   │
│     (CLS token output)     │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│   Visual Feature Vector    │
└─────────────┬─────────────┘
              ▼
```

The Vision Transformer:
- Divides the input image into fixed-size patches (typically 16x16 pixels)
- Projects these patches into an embedding space
- Adds positional embeddings to preserve spatial information
- Processes through multiple transformer encoder layers
- Extracts a global representation via a special CLS token or averaging

#### Text Encoder (Text Transformer)

```
┌────────────────────────────────────┐
│        Text Transformer            │
└────────────────────────────────────┘
              │
┌─────────────▼─────────────┐
│     Tokenization of Text   │
│   (text -> token sequence) │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│     Token Embedding +      │
│     Position Embedding     │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│                           │
│   Transformer Encoders    │◄─┐
│  (Multi-head Attention +  │  │
│      MLP Blocks)          │  │ x N layers
│                           │  │
└─────────────┬─────────────┘  │
              └─────────────────┘
              │
┌─────────────▼─────────────┐
│    Global Representation   │
│     (CLS token output)     │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│     Text Feature Vector    │
└─────────────┬─────────────┘
              ▼
```

The Text Transformer:
- Tokenizes input text into a sequence of tokens
- Embeds these tokens and adds positional information
- Processes through multiple transformer encoder layers
- Extracts a global text representation

### 2. Joint Embedding Space

CLIP maps both image and text into a common embedding space where similar concepts are located near each other:

```
                    ┌───────────────────────────┐
                    │   Joint Embedding Space   │
                    └───────────────────────────┘
                            
    Image Features                        Text Features
         │                                     │
         ▼                                     ▼
      ┌──────┐                             ┌──────┐
      │Norm  │                             │Norm  │
      └──┬───┘                             └──┬───┘
         │                                     │
         │               Cosine                │
         └─────────────Similarity─────────────┘
                         │
                         ▼
                   ┌────────────┐
                   │Contrastive │
                   │ Loss       │
                   └────────────┘
```

The key idea is that images and their matching text descriptions should have similar representations in this space, while unrelated pairs should be far apart.

### 3. HAR-Specific Adaptations

For human action recognition, we extend the CLIP architecture with specialized components:

#### Temporal Modeling for Actions

```
┌─────────────────────────────────────────┐
│        Temporal Modeling Module         │
└─────────────────────────────────────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
┌─────▼─────┐          ┌──────▼──────┐
│ Attention │          │ Motion       │
│ Pooling   │          │ Feature      │
└─────┬─────┘          │ Extraction   │
      │                └──────┬───────┘
      │                       │
      └───────────┬───────────┘
                  │
         ┌────────▼────────┐
         │  Fusion Module  │
         └────────┬────────┘
                  │
                  ▼
```

This module helps the model understand temporal relationships critical for action recognition.

#### Action-Specific Classifier Head

```
┌──────────────────────────────────────┐
│        Action Classifier Head        │
└──────────────────────────────────────┘
                  │
          ┌───────▼───────┐
          │Global Feature │
          │     Vector    │
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │  Dropout Layer│
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │  FC Layer 1   │
          │  (with ReLU)  │
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │  FC Layer 2   │
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │    Softmax    │
          └───────┬───────┘
                  │
                  ▼
            Action Classes
```

The classifier head transforms the joint features into predictions for specific action classes.

## Training Process

During training, our model uses a combination of:

1. **Contrastive Loss**: To align image and text embeddings (from CLIP pre-training)
2. **Classification Loss**: To optimize for action classification
3. **Fine-Tuning Strategy**: We use a staged approach:
   - Initially freeze CLIP weights and train only adapter layers
   - Gradually unfreeze deeper layers for fine-tuning

```
┌──────────────────────────────────────────────────────────────┐
│                 Training Process Overview                     │
└──────────────────────────────────────────────────────────────┘

┌─────────┐     ┌───────────┐     ┌──────────────┐    ┌──────────┐
│ Images  │────►│ CLIP Image│─────►Action-Specific│───►│Prediction│
│         │     │ Encoder   │     │  Components   │    │          │
└─────────┘     └───────────┘     └──────────────┘    └────┬─────┘
                                                           │
                                                           │
┌─────────┐     ┌───────────┐                         ┌────▼─────┐
│ Action  │────►│ CLIP Text │─────────────────────────►│  Loss    │
│ Labels  │     │ Encoder   │                         │ Function │
└─────────┘     └───────────┘                         └──────────┘
```

## Inference Process

During inference, we have a streamlined pipeline:

```
┌───────────────────────────────────────────────────────────────────┐
│                       Inference Pipeline                           │
└───────────────────────────────────────────────────────────────────┘

┌──────────┐      ┌─────────────┐      ┌───────────────┐     ┌──────────────┐
│  Input   │      │  Pre-       │      │  CLIP-based   │     │  Post-       │
│  Image   │─────►│  processing │─────►│  HAR Model    │────►│  processing   │
│          │      │             │      │               │     │               │
└──────────┘      └─────────────┘      └───────────────┘     └──────┬───────┘
                                                                    │
                                                                    ▼
                                                             ┌─────────────┐
                                                             │  Action     │
                                                             │  Prediction │
                                                             └─────────────┘
```

## Implementation Details

Our implementation uses PyTorch with Hugging Face Transformers for the CLIP components:

```python
from transformers import CLIPModel, CLIPProcessor

class CLIPHARModel(nn.Module):
    def __init__(self, clip_model_name, num_classes):
        super().__init__()
        # Load pre-trained CLIP
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # Freeze CLIP parameters initially
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # Action-specific layers
        embed_dim = self.clip.projection_dim
        self.temporal_layer = TemporalAttention(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, pixel_values, input_ids=None):
        # Get image features from CLIP
        image_features = self.clip.get_image_features(pixel_values)
        
        # Apply temporal modeling
        features = self.temporal_layer(image_features)
        
        # Get class predictions
        logits = self.classifier(features)
        
        # If text input is provided, compute similarity too
        if input_ids is not None:
            text_features = self.clip.get_text_features(input_ids)
            similarity = F.cosine_similarity(image_features, text_features)
            return logits, similarity
        
        return logits
```

## Model Variations

We support several model variations:

```
┌───────────────────────────────────────────────────────┐
│                 CLIP HAR Model Variants               │
└───────────────────────────────────────────────────────┘
                         │
          ┌──────────────┼────────────────┐
          │              │                │
┌─────────▼────────┐┌────▼─────┐┌─────────▼─────────┐
│  Base CLIP HAR   ││ CLIP HAR ││   CLIP HAR with   │
│     Model        ││ Ensemble ││Temporal Attention │
└─────────┬────────┘└────┬─────┘└─────────┬─────────┘
          │              │                │
          │         ┌────▼─────┐          │
          │         │Multiple  │          │
          │         │ Models   │          │
          │         └────┬─────┘          │
          │              │                │
          └──────────────┼────────────────┘
                         │
                 ┌───────▼───────┐
                 │ Export Format │
                 └───────┬───────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼─────┐   ┌──────▼───────┐ ┌──────▼─────┐
│   PyTorch   │   │    ONNX     │ │  TensorRT  │
└─────────────┘   └──────────────┘ └────────────┘
```

## Hardware Considerations

Our model is designed to work efficiently on various hardware:

```
┌────────────────────────────────────────────────────┐
│            Hardware Support Matrix                 │
└────────────────────────────────────────────────────┘

Hardware      | Training    | Inference   | Export Formats
─────────────────────────────────────────────────────────
CPU           |   ✓*        |     ✓       | PyTorch, ONNX
Single GPU    |   ✓        |     ✓       | PyTorch, ONNX, TensorRT
Multi-GPU     |   ✓        |     ✓       | PyTorch, ONNX, TensorRT
Edge Device   |   ✗        |     ✓       | ONNX, TensorRT

* Limited by performance and memory constraints
```

## Further Reading

To better understand our model architecture:

1. Explore the CLIP HAR model implementation in the `models/` directory
2. Review the original CLIP paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
3. Check our model adaptation code in `CLIP_HAR_PROJECT/models/clip_model.py`
4. Examine the training configurations in `CLIP_HAR_PROJECT/configs/` 