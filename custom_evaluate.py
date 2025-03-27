#!/usr/bin/env python
# Script to evaluate a trained model on the HAR test set

# Set tokenizers parallelism to avoid deadlocks with multiprocessing
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
from pathlib import Path
from prettytable import PrettyTable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, CLIPTokenizerFast

# Import project modules
from configs.default import get_config
from data.preprocessing import (
    collate_fn,
    get_class_mappings,
    prepare_har_dataset,
)
from models.clip_model import CLIPLabelRetriever


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on HAR test set"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="CLIP model name or path",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="a photo of person/people who is/are {label}",
        help="Template for creating text prompts",
    )

    # Evaluation arguments
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Evaluation batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for evaluation results",
    )

    # Visualization arguments
    parser.add_argument(
        "--visualize_samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )

    return parser.parse_args()


def load_model_from_checkpoint(model_path, model_name, labels, prompt_template):
    """Load a model from a checkpoint."""
    # Create model
    model = CLIPLabelRetriever.from_pretrained(
        model_name, labels=labels, prompt_template=prompt_template
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    return model


def evaluate_model(model, test_dataset, batch_size=64, num_workers=4, device="cuda"):
    """Evaluate a model on the test set."""
    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # Move model to device
    model = model.to(device)
    model.eval()

    # Collect predictions and references
    all_predictions = []
    all_references = []
    all_scores = []

    # Evaluate on test set
    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Get predictions
            scores = model(**batch, return_loss=False)
            predictions = torch.argmax(scores, dim=1)

            # Store predictions and references
            all_predictions.append(predictions.cpu())
            all_references.append(batch["label_id"].cpu())
            all_scores.append(scores.cpu())

    # Concatenate results
    predictions = torch.cat(all_predictions).numpy()
    references = torch.cat(all_references).numpy()
    scores = torch.cat(all_scores).numpy()
    
    # Clean up CUDA cache to prevent memory issues
    if device == "cuda":
        torch.cuda.empty_cache()

    return predictions, references, scores


def compute_metrics(predictions, references, class_names):
    """Compute evaluation metrics."""
    # Compute classification report
    report = classification_report(
        references, predictions, target_names=class_names, output_dict=True
    )

    # Convert to DataFrame
    report_df = pd.DataFrame(report).T

    # Compute confusion matrix
    cm = confusion_matrix(references, predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    return report_df, cm, cm_normalized


def plot_confusion_matrix(cm_normalized, class_names, output_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(15, 15))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def visualize_predictions(
    model, test_dataset, class_names, num_samples=5, output_dir="results"
):
    """Visualize model predictions on random samples."""
    # Create directory for visualization
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    # Get device
    device = next(model.parameters()).device

    # Choose random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    # Create subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))
    if num_samples == 1:
        axes = [axes]

    # Process each sample
    for i, idx in enumerate(indices):
        # Get sample
        sample = test_dataset[idx]
        image = sample["image"]
        true_label = sample["label_id"]

        # Predict
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        with torch.no_grad():
            _, pred_labels, scores = model.predict(pixel_values)

        pred_label = pred_labels[0]

        # Get top-3 predictions
        top_scores = torch.from_numpy(scores).topk(3, dim=1)
        top_idxs = top_scores.indices[0].cpu().numpy()
        top_values = top_scores.values[0].cpu().numpy()
        top_classes = [class_names[idx] for idx in top_idxs]

        # Plot image
        axes[i].imshow(image)
        axes[i].set_title(f"True: {class_names[true_label]}\nPred: {pred_label}")
        axes[i].axis("off")

        # Add top-3 predictions as text
        text = "\n".join([f"{c}: {v:.2f}" for c, v in zip(top_classes, top_values)])
        axes[i].text(
            0.05,
            0.95,
            text,
            transform=axes[i].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "visualizations", "predictions.png"), dpi=300)
    plt.close()


def save_metrics_summary(report_df, predictions, references, class_names, output_dir, model_dir=None):
    """Create and save a summary table with evaluation metrics.
    
    Args:
        report_df: DataFrame with classification report
        predictions: Model predictions
        references: Ground truth labels
        class_names: List of class names
        output_dir: Output directory for evaluation results
        model_dir: Model checkpoint directory (optional)
    """
    # Create summary table
    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Value"]
    
    # Add overall metrics - fix accuracy access
    # Accuracy is stored directly in the DataFrame
    accuracy = np.mean(predictions == references)
    summary_table.add_row(["Accuracy", f"{accuracy:.4f}"])
    
    # Add macro averages
    for metric in ["precision", "recall", "f1-score"]:
        value = report_df.loc["macro avg", metric]
        summary_table.add_row([f"Macro {metric.replace('-', ' ').title()}", f"{value:.4f}"])
    
    # Create per-class table
    class_table = PrettyTable()
    class_table.field_names = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        if class_name in report_df.index:
            row = report_df.loc[class_name]
            class_table.add_row([
                class_name,
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1-score']:.4f}",
                f"{row['support']}"
            ])
    
    # Calculate class distribution and confusion
    class_counts = {}
    class_correct = {}
    
    for ref in references:
        if ref not in class_counts:
            class_counts[ref] = 0
        class_counts[ref] += 1
    
    for pred, ref in zip(predictions, references):
        if ref not in class_correct:
            class_correct[ref] = {"correct": 0, "total": 0}
        class_correct[ref]["total"] += 1
        if pred == ref:
            class_correct[ref]["correct"] += 1
    
    # Create per-class accuracy table
    accuracy_table = PrettyTable()
    accuracy_table.field_names = ["Class", "Accuracy", "Correct/Total", "Distribution"]
    
    for label_id in sorted(class_counts.keys()):
        class_name = class_names[label_id] if label_id < len(class_names) else f"Unknown ({label_id})"
        acc = class_correct[label_id]["correct"] / class_correct[label_id]["total"]
        dist = class_counts[label_id] / len(references) * 100
        accuracy_table.add_row([
            class_name,
            f"{acc*100:.2f}%",
            f"{class_correct[label_id]['correct']}/{class_correct[label_id]['total']}",
            f"{dist:.1f}%"
        ])
    
    # Save to output directory
    with open(os.path.join(output_dir, "metrics_summary.txt"), "w") as f:
        f.write("# Evaluation Metrics Summary\n\n")
        f.write("## Overall Metrics\n\n")
        f.write(summary_table.get_string())
        f.write("\n\n## Per-Class Metrics\n\n")
        f.write(class_table.get_string())
        f.write("\n\n## Per-Class Accuracy\n\n")
        f.write(accuracy_table.get_string())
    
    # Also save to model directory if provided
    if model_dir:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            model_dir = model_dir.parent
        
        with open(os.path.join(model_dir, "evaluation_metrics.txt"), "w") as f:
            f.write("# Evaluation Metrics Summary\n\n")
            f.write("## Overall Metrics\n\n")
            f.write(summary_table.get_string())
            f.write("\n\n## Per-Class Metrics\n\n")
            f.write(class_table.get_string())
            f.write("\n\n## Per-Class Accuracy\n\n")
            f.write(accuracy_table.get_string())
    
    return summary_table, class_table, accuracy_table


def visualize_top_predictions(model, test_dataset, class_names, output_dir, num_samples=10, top_k=5, device="cuda"):
    """
    Visualize the top k predictions for sample images from the test set.
    
    Args:
        model: The model to use for predictions
        test_dataset: The test dataset
        class_names: List of class names
        output_dir: Output directory for visualizations
        num_samples: Number of samples to visualize
        top_k: Number of top predictions to show
        device: Device to run the model on
    """
    import random
    from PIL import Image
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Select random samples
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    # Process each sample
    visualization_paths = []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            sample = test_dataset[idx]
            
            # Get image and label
            image = sample["image"]
            label_id = sample["label_id"]
            true_label = class_names[label_id] if label_id < len(class_names) else f"Unknown ({label_id})"
            
            # Process sample through model
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            
            # Get logits
            logits = model(**batch, return_loss=False)
            
            # Get top-k predictions
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=1)
            
            # Convert to numpy for plotting
            top_probs = top_probs.cpu().numpy().flatten()
            top_indices = top_indices.cpu().numpy().flatten()
            
            # Get class names
            top_classes = [class_names[idx] for idx in top_indices]
            
            # Plot image and predictions
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot image
            ax1.imshow(image)
            ax1.axis('off')
            ax1.set_title(f"True label: {true_label}")
            
            # Plot predictions as horizontal bar chart
            y_pos = range(top_k)
            ax2.barh(y_pos, top_probs)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_classes)
            ax2.set_xlabel('Probability')
            ax2.set_title('Top Predictions')
            
            # Invert y-axis to show top prediction at the top
            ax2.invert_yaxis()
            
            # Add text with probability values
            for j, (prob, label) in enumerate(zip(top_probs, top_classes)):
                ax2.text(prob + 0.01, j, f"{prob:.4f}", va='center')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            vis_path = os.path.join(vis_dir, f"sample_{i+1}_vis.png")
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            visualization_paths.append(vis_path)
            
            print(f"Saved visualization for sample {i+1}/{num_samples}")
    
    # Create summary visualization of all samples
    if num_samples > 1:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8)) if num_samples >= 10 else plt.subplots(2, num_samples//2 + num_samples%2, figsize=(num_samples*2, 8))
        axes = axes.flatten()
        
        for i, vis_path in enumerate(visualization_paths[:min(10, num_samples)]):
            img = plt.imread(vis_path)
            axes[i].imshow(img)
            axes[i].axis('off')
        
        # Remove empty axes
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        # Save summary visualization
        summary_path = os.path.join(vis_dir, "summary_visualization.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        visualization_paths.append(summary_path)
    
    return visualization_paths


def visualize_top_predictions_table(model, test_dataset, class_names, output_dir, num_samples=10, top_k=5, device="cuda"):
    """
    Create a table visualization of top-k predictions vs ground truth for sample images.
    
    Args:
        model: The model to use for predictions
        test_dataset: The test dataset
        class_names: List of class names
        output_dir: Output directory for visualizations
        num_samples: Number of samples to visualize
        top_k: Number of top predictions to show
        device: Device to run the model on
    """
    import random
    from PIL import Image
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from prettytable import PrettyTable
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Select random samples
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    # Create results table
    results_table = PrettyTable()
    results_table.field_names = ["Sample", "Ground Truth"] + [f"Prediction {i+1}" for i in range(top_k)]
    
    # Process each sample
    visualization_paths = []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            sample = test_dataset[idx]
            
            # Get image and label
            image = sample["image"]
            label_id = sample["label_id"]
            true_label = class_names[label_id] if label_id < len(class_names) else f"Unknown ({label_id})"
            
            # Process sample through model
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            
            # Get logits
            logits = model(**batch, return_loss=False)
            
            # Get top-k predictions
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=1)
            
            # Convert to numpy for plotting
            top_probs = top_probs.cpu().numpy().flatten()
            top_indices = top_indices.cpu().numpy().flatten()
            
            # Get class names
            top_classes = [class_names[idx] for idx in top_indices]
            
            # Add to table
            row = [f"Sample {i+1}", true_label]
            for j in range(top_k):
                pred_text = f"{top_classes[j]} ({top_probs[j]:.4f})"
                row.append(pred_text)
            results_table.add_row(row)
            
            # Save individual visualization as before
            # [... existing visualization code ...]
    
    # Save table to text file
    table_path = os.path.join(vis_dir, "top_predictions_table.txt")
    with open(table_path, "w") as f:
        f.write("# Top Predictions vs Ground Truth\n\n")
        f.write(results_table.get_string())
    
    # Save table to CSV for easier processing
    csv_path = os.path.join(vis_dir, "top_predictions_table.csv")
    with open(csv_path, "w") as f:
        f.write("Sample,Ground Truth," + ",".join([f"Prediction {i+1}" for i in range(top_k)]) + "\n")
        for i, idx in enumerate(indices):
            sample = test_dataset[idx]
            label_id = sample["label_id"]
            true_label = class_names[label_id] if label_id < len(class_names) else f"Unknown ({label_id})"
            
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            logits = model(**batch, return_loss=False)
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=1)
            
            top_probs = top_probs.cpu().numpy().flatten()
            top_indices = top_indices.cpu().numpy().flatten()
            top_classes = [class_names[idx] for idx in top_indices]
            
            row = [f"Sample {i+1}", true_label]
            for j in range(top_k):
                pred_text = f"{top_classes[j]} ({top_probs[j]:.4f})"
                row.append(pred_text)
            f.write(",".join([f'"{item}"' for item in row]) + "\n")
    
    print(f"Saved prediction table to {table_path}")
    return table_path


def main():
    """Main evaluation function."""
    # Parse command line arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and image processor
    tokenizer = CLIPTokenizerFast.from_pretrained(args.model_name)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_name)

    # Prepare dataset
    datasets, class_names = prepare_har_dataset(
        tokenizer=tokenizer, image_processor=image_processor
    )

    # Load model
    model = load_model_from_checkpoint(
        args.model_path, args.model_name, class_names, args.prompt_template
    )

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate model
    print(f"Evaluating model on test set...")
    predictions, references, scores = evaluate_model(
        model,
        datasets["test"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # Calculate and print direct metrics
    accuracy = np.mean(predictions == references)
    print(f"\nDirect Metrics:")
    print(f"Accuracy: {accuracy:.4f}")

    # Save raw predictions and references for debugging
    print(f"Saving raw predictions and references for debugging...")
    np.save(os.path.join(args.output_dir, "predictions.npy"), predictions)
    np.save(os.path.join(args.output_dir, "references.npy"), references)
    
    # Print detailed class distribution
    class_counts = {}
    for label in references:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    print("\nTest Set Class Distribution:")
    for label_id, count in sorted(class_counts.items()):
        class_name = class_names[label_id] if label_id < len(class_names) else f"Unknown ({label_id})"
        print(f"  {class_name}: {count} samples ({count/len(references)*100:.1f}%)")
    
    # Print per-class accuracy
    class_correct = {}
    for pred, ref in zip(predictions, references):
        if ref not in class_correct:
            class_correct[ref] = {"correct": 0, "total": 0}
        class_correct[ref]["total"] += 1
        if pred == ref:
            class_correct[ref]["correct"] += 1
    
    print("\nPer-Class Accuracy:")
    for label_id in sorted(class_correct.keys()):
        class_name = class_names[label_id] if label_id < len(class_names) else f"Unknown ({label_id})"
        acc = class_correct[label_id]["correct"] / class_correct[label_id]["total"]
        print(f"  {class_name}: {acc*100:.1f}% ({class_correct[label_id]['correct']}/{class_correct[label_id]['total']})")

    # Compute metrics
    print(f"Computing metrics...")
    report_df, cm, cm_normalized = compute_metrics(predictions, references, class_names)

    # Print results
    print("\nClassification Report:")
    print(report_df.round(4))

    # Save results
    report_df.round(4).to_csv(
        os.path.join(args.output_dir, "classification_report.csv")
    )
    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), cm)

    # Plot confusion matrix
    print(f"Plotting confusion matrix...")
    plot_confusion_matrix(cm_normalized, class_names, args.output_dir)

    # Visualize predictions
    print(f"Visualizing predictions...")
    visualization_paths = visualize_top_predictions(
        model,
        datasets["test"],
        class_names,
        args.output_dir,
        num_samples=10,
        top_k=5,
        device=device
    )
    
    # Generate table visualization
    print(f"\nGenerating prediction table...")
    table_path = visualize_top_predictions_table(
        model,
        datasets["test"],
        class_names,
        args.output_dir,
        num_samples=10,
        top_k=5,
        device=device
    )
    
    # Log visualizations to MLflow
    try:
        import mlflow
        if mlflow.active_run():
            print("Logging visualizations to MLflow...")
            # Log image visualizations
            for vis_path in visualization_paths:
                mlflow.log_artifact(vis_path)
            
            # Log table visualizations
            mlflow.log_artifact(table_path)
            csv_path = os.path.join(os.path.dirname(table_path), "top_predictions_table.csv")
            if os.path.exists(csv_path):
                mlflow.log_artifact(csv_path)
    except (ImportError, Exception) as e:
        print(f"Could not log to MLflow: {e}")
    
    # Log visualizations to wandb
    try:
        import wandb
        if wandb.run:
            print("Logging visualizations to Weights & Biases...")
            # Log image visualizations
            for vis_path in visualization_paths:
                wandb.log({os.path.basename(vis_path): wandb.Image(vis_path)})
            
            # Log table as artifact
            wandb.save(table_path)
            
            # Also create a wandb Table for interactive viewing
            try:
                import pandas as pd
                csv_path = os.path.join(os.path.dirname(table_path), "top_predictions_table.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    wandb_table = wandb.Table(dataframe=df)
                    wandb.log({"Top Predictions": wandb_table})
            except Exception as e:
                print(f"Could not create wandb table: {e}")
    except (ImportError, Exception) as e:
        print(f"Could not log to Weights & Biases: {e}")
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Explicitly close all matplotlib figures
    plt.close('all')
    
    # Save metrics summary
    save_metrics_summary(report_df, predictions, references, class_names, args.output_dir, args.model_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
