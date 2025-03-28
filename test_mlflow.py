#!/usr/bin/env python
"""Simple script to test MLflow logging."""

import mlflow
import numpy as np
from datetime import datetime
import time
import os
import sys

print("Starting MLflow test script...")

# Check MLflow version
print(f"MLflow version: {mlflow.__version__}")

# Connect to MLflow server
tracking_uri = "http://localhost:5001"
print(f"Setting tracking URI to: {tracking_uri}")
try:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    print("Successfully connected to MLflow server")
except Exception as e:
    print(f"Error connecting to MLflow server: {e}")
    sys.exit(1)

# Set experiment
experiment_name = "test_experiment"
try:
    print(f"Looking for experiment named: {experiment_name}")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Found existing experiment with ID: {experiment_id}")
    else:
        print(f"Creating new experiment named: {experiment_name}")
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment with ID: {experiment_id}")
    
    mlflow.set_experiment(experiment_name)
except Exception as e:
    print(f"Error setting up experiment: {e}")
    sys.exit(1)

# Start a run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"test_run_{timestamp}"
print(f"Starting MLflow run named: {run_name}")

try:
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"Started MLflow run with ID: {run_id}")
        
        # Log some parameters
        print("Logging parameters...")
        mlflow.log_param("test_param_1", "value1")
        mlflow.log_param("test_param_2", 42)
        
        # Log some metrics
        print("Logging metrics...")
        for i in range(5):
            mlflow.log_metric("accuracy", 0.5 + i*0.1, step=i)
            mlflow.log_metric("loss", 1.0 - i*0.1, step=i)
            print(f"  Logged step {i}: accuracy={0.5 + i*0.1:.4f}, loss={1.0 - i*0.1:.4f}")
            time.sleep(0.1)  # Small delay to simulate training steps
        
        # Create and log a simple plot
        try:
            print("Creating and logging a plot...")
            import matplotlib.pyplot as plt
            
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            
            plt.figure(figsize=(8, 6))
            plt.plot(x, y)
            plt.title("Test Plot")
            plt.xlabel("X")
            plt.ylabel("sin(X)")
            plt.grid(True)
            
            # Save and log plot
            plot_path = "test_plot.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            print(f"Logged plot as artifact: {plot_path}")
            plt.close()
            
            # Check if file was created
            if os.path.exists(plot_path):
                print(f"Plot file exists locally: {os.path.abspath(plot_path)}")
            else:
                print(f"Warning: Plot file not found locally")
                
        except Exception as e:
            print(f"Warning: Could not create or log plot: {e}")
            import traceback
            traceback.print_exc()

        print(f"Successfully logged to MLflow!")
        print(f"View this run at: {tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}")
        
except Exception as e:
    print(f"Error during MLflow run: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
print("MLflow test script completed successfully!") 