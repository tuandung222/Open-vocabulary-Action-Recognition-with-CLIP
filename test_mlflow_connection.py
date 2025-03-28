#!/usr/bin/env python
"""Minimal script to test MLflow server connection."""

import sys
print(f"Python version: {sys.version}")

import mlflow
print(f"MLflow version: {mlflow.__version__}")

# Set the tracking URI
tracking_uri = "http://localhost:5001"
print(f"Setting tracking URI to: {tracking_uri}")
mlflow.set_tracking_uri(tracking_uri)

# Just check if we can get the tracking URI back
retrieved_uri = mlflow.get_tracking_uri()
print(f"Retrieved tracking URI: {retrieved_uri}")

# Try to list experiments
try:
    print("Attempting to list experiments...")
    experiments = mlflow.search_experiments()
    print(f"SUCCESS: Found {len(experiments)} experiments")
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
except Exception as e:
    print(f"ERROR: Failed to list experiments: {e}")
    import traceback
    traceback.print_exc()

print("Test completed") 