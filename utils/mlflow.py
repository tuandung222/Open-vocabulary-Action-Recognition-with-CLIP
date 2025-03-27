"""
MLflow tracking utilities.
"""

def log_params(client, params):
    """Log parameters to MLflow."""
    if client is None:
        return
    
    # Log parameters
    for key, value in params.items():
        client.log_param(key, value)

def log_metrics(client, metrics, step=None):
    """Log metrics to MLflow."""
    if client is None:
        return
    
    # Log metrics
    for key, value in metrics.items():
        client.log_metric(key, value, step=step)

def log_model(client, model, name="model"):
    """Log model to MLflow."""
    if client is None:
        return
    
    # Log model
    client.log_model(model, name)

def log_artifact(client, local_path, artifact_path=None):
    """Log artifact to MLflow."""
    if client is None:
        return
    
    # Log artifact
    client.log_artifact(local_path, artifact_path)