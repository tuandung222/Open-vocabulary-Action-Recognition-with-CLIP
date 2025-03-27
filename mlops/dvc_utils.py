import os
import subprocess
import yaml
from typing import Dict, List, Optional, Union, Any
from pathlib import Path


def setup_dvc_repo(
    root_dir: str = ".", remote_url: Optional[str] = None, remote_name: str = "origin"
) -> None:
    """
    Set up a DVC repository in the given directory.

    Args:
        root_dir: Root directory for the repository
        remote_url: Optional URL for a remote DVC storage
        remote_name: Name for the remote
    """
    # Initialize DVC if not already initialized
    dvc_dir = os.path.join(root_dir, ".dvc")
    if not os.path.exists(dvc_dir):
        subprocess.run(["dvc", "init"], cwd=root_dir, check=True)
        print(f"DVC initialized in {root_dir}")
    else:
        print(f"DVC already initialized in {root_dir}")

    # Add remote storage if provided
    if remote_url:
        try:
            # Check if remote exists first
            result = subprocess.run(
                ["dvc", "remote", "list"],
                cwd=root_dir,
                check=True,
                capture_output=True,
                text=True,
            )

            if remote_name not in result.stdout:
                subprocess.run(
                    ["dvc", "remote", "add", remote_name, remote_url],
                    cwd=root_dir,
                    check=True,
                )
                print(f"Added DVC remote '{remote_name}' at {remote_url}")
            else:
                print(f"DVC remote '{remote_name}' already exists")

            # Set as default remote
            subprocess.run(
                ["dvc", "remote", "default", remote_name], cwd=root_dir, check=True
            )
            print(f"Set '{remote_name}' as default remote")

        except subprocess.CalledProcessError as e:
            print(f"Error setting up DVC remote: {e}")


def add_data_to_dvc(
    data_path: str,
    root_dir: str = ".",
    commit_message: Optional[str] = None,
    git_add: bool = True,
    git_commit: bool = True,
) -> bool:
    """
    Add data to DVC tracking.

    Args:
        data_path: Path to the data to track
        root_dir: DVC repository root directory
        commit_message: Optional message for git commit
        git_add: Whether to add DVC file to git
        git_commit: Whether to commit DVC file to git

    Returns:
        True if successful, False otherwise
    """
    try:
        # Add the data to DVC
        subprocess.run(["dvc", "add", data_path], cwd=root_dir, check=True)
        print(f"Added {data_path} to DVC tracking")

        if git_add:
            # Add the .dvc file to git
            dvc_file = f"{data_path}.dvc"
            subprocess.run(["git", "add", dvc_file], cwd=root_dir, check=True)
            print(f"Added {dvc_file} to git")

            if git_commit:
                # Commit the .dvc file
                msg = commit_message or f"Add {data_path} to DVC tracking"
                subprocess.run(["git", "commit", "-m", msg], cwd=root_dir, check=True)
                print(f"Committed {dvc_file} to git with message: '{msg}'")

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adding data to DVC: {e}")
        return False


def get_tracked_data(root_dir: str = ".") -> List[str]:
    """
    Get a list of all DVC-tracked data files.

    Args:
        root_dir: DVC repository root directory

    Returns:
        List of tracked data file paths
    """
    try:
        result = subprocess.run(
            ["dvc", "list", "--dvc-only", "--recursive", "."],
            cwd=root_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        tracked_files = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        return tracked_files
    except subprocess.CalledProcessError as e:
        print(f"Error listing DVC-tracked files: {e}")
        return []


def push_data_to_remote(
    data_path: Optional[str] = None,
    remote_name: Optional[str] = None,
    root_dir: str = ".",
) -> bool:
    """
    Push DVC-tracked data to remote storage.

    Args:
        data_path: Optional specific data path to push
        remote_name: Optional remote name to push to
        root_dir: DVC repository root directory

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = ["dvc", "push"]

        # Add specific data path if provided
        if data_path:
            cmd.append(data_path)

        # Add specific remote if provided
        if remote_name:
            cmd.extend(["-r", remote_name])

        # Execute the push command
        subprocess.run(cmd, cwd=root_dir, check=True)

        what = data_path or "all tracked data"
        where = f"to remote '{remote_name}'" if remote_name else "to default remote"
        print(f"Successfully pushed {what} {where}")

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pushing data to remote: {e}")
        return False


def create_dvc_pipeline(
    stages: Dict[str, Dict[str, Any]],
    output_file: str = "dvc.yaml",
    root_dir: str = ".",
) -> bool:
    """
    Create a DVC pipeline configuration file.

    Args:
        stages: Dictionary of stage definitions
        output_file: Path to the output YAML file
        root_dir: DVC repository root directory

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create the pipeline config
        pipeline_config = {"stages": stages}

        # Write the pipeline config to the YAML file
        output_path = os.path.join(root_dir, output_file)
        with open(output_path, "w") as f:
            yaml.dump(pipeline_config, f, default_flow_style=False)

        print(f"Created DVC pipeline configuration at {output_path}")
        return True
    except Exception as e:
        print(f"Error creating DVC pipeline: {e}")
        return False


def run_dvc_pipeline(stage_name: Optional[str] = None, root_dir: str = ".") -> bool:
    """
    Run a DVC pipeline or specific stage.

    Args:
        stage_name: Optional specific stage to run
        root_dir: DVC repository root directory

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = ["dvc", "repro"]

        # Add specific stage if provided
        if stage_name:
            cmd.append(stage_name)

        # Execute the reproduce command
        subprocess.run(cmd, cwd=root_dir, check=True)

        what = f"stage '{stage_name}'" if stage_name else "all stages"
        print(f"Successfully reproduced {what} in the DVC pipeline")

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running DVC pipeline: {e}")
        return False
