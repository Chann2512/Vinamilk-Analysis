import os
import sys
import tempfile

def setup_environment():
    """
    Configures environment variables to:
    - Save local space on C drive by redirecting Hugging Face cache and temp directories to the E drive workspace.
    - Prevent connection timeouts/DNS blocking in Vietnam by using the hf-mirror endpoint.
    - Prevent TensorFlow-Keras compatibility warnings by forcing PyTorch-only mode.
    """
    # Define project root and local cache directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(project_root, ".cache", "huggingface")
    tmp_dir = os.path.join(project_root, ".cache", "tmp")
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Hugging Face cache redirection
    os.environ["HF_HOME"] = cache_dir
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    tempfile.tempdir = tmp_dir
    
    # Mirror and timeout options
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "300"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # Pure PyTorch Mode
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["USE_TF"] = "NO"

def ensure_project_dirs():
    """
    Verifies and creates all directories required by the project layout.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = [
        os.path.join(project_root, "data", "raw"),
        os.path.join(project_root, "data", "processed"),
        os.path.join(project_root, "notebooks"),
        os.path.join(project_root, "src"),
        os.path.join(project_root, "models", "phobert_sentiment"),
        os.path.join(project_root, "outputs", "figures"),
        os.path.join(project_root, "outputs", "reports"),
        os.path.join(project_root, "docs")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# Run environment setup immediately upon import to ensure configurations take effect
setup_environment()
ensure_project_dirs()
