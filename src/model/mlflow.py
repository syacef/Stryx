import mlflow
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# CONFIGURATION
TRACKING_URI = os.getenv("TRACKING_URI") 
SSL_EXPERIMENT = "wildlife-ssl-pretraining"
CLS_EXPERIMENT = "wildlife-species-classification"

logger = logging.getLogger(__name__)

class WildlifeTracker:
    def __init__(self, task_type="ssl"):
        """
        task_type: 'ssl' or 'classification'
        """
        mlflow.set_tracking_uri(TRACKING_URI)
        
        if task_type == "ssl":
            mlflow.set_experiment(SSL_EXPERIMENT)
        else:
            mlflow.set_experiment(CLS_EXPERIMENT)
            
        self.task_type = task_type

    def start_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name)

    def log_ssl_model(self, model, architecture="resnet18"):
        """Saves the Self-Supervised model and registers it directly."""
        logger.info(f"Logging SSL model ({architecture})...")
        
        mlflow.log_param("architecture", architecture)
        
        # log_model automatically handles saving and registration
        mlflow.pytorch.log_model(
            model, 
            name="ssl_model",
            registered_model_name="SSL_Foundation_Base"
        )

    def load_latest_ssl_model(self, version="latest"):
        """Helper for the Classification team to get the pretrained weights."""
        logger.info(f"Downloading SSL Base Model (Version: {version})...")
        model_uri = f"models:/SSL_Foundation_Base/{version}"
        return mlflow.pytorch.load_model(model_uri)

    def log_classifier(self, model, source_ssl_version, accuracy):
        """Saves the fine-tuned classifier and links it to the parent SSL model."""
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("source_ssl_version", source_ssl_version)
        mlflow.set_tag("parent_ssl_model", f"SSL_Foundation_Base_v{source_ssl_version}")
        
        logger.info(f"Logging classifier with accuracy {accuracy}...")

        # log_model automatically handles saving and registration
        mlflow.pytorch.log_model(
            model, 
            artifact_path="classifier",
            registered_model_name="Species_Classifier"
        )
