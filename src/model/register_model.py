
# register model

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "shubhamkapase07"
repo_name = "Capstone"
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri(
#     'https://dagshub.com/shubhamkapase07/Capstone.mlflow'
# )

# dagshub.init(
#     repo_owner='shubhamkapase07',
#     repo_name='Capstone',
#     mlflow=True
# )
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load model information from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)

        logging.info('Model info loaded from %s', file_path)
        return model_info

    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise

    except Exception as e:
        logging.error(
            'Unexpected error occurred while loading model info: %s', e
        )
        raise


def register_model(model_name: str, model_info: dict):
    """Register the logged MLflow 3 model in the Model Registry."""

    try:
        # Read the actual model URI generated during model evaluation
        model_uri = model_info.get('model_uri')

        if not model_uri:
            raise ValueError(
                "model_uri not found in experiment_info.json. "
                "Run model_evaluation.py first."
            )

        print("Model URI:", model_uri)

        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        print(f"Successfully registered model: {model_name}")
        print(f"Model version: {model_version.version}")

        # Transition model version to Staging
        client = mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.info(
            'Model %s version %s registered and transitioned to Staging.',
            model_name,
            model_version.version
        )

        print("Model successfully transitioned to Staging.")

    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'

        model_info = load_model_info(model_info_path)

        model_name = "my_model"

        register_model(model_name, model_info)

    except Exception as e:
        logging.error(
            'Failed to complete the model registration process: %s', e
        )
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
