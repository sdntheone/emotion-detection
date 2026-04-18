import numpy as np
import pandas as pd
import pickle
import json
import os

from typing import Tuple, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.base import ClassifierMixin
from src.logging.logging import get_logger

logger = get_logger(__name__)


def load_model(path: str) -> ClassifierMixin:
    try:
        logger.info(f"Loading model from: {path}")

        with open(path, 'rb') as f:
            clf = pickle.load(f)

        logger.info("Model loaded successfully")
        return clf

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise

    except pickle.UnpicklingError as e:
        logger.error(f"Error loading model: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in load_model: {e}")
        raise


def load_test_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        logger.info(f"Loading test data from: {data_path}")

        test_data = pd.read_csv(data_path)

        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values

        logger.debug(f"Test shape: {test_data.shape}")
        return X_test, y_test

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty CSV file: {e}")
        raise

    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in load_test_data: {e}")
        raise


def prediction(clf: ClassifierMixin, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        logger.info("Generating predictions")

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        return y_pred, y_pred_proba

    except AttributeError as e:
        logger.error(f"Model not fitted or invalid: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in prediction: {e}")
        raise


def evaluation_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:

    try:
        logger.info("Calculating evaluation metrics")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict: Dict[str, float] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        logger.debug(f"Metrics: {metrics_dict}")
        return metrics_dict

    except ValueError as e:
        logger.error(f"Metrics calculation error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in evaluation_metrics: {e}")
        raise


def save_metrics(metrics_dict: Dict[str, float]) -> None:
    try:
        logger.info("Saving metrics to metrics.json")

        with open('metrics.json', 'w') as file:
            json.dump(metrics_dict, file, indent=4)

        logger.debug("Metrics saved successfully")

    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in save_metrics: {e}")
        raise


def main() -> None:
    try:
        logger.info("Evaluation pipeline started")

        model_path = 'model.pkl'
        test_data_path = os.path.join("data", "features", "test_tfidf.csv")

        clf = load_model(model_path)

        X_test, y_test = load_test_data(test_data_path)

        y_pred, y_pred_proba = prediction(clf, X_test)

        metrics_dict = evaluation_metrics(y_test, y_pred, y_pred_proba)

        save_metrics(metrics_dict)

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()