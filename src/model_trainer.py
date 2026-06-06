import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

class ModelTrainer:
    """
    Handles machine learning pipeline: training, tuning, evaluating,
    and saving model checkpoints for software defect prediction.
    """
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42, **kwargs):
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.features = []
        self.model_params = kwargs
        
    def _initialize_model(self, scale_pos_weight: float = 1.0):
        """Initializes the model with specified type and parameters."""
        params = self.model_params.copy()
        
        if self.model_type == 'random_forest':
            if 'class_weight' not in params:
                params['class_weight'] = 'balanced'
            self.model = RandomForestClassifier(random_state=self.random_state, **params)
            
        elif self.model_type == 'logistic_regression':
            if 'class_weight' not in params:
                params['class_weight'] = 'balanced'
            if 'max_iter' not in params:
                params['max_iter'] = 1000
            self.model = LogisticRegression(random_state=self.random_state, **params)
            
        elif self.model_type == 'xgboost':
            if 'scale_pos_weight' not in params:
                params['scale_pos_weight'] = scale_pos_weight
            if 'eval_metric' not in params:
                params['eval_metric'] = 'logloss'
            self.model = XGBClassifier(random_state=self.random_state, **params)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, df: pd.DataFrame, feature_cols: list, target_col: str, test_size: float = 0.2):
        """
        Runs the full training pipeline: train-test split, scaling, training, and evaluation.
        """
        self.features = list(feature_cols)
        
        # Split features and target
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Fit scaler on training set
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate scale_pos_weight for XGBoost to handle class imbalance
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / max(1, pos_count)
        
        # Init and fit model
        self._initialize_model(scale_pos_weight=scale_pos_weight)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_prob = None
        
        # Try to get prediction probabilities
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(self.model, "decision_function"):
            y_prob = self.model.decision_function(X_test_scaled)
            
        metrics = self._compute_metrics(y_test, y_pred, y_prob)
        
        # Compile evaluation data (ROC/PR curves, confusion matrix)
        eval_plots = self._generate_plot_data(y_test, y_prob)
        
        return metrics, eval_plots

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts labels for a given DataFrame.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Load or train a model first.")
        X = df[self.features].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts probabilities for a given DataFrame.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Load or train a model first.")
        X = df[self.features].values
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)[:, 1]
        elif hasattr(self.model, "decision_function"):
            # Normalize to 0-1 range
            scores = self.model.decision_function(X_scaled)
            return 1 / (1 + np.exp(-scores))
        else:
            return self.model.predict(X_scaled).astype(float)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
        """Computes key classification metrics."""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
        
        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc)
        }

    def _generate_plot_data(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        """Generates coordinate data for curves and confusion matrix."""
        # Confusion matrix
        y_pred = (y_prob >= 0.5).astype(int) if y_prob is not None else y_true
        cm = confusion_matrix(y_true, y_pred)
        
        plot_data = {
            "confusion_matrix": cm.tolist()
        }
        
        # ROC Curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plot_data["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            }
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            plot_data["pr_curve"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist()
            }
            
        return plot_data

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extracts feature importances or coefficients depending on the model type.
        """
        if self.model is None:
            raise ValueError("Model is not trained.")
            
        importances = []
        if self.model_type in ('random_forest', 'xgboost'):
            importances = self.model.feature_importances_
        elif self.model_type == 'logistic_regression':
            importances = np.abs(self.model.coef_[0])
            
        df_imp = pd.DataFrame({
            'feature': self.features,
            'importance': importances
        })
        
        return df_imp.sort_values(by='importance', ascending=False).reset_index(drop=True)

    def save(self, filepath: str):
        """Saves the trained model, scaler, and features list."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model_type": self.model_type,
            "model": self.model,
            "scaler": self.scaler,
            "features": self.features,
            "model_params": self.model_params
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Loads a model from checkpoint and reconstructs ModelTrainer object."""
        checkpoint = joblib.load(filepath)
        trainer = cls(
            model_type=checkpoint["model_type"],
            **checkpoint["model_params"]
        )
        trainer.model = checkpoint["model"]
        trainer.scaler = checkpoint["scaler"]
        trainer.features = checkpoint["features"]
        return trainer
