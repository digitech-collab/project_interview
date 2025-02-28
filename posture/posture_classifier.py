import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class PostureClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.feature_columns = None  # Will be set dynamically after training
    
    def prepare_data(self, data):
        """Prepare the dataset for training and prediction."""
        if self.feature_columns is None:
            self.feature_columns = [col for col in data.columns if 'posture' not in col]
        
        X = data[self.feature_columns]
        return X
    
    def train(self, data):
        """Train the RandomForest model using given dataset."""
        X = self.prepare_data(data)
        y = self.label_encoder.fit_transform(data['body_posture'])  # Encode target labels
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate performance
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions for classification report
        y_pred = self.model.predict(X_test)
        
        # Feature importances
        importances = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importances': importances,
            'classes': self.label_encoder.classes_
        }
    
    def predict(self, angles_dict):
        """Predict body posture based on input angles."""
        input_data = pd.DataFrame([angles_dict])
        
        missing_features = set(self.feature_columns) - set(angles_dict.keys())
        if missing_features:
            raise ValueError(f"Missing required angles: {missing_features}")
        
        prediction_encoded = self.model.predict(input_data[self.feature_columns])
        prediction = self.label_encoder.inverse_transform(prediction_encoded)
        
        probabilities = self.model.predict_proba(input_data[self.feature_columns])[0]
        prob_dict = dict(zip(self.label_encoder.classes_, probabilities))
        
        return {
            'predicted_category': prediction[0],
            'probabilities': prob_dict
        }
    
    def save_model(self, filepath):
        """Save trained model to a file."""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file."""
        model_data = joblib.load(filepath)
        
        classifier = cls()
        classifier.model = model_data['model']
        classifier.label_encoder = model_data['label_encoder']
        classifier.feature_columns = model_data['feature_columns']
        
        return classifier