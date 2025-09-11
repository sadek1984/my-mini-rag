# # src/services/pesticide_prediction_service.py
# import logging
# import joblib
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from typing import Dict, Any, Optional
# from datetime import datetime

# from helper.config import get_settings

# class PesticidePredictionService:
#     """
#     Production-ready service for pesticide classification predictions
#     Wraps the SingleDatasetPesticideClassifier for API use
#     """
    
#     def __init__(self, model_path: str = None):
#         self.logger = logging.getLogger(__name__)
#         self.settings = get_settings()
#         self.classifier = None
#         self.model_loaded = False
#         self.model_path = model_path or "single_dataset_system/trained_models"
        
#         # Load the trained model
#         self._load_model()
    
#     def _load_model(self):
#         """Load the trained classifier and preprocessing components"""
#         try:
#             model_dir = Path(self.model_path)
            
#             if not model_dir.exists():
#                 self.logger.error(f"Model directory not found: {model_dir}")
#                 return False
            
#             # Import the classifier class
#             from models.PesticideClassificationModel import PesticideClassificationModel
            
#             # Initialize classifier
#             self.classifier = PesticideClassificationModel()
            
#             # Load saved components
#             try:
#                 self.classifier.scaler = joblib.load(model_dir / "scaler.pkl")
#                 self.classifier.group_label_encoder = joblib.load(model_dir / "group_label_encoder.pkl")
#                 self.classifier.onehot_encoders = joblib.load(model_dir / "onehot_encoders.pkl")
                
#                 # Load models
#                 self.classifier.compliance_model = joblib.load(model_dir / "compliance_model.pkl")
#                 self.classifier.concentration_predictor = joblib.load(model_dir / "concentration_predictor.pkl")
                
#                 # Load feature imputer if exists
#                 imputer_path = model_dir / "feature_imputer.pkl"
#                 if imputer_path.exists():
#                     self.classifier.feature_imputer = joblib.load(imputer_path)
                
#                 # Load TensorFlow models
#                 try:
#                     import tensorflow as tf
#                     group_model_path = model_dir / "group_classifier.keras"
#                     if group_model_path.exists():
#                         self.classifier.group_classifier = tf.keras.models.load_model(group_model_path)
#                     else:
#                         # Fallback to pickle
#                         self.classifier.group_classifier = joblib.load(model_dir / "group_classifier.pkl")
                    
#                     risk_model_path = model_dir / "risk_classifier.keras"
#                     if risk_model_path.exists():
#                         self.classifier.risk_classifier = tf.keras.models.load_model(risk_model_path)
                        
#                 except Exception as tf_e:
#                     self.logger.warning(f"TensorFlow model loading failed: {tf_e}")
                
#                 self.model_loaded = True
#                 self.logger.info("Pesticide prediction models loaded successfully")
#                 return True
                
#             except Exception as e:
#                 self.logger.error(f"Error loading model components: {e}")
#                 return False
                
#         except Exception as e:
#             self.logger.error(f"Failed to load prediction model: {e}")
#             return False
    
#     def predict_sample_compliance(self, 
#                                 vegetable: str,
#                                 pesticide_group: str,
#                                 reading: float,
#                                 limits: float,
#                                 sample_code: str = None) -> Dict[str, Any]:
#         """
#         Predict compliance and risk for a single sample
        
#         Args:
#             vegetable: Vegetable type (e.g., 'tomato', 'pepper')
#             pesticide_group: Pesticide group (e.g., 'pyrethroid', 'organophosphorus')
#             reading: Device reading value
#             limits: Acceptable limits
#             sample_code: Optional sample identifier
            
#         Returns:
#             Dictionary with prediction results and metadata
#         """
        
#         if not self.model_loaded:
#             return {
#                 'success': False,
#                 'error': 'Prediction model not loaded',
#                 'timestamp': datetime.now().isoformat()
#             }
        
#         try:
#             # Validate inputs
#             validation_result = self._validate_inputs(vegetable, pesticide_group, reading, limits)
#             if not validation_result['valid']:
#                 return {
#                     'success': False,
#                     'error': validation_result['error'],
#                     'timestamp': datetime.now().isoformat()
#                 }
            
#             # Get predictions from classifier
#             prediction_result = self.classifier.predict_comprehensive_analysis(
#                 vegetable=vegetable,
#                 pesticide_group=pesticide_group,
#                 reading=reading,
#                 limits=limits
#             )
            
#             if 'error' in prediction_result:
#                 return {
#                     'success': False,
#                     'error': prediction_result['error'],
#                     'timestamp': datetime.now().isoformat()
#                 }
            
#             # Calculate additional metrics
#             exceedance_ratio = reading / limits if limits > 0 else float('inf')
            
#             # Generate comprehensive response
#             response = {
#                 'success': True,
#                 'sample_info': {
#                     'sample_code': sample_code or f"PRED_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#                     'vegetable': vegetable,
#                     'pesticide_group': pesticide_group,
#                     'reading': reading,
#                     'limits': limits,
#                     'exceedance_ratio': round(exceedance_ratio, 3)
#                 },
#                 'predictions': prediction_result,
#                 'risk_assessment': self._assess_risk_level(exceedance_ratio, pesticide_group),
#                 'recommendations': self._generate_recommendations(prediction_result, vegetable, pesticide_group, exceedance_ratio),
#                 'timestamp': datetime.now().isoformat(),
#                 'model_version': '1.0'
#             }
            
#             return response
            
#         except Exception as e:
#             self.logger.error(f"Prediction failed: {e}")
#             return {
#                 'success': False,
#                 'error': f'Prediction error: {str(e)}',
#                 'timestamp': datetime.now().isoformat()
#             }
    
#     def predict_batch_samples(self, samples: list) -> Dict[str, Any]:
#         """
#         Predict multiple samples at once
        
#         Args:
#             samples: List of sample dictionaries with required fields
            
#         Returns:
#             Dictionary with batch prediction results
#         """
        
#         if not self.model_loaded:
#             return {
#                 'success': False,
#                 'error': 'Prediction model not loaded'
#             }
        
#         results = []
#         failed_predictions = []
        
#         for i, sample in enumerate(samples):
#             try:
#                 result = self.predict_sample_compliance(
#                     vegetable=sample.get('vegetable'),
#                     pesticide_group=sample.get('pesticide_group'),
#                     reading=sample.get('reading'),
#                     limits=sample.get('limits'),
#                     sample_code=sample.get('sample_code', f'BATCH_{i+1}')
#                 )
                
#                 if result['success']:
#                     results.append(result)
#                 else:
#                     failed_predictions.append({'index': i, 'error': result['error']})
                    
#             except Exception as e:
#                 failed_predictions.append({'index': i, 'error': str(e)})
        
#         return {
#             'success': True,
#             'total_samples': len(samples),
#             'successful_predictions': len(results),
#             'failed_predictions': len(failed_predictions),
#             'results': results,
#             'failures': failed_predictions,
#             'timestamp': datetime.now().isoformat()
#         }
    
#     def _validate_inputs(self, vegetable: str, pesticide_group: str, reading: float, limits: float) -> Dict[str, Any]:
#         """Validate input parameters"""
        
#         if not vegetable or not isinstance(vegetable, str):
#             return {'valid': False, 'error': 'Invalid vegetable parameter'}
        
#         if not pesticide_group or not isinstance(pesticide_group, str):
#             return {'valid': False, 'error': 'Invalid pesticide_group parameter'}
        
#         try:
#             reading = float(reading)
#             limits = float(limits)
#         except (ValueError, TypeError):
#             return {'valid': False, 'error': 'Reading and limits must be numeric'}
        
#         if reading < 0 or limits <= 0:
#             return {'valid': False, 'error': 'Reading must be non-negative and limits must be positive'}
        
#         return {'valid': True}
    
#     def _assess_risk_level(self, exceedance_ratio: float, pesticide_group: str) -> Dict[str, Any]:
#         """Assess risk level based on exceedance ratio and pesticide group"""
        
#         if exceedance_ratio <= 1.0:
#             risk_level = 'Low'
#             risk_description = 'Within acceptable limits'
#         elif exceedance_ratio <= 2.0:
#             risk_level = 'Medium'
#             risk_description = 'Slightly above limits'
#         elif exceedance_ratio <= 5.0:
#             risk_level = 'High'
#             risk_description = 'Significantly above limits'
#         else:
#             risk_level = 'Critical'
#             risk_description = 'Critically above limits'
        
#         return {
#             'risk_level': risk_level,
#             'risk_description': risk_description,
#             'exceedance_factor': round(exceedance_ratio, 2),
#             'pesticide_group_risk': self.classifier.group_risk_profiles.get(pesticide_group, {})
#         }
    
#     def _generate_recommendations(self, prediction_result: Dict, vegetable: str, 
#                                 pesticide_group: str, exceedance_ratio: float) -> list:
#         """Generate actionable recommendations"""
        
#         recommendations = []
        
#         # Compliance-based recommendations
#         if 'compliance' in prediction_result:
#             compliance_prob = prediction_result['compliance'].get('probability_compliant', 0)
#             if compliance_prob < 0.5:
#                 recommendations.append("Sample likely non-compliant - requires immediate attention")
#                 recommendations.append("Consider retesting to confirm results")
        
#         # Risk-based recommendations
#         if exceedance_ratio > 5.0:
#             recommendations.append("Critical contamination level - investigate source immediately")
#             recommendations.append("Implement enhanced cleaning protocols")
#         elif exceedance_ratio > 2.0:
#             recommendations.append("Elevated pesticide levels detected - monitor closely")
        
#         # Vegetable-specific recommendations
#         if 'pepper' in vegetable.lower():
#             recommendations.append("Peppers show high contamination rates - enhance washing protocols")
#         elif 'leafy' in self.classifier.categorize_vegetable(vegetable):
#             recommendations.append("Leafy greens require special attention - difficult to decontaminate")
        
#         # Pesticide group recommendations
#         if pesticide_group in self.classifier.group_risk_profiles:
#             group_risks = self.classifier.group_risk_profiles[pesticide_group]
#             for risk_type, level in group_risks.items():
#                 if level == 'high':
#                     recommendations.append(f"Monitor for {risk_type} effects due to {pesticide_group} exposure")
        
#         return recommendations if recommendations else ["Sample appears normal - continue regular monitoring"]
    
#     def get_model_status(self) -> Dict[str, Any]:
#         """Get current model status and metadata"""
        
#         return {
#             'model_loaded': self.model_loaded,
#             'model_path': str(self.model_path),
#             'components_loaded': {
#                 'scaler': hasattr(self.classifier, 'scaler') and self.classifier.scaler is not None,
#                 'encoders': hasattr(self.classifier, 'onehot_encoders') and bool(self.classifier.onehot_encoders),
#                 'compliance_model': hasattr(self.classifier, 'compliance_model') and self.classifier.compliance_model is not None,
#                 'group_classifier': hasattr(self.classifier, 'group_classifier') and self.classifier.group_classifier is not None,
#             } if self.classifier else {},
#             'timestamp': datetime.now().isoformat()
#         }



# services/pesticide_prediction_service.py
import logging
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class PesticidePredictionService:
    """
    Pesticide prediction service 
    """
    
    def __init__(self, model_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.model_loaded = True  # Set to True for API testing
        self.model_path = model_path or "trained_models"
        self.model_version = "1.0.0"
        
        # Pesticide group health risk profiles
        self.group_risk_profiles = {
            'pyrethroid': {'neurotoxicity': 'medium', 'skin_irritation': 'high', 'persistence': 'medium'},
            'organophosphorus': {'neurotoxicity': 'high', 'cholinesterase': 'high', 'persistence': 'low'},
            'chitin_synthesis_inhibitor': {'toxicity': 'low', 'endocrine': 'medium', 'persistence': 'high'},
            'mite_control': {'neurotoxicity': 'medium', 'dermal': 'medium', 'persistence': 'medium'},
            'phenylpyrazole': {'neurotoxicity': 'high', 'bioaccumulation': 'high', 'persistence': 'high'}
        }
        
        self.logger.info("PesticidePredictionService initialized successfully")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return {
            "model_loaded": self.model_loaded,
            "model_path": str(self.model_path),
            "components_loaded": {
                'scaler': True,
                'encoders': True,
                'compliance_model': True,
                'group_classifier': True,
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def predict_sample_compliance(
        self, 
        vegetable: str, 
        pesticide_group: str, 
        reading: float, 
        limits: float, 
        sample_code: str = None
    ) -> Dict[str, Any]:
        """
        Predict compliance for a single sample
        """
        try:
            # Validate inputs
            validation_result = self._validate_inputs(vegetable, pesticide_group, reading, limits)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate basic metrics
            exceedance_ratio = reading / limits if limits > 0 else float('inf')
            is_compliant = reading <= limits
            
            # Mock prediction logic (replace with actual model)
            compliance_probability = max(0.0, min(1.0, 1.0 - (exceedance_ratio - 1.0)))
            
            # Risk assessment logic
            risk_assessment = self._assess_risk_level(exceedance_ratio, pesticide_group)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(is_compliant, exceedance_ratio, vegetable, pesticide_group)
            
            # Build result
            result = {
                "success": True,
                "sample_info": {
                    "sample_code": sample_code or f"AUTO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "vegetable": vegetable,
                    "pesticide_group": pesticide_group,
                    "reading": reading,
                    "limits": limits,
                    "exceedance_ratio": round(exceedance_ratio, 3)
                },
                "predictions": {
                    "compliance": {
                        "prediction": "Compliant" if is_compliant else "Non-Compliant",
                        "probability_compliant": round(compliance_probability, 3),
                        "confidence": 0.85  # Mock confidence
                    }
                },
                "risk_assessment": risk_assessment,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat(),
                "model_version": self.model_version
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def predict_batch_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict compliance for multiple samples
        """
        try:
            results = []
            successful = 0
            failed = 0
            failures = []
            
            for i, sample in enumerate(samples):
                result = self.predict_sample_compliance(
                    vegetable=sample.get('vegetable'),
                    pesticide_group=sample.get('pesticide_group'),
                    reading=sample.get('reading'),
                    limits=sample.get('limits'),
                    sample_code=sample.get('sample_code')
                )
                
                results.append(result)
                
                if result.get('success'):
                    successful += 1
                else:
                    failed += 1
                    failures.append({'index': i, 'error': result.get('error')})
            
            return {
                "success": True,
                "total_samples": len(samples),
                "successful_predictions": successful,
                "failed_predictions": failed,
                "results": results,
                "failures": failures,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_inputs(self, vegetable: str, pesticide_group: str, reading: float, limits: float) -> Dict[str, Any]:
        """Validate input parameters"""
        
        if not vegetable or not isinstance(vegetable, str):
            return {'valid': False, 'error': 'Invalid vegetable parameter'}
        
        if not pesticide_group or not isinstance(pesticide_group, str):
            return {'valid': False, 'error': 'Invalid pesticide_group parameter'}
        
        try:
            reading = float(reading)
            limits = float(limits)
        except (ValueError, TypeError):
            return {'valid': False, 'error': 'Reading and limits must be numeric'}
        
        if reading < 0 or limits <= 0:
            return {'valid': False, 'error': 'Reading must be non-negative and limits must be positive'}
        
        return {'valid': True}
    
    def _assess_risk_level(self, exceedance_ratio: float, pesticide_group: str) -> Dict[str, Any]:
        """Assess risk level based on exceedance ratio and pesticide group"""
        
        if exceedance_ratio <= 1.0:
            risk_level = 'Low'
            risk_description = 'Within acceptable limits'
        elif exceedance_ratio <= 2.0:
            risk_level = 'Medium'
            risk_description = 'Slightly above limits'
        elif exceedance_ratio <= 5.0:
            risk_level = 'High'
            risk_description = 'Significantly above limits'
        else:
            risk_level = 'Critical'
            risk_description = 'Critically above limits'
        
        return {
            'risk_level': risk_level,
            'risk_description': risk_description,
            'exceedance_factor': round(exceedance_ratio, 2),
            'pesticide_group_risk': self.group_risk_profiles.get(pesticide_group, {})
        }
    
    def _generate_recommendations(self, is_compliant: bool, exceedance_ratio: float, vegetable: str, pesticide_group: str) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Compliance-based recommendations
        if not is_compliant:
            recommendations.append("Sample does not meet compliance standards")
            recommendations.append("Consider retesting or investigating source")
        
        # Risk-based recommendations
        if exceedance_ratio > 5.0:
            recommendations.append("Critical contamination level - investigate source immediately")
            recommendations.append("Implement enhanced cleaning protocols")
        elif exceedance_ratio > 2.0:
            recommendations.append("Elevated pesticide levels detected - monitor closely")
        
        # Vegetable-specific recommendations
        if 'pepper' in vegetable.lower():
            recommendations.append("Peppers show high contamination rates - enhance washing protocols")
        elif any(term in vegetable.lower() for term in ['lettuce', 'spinach', 'leafy']):
            recommendations.append("Leafy greens require special attention - difficult to decontaminate")
        
        # Pesticide group recommendations
        if pesticide_group in self.group_risk_profiles:
            group_risks = self.group_risk_profiles[pesticide_group]
            for risk_type, level in group_risks.items():
                if level == 'high':
                    recommendations.append(f"Monitor for {risk_type} effects due to {pesticide_group} exposure")
        
        return recommendations if recommendations else ["Sample appears normal - continue regular monitoring"]