# src/models/pesticide_prediction_models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class PesticideSampleInput(BaseModel):
    """Input model for single pesticide sample prediction"""
    vegetable: str = Field(..., description="Vegetable type (e.g., tomato, pepper, cucumber)")
    pesticide_group: str = Field(..., description="Pesticide group classification")
    reading: float = Field(..., description="Device reading value", ge=0)
    limits: float = Field(..., description="Acceptable limits", gt=0)
    sample_code: Optional[str] = Field(None, description="Optional sample identifier")
    
    @validator('vegetable')
    def validate_vegetable(cls, v):
        if not v or not v.strip():
            raise ValueError('Vegetable name cannot be empty')
        return v.strip().lower()
    
    @validator('pesticide_group')
    def validate_pesticide_group(cls, v):
        valid_groups = [
            'pyrethroid', 'organophosphorus', 'chitin_synthesis_inhibitor',
            'mite_control', 'phenylpyrazole', 'triazole_fungicide',
            'juvenile_hormone_mimic', 'herbicide', 'dicarboximide_fungicide',
            'carbamate', 'gibberellin_inhibitor', 'other'
        ]
        if v not in valid_groups:
            raise ValueError(f'Invalid pesticide group. Must be one of: {valid_groups}')
        return v

class BatchPredictionInput(BaseModel):
    """Input model for batch pesticide sample predictions"""
    samples: List[PesticideSampleInput] = Field(..., description="List of samples to predict")
    
    @validator('samples')
    def validate_samples(cls, v):
        if not v:
            raise ValueError('At least one sample is required')
        if len(v) > 100:  # Limit batch size
            raise ValueError('Maximum 100 samples per batch')
        return v

class SampleInfo(BaseModel):
    """Sample information in prediction response"""
    sample_code: str
    vegetable: str
    pesticide_group: str
    reading: float
    limits: float
    exceedance_ratio: float

class RiskAssessment(BaseModel):
    """Risk assessment information"""
    risk_level: str
    risk_description: str
    exceedance_factor: float
    pesticide_group_risk: Dict[str, Any]

class PredictionResult(BaseModel):
    """Comprehensive prediction result"""
    success: bool
    sample_info: Optional[SampleInfo] = None
    predictions: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[RiskAssessment] = None
    recommendations: Optional[List[str]] = None
    timestamp: str
    model_version: Optional[str] = None
    error: Optional[str] = None

class BatchPredictionResult(BaseModel):
    """Batch prediction result"""
    success: bool
    total_samples: int
    successful_predictions: int
    failed_predictions: int
    results: List[PredictionResult]
    failures: List[Dict[str, Any]]
    timestamp: str

class ModelStatus(BaseModel):
    """Model status information"""
    model_loaded: bool
    model_path: str
    components_loaded: Dict[str, bool]
    timestamp: str

class PredictionIndexRequest(BaseModel):
    """Request to index prediction results into RAG system"""
    prediction_result: PredictionResult
    project_id: int
    include_in_rag: bool = Field(default=True, description="Whether to include in RAG system")
    metadata_tags: Optional[List[str]] = Field(default=None, description="Additional tags for categorization")