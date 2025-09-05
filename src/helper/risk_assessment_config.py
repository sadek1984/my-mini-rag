"""
Risk Assessment Configuration
Location: src/config/risk_assessment_config.py
"""

from typing import Dict

class RiskAssessmentConfig:
    """Configuration for health risk assessment"""
    
    # ADI values for pesticides (mg/kg/day) - WHO/FAO values
    PESTICIDE_ADI: Dict[str, float] = {
        "Buprofezin": 0.02,
        "Fipronil": 0.0002,
        "Chlorpyrifos": 0.001,
        "Malathion": 0.3,
        "Cypermethrin": 0.02,
        "Imidacloprid": 0.057,
        "Diazinon": 0.002,
        "Carbaryl": 0.008,
        "Parathion": 0.004,
        "Dimethoate": 0.002,
    }
    
    # Daily food consumption in Saudi Arabia (kg/day)
    SAUDI_FOOD_INTAKE: Dict[str, float] = {
        "tomato": 0.08,        # طماطم
        "pepper": 0.05,        # فلفل
        "cucumber": 0.06,      # خيار
        "lettuce": 0.03,       # خس
        "apple": 0.12,         # تفاح
        "orange": 0.10,        # برتقال
        "banana": 0.15,        # موز
        "onion": 0.04,         # بصل
        "potato": 0.20,        # بطاطس
        "carrot": 0.06,        # جزر
        "cabbage": 0.04,       # ملفوف
        "grape": 0.08,         # عنب
    }
    
    # Body weights for Saudi population (kg)
    SAUDI_BODY_WEIGHT: Dict[str, float] = {
        "adult_male": 75,      # ذكر بالغ
        "adult_female": 65,    # أنثى بالغة
        "adult_average": 70,   # متوسط البالغين
        "child": 25,           # طفل (5-12 سنة)
        "teenager": 55,        # مراهق (13-17 سنة)
        "elderly": 68,         # كبار السن
    }
    
    # Food name mapping (Arabic to English)
    FOOD_NAME_MAPPING: Dict[str, str] = {
        "طماطم": "tomato",
        "فلفل": "pepper",
        "فلفل حار": "pepper",
        "فلفل بارد": "pepper",
        "خيار": "cucumber",
        "خس": "lettuce",
        "تفاح": "apple",
        "برتقال": "orange",
        "موز": "banana",
        "بصل": "onion",
        "بطاطس": "potato",
        "جزر": "carrot",
        "ملفوف": "cabbage",
        "عنب": "grape",
        "باميا": "okra",
        "ليمون": "lemon",
        "باذنجان": "eggplant",
        "كوسا": "zucchini",
    }
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        "acceptable": 1.0,      # EDI/ADI <= 1.0
        "low_concern": 2.0,     # 1.0 < EDI/ADI <= 2.0
        "moderate_concern": 5.0, # 2.0 < EDI/ADI <= 5.0
        "high_concern": float('inf') # EDI/ADI > 5.0
    }
    
    @classmethod
    def get_pesticide_adi(cls, pesticide_name: str) -> float:
        """Get ADI value for a pesticide"""
        return cls.PESTICIDE_ADI.get(pesticide_name, 0.01)  # Conservative default
    
    @classmethod
    def get_food_intake(cls, food_item: str) -> float:
        """Get daily food intake for Saudi population"""
        return cls.SAUDI_FOOD_INTAKE.get(food_item.lower(), 0.05)  # Default
    
    @classmethod
    def get_body_weight(cls, population_type: str) -> float:
        """Get body weight for population type"""
        return cls.SAUDI_BODY_WEIGHT.get(population_type, 70)  # Default adult
    
    @classmethod
    def map_food_name(cls, arabic_name: str) -> str:
        """Map Arabic food name to English"""
        return cls.FOOD_NAME_MAPPING.get(arabic_name.strip(), "unknown")