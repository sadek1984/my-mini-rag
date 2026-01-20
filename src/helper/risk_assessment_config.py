# """
# Risk Assessment Configuration
# Location: src/config/risk_assessment_config.py
# """

# from typing import Dict

# class RiskAssessmentConfig:
#     """Configuration for health risk assessment"""
    
#     # ADI values for pesticides (mg/kg/day) - WHO/FAO values
#     PESTICIDE_ADI: Dict[str, float] = {
#         "Buprofezin": 0.009,
#         "Fipronil": 0.0002,
#         "Chlorpyrifos": 0.001,
#         "Malathion": 0.3,
#         "Cypermethrin": 0.02,
#         "Imidacloprid": 0.057,
#         "Diazinon": 0.002,
#         "Carbaryl": 0.008,
#         "Parathion": 0.004,
#         "Dimethoate": 0.002,
#         "bifenthrin": 0.015,
#         "bifenazate": 0.01,
#         "pyridaben": 0.01,
#         "quinalphos": 0.01,
#     }
    
#     # Daily food consumption in Saudi Arabia (kg/day)
#     SAUDI_FOOD_INTAKE: Dict[str, float] = {
#         "tomato": 0.08,        # طماطم
#         "pepper": 0.05,        # فلفل
#         "cucumber": 0.06,      # خيار
#         "lettuce": 0.03,       # خس
#         "apple": 0.12,         # تفاح
#         "orange": 0.10,        # برتقال
#         "banana": 0.15,        # موز
#         "onion": 0.04,         # بصل
#         "potato": 0.20,        # بطاطس
#         "carrot": 0.06,        # جزر
#         "cabbage": 0.04,       # ملفوف
#         "grape": 0.08,         # عنب
#     }
    
#     # Body weights for Saudi population (kg)
#     SAUDI_BODY_WEIGHT: Dict[str, float] = {
#         "adult_male": 75,      # ذكر بالغ
#         "adult_female": 65,    # أنثى بالغة
#         "adult_average": 70,   # متوسط البالغين
#         "child": 25,           # طفل (5-12 سنة)
#         "teenager": 55,        # مراهق (13-17 سنة)
#         "elderly": 68,         # كبار السن
#     }
    
#     # Food name mapping (Arabic to English)
#     FOOD_NAME_MAPPING: Dict[str, str] = {
#         "طماطم": "tomato",
#         "فلفل": "pepper",
#         "فلفل حار": "pepper",
#         "فلفل بارد": "pepper",
#         "خيار": "cucumber",
#         "خس": "lettuce",
#         "تفاح": "apple",
#         "برتقال": "orange",
#         "موز": "banana",
#         "بصل": "onion",
#         "بطاطس": "potato",
#         "جزر": "carrot",
#         "ملفوف": "cabbage",
#         "عنب": "grape",
#         "باميا": "okra",
#         "ليمون": "lemon",
#         "باذنجان": "eggplant",
#         "كوسا": "zucchini",
#     }
    
#     # Risk level thresholds
#     RISK_THRESHOLDS = {
#         "acceptable": 1.0,      # EDI/ADI <= 1.0
#         "low_concern": 2.0,     # 1.0 < EDI/ADI <= 2.0
#         "moderate_concern": 5.0, # 2.0 < EDI/ADI <= 5.0
#         "high_concern": float('inf') # EDI/ADI > 5.0
#     }
    
#     @classmethod
#     def get_pesticide_adi(cls, pesticide_name: str) -> float:
#         """Get ADI value for a pesticide"""
#         return cls.PESTICIDE_ADI.get(pesticide_name, 0.01)  # Conservative default
    
#     @classmethod
#     def get_food_intake(cls, food_item: str) -> float:
#         """Get daily food intake for Saudi population"""
#         return cls.SAUDI_FOOD_INTAKE.get(food_item.lower(), 0.05)  # Default
    
#     @classmethod
#     def get_body_weight(cls, population_type: str) -> float:
#         """Get body weight for population type"""
#         return cls.SAUDI_BODY_WEIGHT.get(population_type, 70)  # Default adult
    
#     @classmethod
#     def map_food_name(cls, arabic_name: str) -> str:
#         """Map Arabic food name to English"""
#         return cls.FOOD_NAME_MAPPING.get(arabic_name.strip(), "unknown")



from typing import Dict, Any, Optional

class RiskAssessmentConfig:
    """
    Single source of truth for Health Risk Assessment parameters.
    Updated with standard Saudi national ingestion rates.
    """
    
    # Saudi Arabia Ingestion Rates (IR) in kg/day
    # Source: National Nutrition Surveys & SFDA
    SAUDI_INGESTION_RATES: Dict[str, float] = {
        "Total Vegetables": 0.1111,
        "Total Fruits": 0.0709,
        "Leafy Greens": 0.0165,
        "Tomato": 0.045,
        "Cucumber": 0.045,
        "Pepper": 0.045,
        "Root Vegetables": 0.025,
        "Eggplant": 0.030,
        "Zucchini": 0.025,
        "Beans": 0.020,
        "Okra": 0.015,
        "Squash": 0.020,
        "Cabbage": 0.020,
        "Lettuce": 0.015,
        "Parsley": 0.005,
        "Coriander": 0.005,
        "Mint": 0.003,
        "Default": 0.1111,
    }

    # Arabic to English commodity mapping
    COMMODITY_MAPPING: Dict[str, str] = {
        'طماطم': 'Tomato',
        'خيار': 'Cucumber',
        'فلفل': 'Pepper',
        'باذنجان': 'Eggplant',
        'كوسة': 'Zucchini',
        'فاصوليا': 'Beans',
        'بامية': 'Okra',
        'قرع': 'Squash',
        'ملفوف': 'Cabbage',
        'كرنب': 'Cabbage',
        'خس': 'Lettuce',
        'بقدونس': 'Parsley',
        'كزبرة': 'Coriander',
        'نعناع': 'Mint',
        'ورقيات': 'Leafy Greens',
        'خضروات': 'Total Vegetables',
        'فواكه': 'Total Fruits',
    }

    # Simplified Population Classes
    POPULATION_PROFILES: Dict[str, Dict[str, Any]] = {
        "adult": {
            "name_en": "Adult",
            "name_ar": "بالغ",
            "bw": 53.0
        },
        "child": {
            "name_en": "Child",
            "name_ar": "طفل",
            "bw": 16.0
        },
        # Legacy weights from old config
        "adult_male": {"name_en": "Adult Male", "name_ar": "ذكر بالغ", "bw": 75.0},
        "adult_female": {"name_en": "Adult Female", "name_ar": "أنثى بالغة", "bw": 65.0},
    }

    @classmethod
    def get_ingestion_rate(cls, commodity: str) -> float:
        """Get IR for commodity (handles Arabic and English)"""
        # 1. Try direct English lookup
        if commodity in cls.SAUDI_INGESTION_RATES:
            return cls.SAUDI_INGESTION_RATES[commodity]
        
        # 2. Try Arabic mapping
        mapped = cls.COMMODITY_MAPPING.get(commodity)
        if mapped and mapped in cls.SAUDI_INGESTION_RATES:
            return cls.SAUDI_INGESTION_RATES[mapped]
        
        # 3. Case-insensitive English
        comm_lower = commodity.lower()
        for key, val in cls.SAUDI_INGESTION_RATES.items():
            if key.lower() == comm_lower:
                return val
        
        return cls.SAUDI_INGESTION_RATES["Default"]

    @classmethod
    def get_pesticide_adi(cls, pesticide_name: str) -> Optional[float]:
        """
        Placeholder for ADI lookup. 
        Note: The actual ADI lookup usually happens via ADILookup class reading an Excel file.
        This provides a fallback for key pesticides.
        """
        FALLBACK_ADI = {
            "cypermethrin": 0.005,  # EFSA value (updated to match user expected HQ)
            "imidacloprid": 0.057,
            "chlorpyrifos": 0.001,
        }
        return FALLBACK_ADI.get(pesticide_name.lower())

    @classmethod
    def get_population_bw(cls, target: str) -> float:
        """Get body weight for profile or return default"""
        profile = cls.POPULATION_PROFILES.get(target.lower())
        if profile:
            return profile["bw"]
        return 53.0 # Default Saudi Adult