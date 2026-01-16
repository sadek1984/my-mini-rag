"""
Risk Assessment Service for LARS Application

This module provides comprehensive dietary exposure and risk assessment calculations
based on ADI (Acceptable Daily Intake) values and MRL (Maximum Residue Levels).

Implements PRIMo 4 methodology for:
- EDI (Estimated Daily Intake)
- HQc (Hazard Quotient chronic)
- HIc (Hazard Index chronic) by chemical group

Author: LARS Team
"""

import pandas as pd
import numpy as np
import json
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from functools import lru_cache


# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the directory where this module is located
MODULE_DIR = Path(__file__).parent
SCRIPTS_DIR = MODULE_DIR.parent / 'scripts'

# Default paths
DEFAULT_ADI_FILE = SCRIPTS_DIR / 'pesticide_adi_simple_20260112_221046.xlsx'
CHEMICAL_CLASSIFICATION_FILE = SCRIPTS_DIR / 'chemical_classification.json'

# EU MRL API endpoint (unofficial - with fallback to offline data)
EU_MRL_API_BASE = "https://ec.europa.eu/food/api/pesticides/mrl"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PopulationClass:
    """Represents a population class with body weight for risk calculations."""
    key: str
    name_en: str
    name_ar: str
    age_range: str
    default_weight_kg: float


@dataclass
class ChemicalGroup:
    """Represents a chemical group/class of pesticides."""
    key: str
    name_en: str
    name_ar: str
    pesticides: List[str]


@dataclass
class RiskResult:
    """Result of a risk assessment calculation."""
    pesticide: str
    concentration: float
    adi: Optional[float]
    edi: Optional[float]
    hqc: Optional[float]
    mrl: Optional[float]
    mrl_ratio: Optional[float]
    chemical_group: Optional[str]
    interpretation: str
    unit: str = "mg/kg bw/day"


# ============================================================================
# POPULATION CLASSES (PRIMo 4)
# ============================================================================

POPULATION_CLASSES = {
    'infants': PopulationClass('infants', 'Infants', 'رضع', '<1', 8.0),
    'toddlers': PopulationClass('toddlers', 'Toddlers', 'صغار الأطفال', '1-<3', 12.0),
    'other_children': PopulationClass('other_children', 'Other Children', 'أطفال آخرون', '3-<10', 23.0),
    'adolescents': PopulationClass('adolescents', 'Adolescents', 'مراهقون', '10-<18', 53.0),
    'adults': PopulationClass('adults', 'Adults', 'بالغون', '18-<65', 70.0),
    'elderly': PopulationClass('elderly', 'Elderly', 'كبار السن', '65-<75', 68.0),
    'very_elderly': PopulationClass('very_elderly', 'Very Elderly', 'كبار السن جداً', '>=75', 65.0),
    'pregnant_women': PopulationClass('pregnant_women', 'Pregnant Women', 'نساء حوامل', '15-45', 65.0),
    'lactating_women': PopulationClass('lactating_women', 'Lactating Women', 'نساء مرضعات', '28-39', 65.0),
    # Saudi-specific population classes
    'saudi_adults': PopulationClass('saudi_adults', 'Adults (Saudi)', 'بالغون (سعودي)', '18+', 53.0),
    'saudi_children': PopulationClass('saudi_children', 'Children 2-6 (Saudi)', 'أطفال 2-6 (سعودي)', '2-6', 16.0),
}


# ============================================================================
# SAUDI ARABIA INGESTION RATES (IR)
# Sources: National Nutrition Surveys & SFDA
# ============================================================================

SAUDI_IR_DATA = {
    # Commodity name (English): IR in kg/day
    "Total Vegetables": 0.1111,    # 111.1 g/day
    "Total Fruits": 0.0709,        # 70.9 g/day
    "Leafy Greens": 0.0165,        # 16.5 g/day
    "Tomato": 0.045,               # 45.0 g/day
    "Cucumber": 0.045,             # 45.0 g/day
    "Pepper": 0.045,               # 45.0 g/day
    "Root Vegetables": 0.025,      # 25.0 g/day
    "Eggplant": 0.030,             # 30.0 g/day
    "Zucchini": 0.025,             # 25.0 g/day
    "Beans": 0.020,                # 20.0 g/day
    "Okra": 0.015,                 # 15.0 g/day
    "Squash": 0.020,               # 20.0 g/day
    "Cabbage": 0.020,              # 20.0 g/day
    "Lettuce": 0.015,              # 15.0 g/day
    "Parsley": 0.005,              # 5.0 g/day
    "Coriander": 0.005,            # 5.0 g/day
    "Mint": 0.003,                 # 3.0 g/day
    "Default": 0.1111,             # Default to total vegetables
}

# Arabic to English commodity mapping for Saudi IR lookup
SAUDI_COMMODITY_MAPPING = {
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


def get_saudi_ir(commodity: str) -> float:
    """
    Get Saudi ingestion rate for a commodity in kg/day.
    
    Parameters:
        commodity: Commodity name in English or Arabic
    
    Returns:
        Ingestion rate in kg/day
    """
    # Try direct English lookup
    if commodity in SAUDI_IR_DATA:
        return SAUDI_IR_DATA[commodity]
    
    # Try Arabic mapping
    mapped = SAUDI_COMMODITY_MAPPING.get(commodity)
    if mapped and mapped in SAUDI_IR_DATA:
        return SAUDI_IR_DATA[mapped]
    
    # Try case-insensitive match
    commodity_lower = commodity.lower()
    for key, value in SAUDI_IR_DATA.items():
        if key.lower() == commodity_lower:
            return value
    
    # Default
    return SAUDI_IR_DATA["Default"]


def get_population_weight(popclass: str) -> float:
    """Get default body weight for a population class."""
    return POPULATION_CLASSES.get(popclass, POPULATION_CLASSES['adults']).default_weight_kg


def get_population_options() -> List[Dict[str, str]]:
    """Get list of population options for dropdown."""
    return [
        {
            'key': pc.key,
            'label': f"{pc.name_en} ({pc.name_ar}) - {pc.default_weight_kg} kg",
            'name_en': pc.name_en,
            'name_ar': pc.name_ar,
            'weight': pc.default_weight_kg
        }
        for pc in POPULATION_CLASSES.values()
    ]


# ============================================================================
# CHEMICAL CLASSIFICATION
# ============================================================================

class ChemicalClassifier:
    """Classifier for pesticides by chemical group."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with configuration file."""
        self.config_path = config_path or CHEMICAL_CLASSIFICATION_FILE
        self._groups: Dict[str, ChemicalGroup] = {}
        self._pesticide_to_group: Dict[str, str] = {}
        self._load_configuration()
    
    def _load_configuration(self):
        """Load chemical classification from JSON file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                for key, data in config.get('chemical_groups', {}).items():
                    group = ChemicalGroup(
                        key=key,
                        name_en=data.get('name_en', key),
                        name_ar=data.get('name_ar', key),
                        pesticides=data.get('pesticides', [])
                    )
                    self._groups[key] = group
                    
                    # Build reverse lookup
                    for pesticide in group.pesticides:
                        self._pesticide_to_group[pesticide.lower()] = key
        except Exception as e:
            print(f"Warning: Could not load chemical classification: {e}")
    
    def get_group(self, pesticide_name: str) -> Optional[str]:
        """Get chemical group for a pesticide."""
        return self._pesticide_to_group.get(pesticide_name.lower().strip())
    
    def get_group_info(self, group_key: str) -> Optional[ChemicalGroup]:
        """Get information about a chemical group."""
        return self._groups.get(group_key)
    
    def get_all_groups(self) -> Dict[str, ChemicalGroup]:
        """Get all chemical groups."""
        return self._groups
    
    def classify_pesticides(self, pesticide_list: List[str]) -> Dict[str, List[str]]:
        """Classify a list of pesticides by chemical group."""
        result = {}
        for pesticide in pesticide_list:
            group = self.get_group(pesticide)
            if group:
                if group not in result:
                    result[group] = []
                result[group].append(pesticide)
            else:
                if 'unknown' not in result:
                    result['unknown'] = []
                result['unknown'].append(pesticide)
        return result


# ============================================================================
# ADI LOOKUP
# ============================================================================

class ADILookup:
    """ADI (Acceptable Daily Intake) lookup module for LARS application."""
    
    def __init__(self, adi_file_path: Optional[Path] = None):
        """Initialize with ADI data file."""
        self.adi_file_path = adi_file_path or DEFAULT_ADI_FILE
        self.adi_dict: Dict[str, Dict[str, Any]] = {}
        self._load_adi_data()
    
    def _load_adi_data(self):
        """Load ADI data from Excel file."""
        try:
            if self.adi_file_path.exists():
                df = pd.read_excel(self.adi_file_path)
                
                # Expected columns: Pesticide, ADI_Value, ADI_Unit, Year, Author
                for _, row in df.iterrows():
                    pesticide = str(row.get('Pesticide', '')).strip().lower()
                    adi_value = row.get('ADI_Value')
                    
                    if pesticide and pd.notna(adi_value):
                        self.adi_dict[pesticide] = {
                            'value': float(adi_value),
                            'unit': str(row.get('ADI_Unit', 'mg/kg bw/day')),
                            'year': row.get('Year'),
                            'author': row.get('Author', 'Unknown')
                        }
                
                print(f"✅ Loaded {len(self.adi_dict)} ADI values from {self.adi_file_path}")
            else:
                print(f"⚠️ ADI file not found: {self.adi_file_path}")
        except Exception as e:
            print(f"❌ Error loading ADI data: {e}")
    
    def get_adi(self, pesticide_name: str) -> Optional[Dict[str, Any]]:
        """
        Get ADI value for a pesticide.
        
        Parameters:
            pesticide_name: Name of the pesticide
        
        Returns:
            Dictionary with 'value', 'unit', 'year', 'author' or None if not found
        """
        # Try exact match first
        key = pesticide_name.strip().lower()
        if key in self.adi_dict:
            return self.adi_dict[key]
        
        # Try partial match
        for stored_name, adi_info in self.adi_dict.items():
            if key in stored_name or stored_name in key:
                return adi_info
        
        return None
    
    def get_adi_value(self, pesticide_name: str) -> Optional[float]:
        """Get just the ADI value (mg/kg bw/day) for a pesticide."""
        adi_info = self.get_adi(pesticide_name)
        return adi_info['value'] if adi_info else None
    
    def list_all(self) -> List[str]:
        """List all pesticides with ADI values."""
        return list(self.adi_dict.keys())


# ============================================================================
# EU MRL API CLIENT
# ============================================================================

class EUMRLClient:
    """Client for fetching MRL values from EU Pesticides Database."""
    
    def __init__(self, timeout: int = 30):
        """Initialize MRL client."""
        self.timeout = timeout
        self._cache: Dict[str, Dict] = {}
    
    def fetch_mrl(
        self, 
        pesticide_name: Optional[str] = None,
        pesticide_code: Optional[str] = None,
        commodity_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch MRL value from EU Pesticides Database API.
        
        Parameters:
            pesticide_name: Name of the active substance (e.g., "Chlorpyrifos")
            pesticide_code: Official code of the substance (more accurate)
            commodity_code: Product code per Regulation (EC) No 396/2005
        
        Returns:
            Dictionary with MRL information or error
        """
        # Check cache first
        cache_key = f"{pesticide_name}:{pesticide_code}:{commodity_code}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if not pesticide_code and not pesticide_name:
            return {"error": "Must provide pesticide name or code"}
        
        # Build API request
        params = {}
        if pesticide_code:
            params['activeSubstanceCode'] = pesticide_code
        elif pesticide_name:
            params['activeSubstance'] = pesticide_name
        
        if commodity_code:
            params['productCode'] = commodity_code
        
        try:
            response = requests.get(
                f"{EU_MRL_API_BASE}/activeSubstances",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if data and isinstance(data, list) and len(data) > 0:
                mrl_info = data[0]
                mrl_value_str = str(mrl_info.get('mrl', ''))
                
                # Parse MRL value (handle * for LOQ)
                try:
                    mrl_value = float(mrl_value_str.rstrip('*'))
                    is_at_loq = '*' in mrl_value_str
                except ValueError:
                    mrl_value = None
                    is_at_loq = False
                
                result = {
                    "active_substance": mrl_info.get('activeSubstance'),
                    "product": mrl_info.get('product'),
                    "mrl": mrl_value,
                    "is_at_loq": is_at_loq,
                    "unit": mrl_info.get('unit', 'mg/kg'),
                    "source": "EU Pesticides Database"
                }
                self._cache[cache_key] = result
                return result
            else:
                return {"error": "No MRL data found for specified criteria"}
        
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {e}"}
        except ValueError as e:
            return {"error": f"Failed to parse response: {e}"}
    
    def clear_cache(self):
        """Clear the MRL cache."""
        self._cache.clear()


# ============================================================================
# RISK ASSESSMENT SERVICE
# ============================================================================

class RiskAssessmentService:
    """
    Comprehensive risk assessment service for pesticide residues.
    
    Implements EDI, HQc, and HIc calculations per PRIMo 4 methodology.
    """
    
    def __init__(
        self,
        adi_file_path: Optional[Path] = None,
        classification_file_path: Optional[Path] = None
    ):
        """Initialize the risk assessment service."""
        self.adi_lookup = ADILookup(adi_file_path)
        self.classifier = ChemicalClassifier(classification_file_path)
        self.mrl_client = EUMRLClient()
    
    def calculate_edi(
        self,
        residue_concentration: float,
        consumption_rate: float,
        body_weight: float
    ) -> float:
        """
        Calculate Estimated Daily Intake (EDI).
        
        Parameters:
            residue_concentration: Pesticide concentration in food (mg/kg)
            consumption_rate: Food consumption rate (kg/day)
            body_weight: Body weight (kg)
        
        Returns:
            EDI in mg/kg bw/day
        """
        if body_weight <= 0:
            raise ValueError("Body weight must be positive")
        return (residue_concentration * consumption_rate) / body_weight
    
    def calculate_hqc(
        self,
        pesticide_name: str,
        residue_concentration: float,
        consumption_rate: float,
        body_weight: float
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate Hazard Quotient for chronic exposure (HQc).
        
        HQc = EDI / ADI
        
        Parameters:
            pesticide_name: Name of the pesticide
            residue_concentration: Concentration in mg/kg
            consumption_rate: Food consumption in kg/day
            body_weight: Body weight in kg
        
        Returns:
            Dictionary with HQc calculation results or None if ADI not found
        """
        adi_info = self.adi_lookup.get_adi(pesticide_name)
        
        if not adi_info:
            return None
        
        edi = self.calculate_edi(residue_concentration, consumption_rate, body_weight)
        hqc = edi / adi_info['value']
        
        # Determine risk interpretation
        if hqc >= 1.0:
            interpretation = "⚠️ Risk unacceptable (HQc ≥ 1)"
            interpretation_ar = "⚠️ خطر غير مقبول"
        elif hqc >= 0.5:
            interpretation = "⚡ Elevated risk (0.5 ≤ HQc < 1)"
            interpretation_ar = "⚡ خطر مرتفع"
        elif hqc >= 0.1:
            interpretation = "⚠ Moderate risk (0.1 ≤ HQc < 0.5)"
            interpretation_ar = "⚠ خطر متوسط"
        else:
            interpretation = "✅ Risk acceptable (HQc < 0.1)"
            interpretation_ar = "✅ خطر مقبول"
        
        return {
            'pesticide': pesticide_name,
            'edi': edi,
            'adi': adi_info['value'],
            'adi_unit': adi_info['unit'],
            'adi_year': adi_info.get('year'),
            'hqc': hqc,
            'hqc_percent': hqc * 100,
            'interpretation': interpretation,
            'interpretation_ar': interpretation_ar,
            'chemical_group': self.classifier.get_group(pesticide_name)
        }
    
    def calculate_hic(
        self,
        hqc_results: List[Dict[str, Any]],
        group_by_chemical_class: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate Hazard Index for chronic exposure (HIc).
        
        HIc = Σ HQc (sum of individual hazard quotients)
        
        Parameters:
            hqc_results: List of HQc calculation results
            group_by_chemical_class: If True, calculate HIc per chemical group
        
        Returns:
            Dictionary with HIc results, optionally grouped by chemical class
        """
        if group_by_chemical_class:
            # Group results by chemical class
            groups: Dict[str, List[Dict]] = {}
            for result in hqc_results:
                if result and result.get('hqc') is not None:
                    group = result.get('chemical_group', 'unknown')
                    if group not in groups:
                        groups[group] = []
                    groups[group].append(result)
            
            # Calculate HIc per group
            hic_by_group = {}
            for group, results in groups.items():
                total_hqc = sum(r['hqc'] for r in results)
                group_info = self.classifier.get_group_info(group)
                
                hic_by_group[group] = {
                    'name_en': group_info.name_en if group_info else group,
                    'name_ar': group_info.name_ar if group_info else group,
                    'hic': total_hqc,
                    'hic_percent': total_hqc * 100,
                    'pesticide_count': len(results),
                    'pesticides': [r['pesticide'] for r in results],
                    'interpretation': "⚠️ Risk unacceptable" if total_hqc >= 1 else "✅ Risk acceptable"
                }
            
            # Total HIc across all groups
            total_hic = sum(g['hic'] for g in hic_by_group.values())
            
            return {
                'total_hic': total_hic,
                'total_hic_percent': total_hic * 100,
                'by_chemical_group': hic_by_group,
                'interpretation': "⚠️ Total risk unacceptable" if total_hic >= 1 else "✅ Total risk acceptable"
            }
        else:
            # Simple sum of all HQc
            total_hic = sum(r['hqc'] for r in hqc_results if r and r.get('hqc') is not None)
            return {
                'total_hic': total_hic,
                'total_hic_percent': total_hic * 100,
                'interpretation': "⚠️ Risk unacceptable" if total_hic >= 1 else "✅ Risk acceptable"
            }
    
    def analyze_sample_risk(
        self,
        sample_results: List[Dict[str, Any]],
        population_class: str = 'adults',
        consumption_rate: float = 0.1  # Default consumption in kg/day
    ) -> Dict[str, Any]:
        """
        Perform complete risk analysis for a sample with multiple pesticide residues.
        
        Parameters:
            sample_results: List of dicts with 'pesticide' and 'concentration' keys
            population_class: Population class for body weight
            consumption_rate: Food consumption rate in kg/day
        
        Returns:
            Comprehensive risk analysis results
        """
        body_weight = get_population_weight(population_class)
        pop_info = POPULATION_CLASSES.get(population_class, POPULATION_CLASSES['adults'])
        
        # Calculate HQc for each pesticide
        hqc_results = []
        missing_adi = []
        
        for residue in sample_results:
            pesticide = residue.get('pesticide', '')
            concentration = residue.get('concentration', 0)
            
            if concentration > 0:
                hqc_result = self.calculate_hqc(
                    pesticide,
                    concentration,
                    consumption_rate,
                    body_weight
                )
                
                if hqc_result:
                    hqc_result['concentration'] = concentration
                    hqc_results.append(hqc_result)
                else:
                    missing_adi.append(pesticide)
        
        # Calculate HIc grouped by chemical class
        hic_results = self.calculate_hic(hqc_results, group_by_chemical_class=True)
        
        # Find highest risk pesticide
        highest_risk = None
        if hqc_results:
            highest_risk = max(hqc_results, key=lambda x: x['hqc'])
        
        # Find highest risk chemical group
        highest_risk_group = None
        if hic_results.get('by_chemical_group'):
            groups = hic_results['by_chemical_group']
            if groups:
                highest_risk_group_key = max(groups.keys(), key=lambda k: groups[k]['hic'])
                highest_risk_group = groups[highest_risk_group_key]
                highest_risk_group['key'] = highest_risk_group_key
        
        return {
            'population': {
                'class': population_class,
                'name_en': pop_info.name_en,
                'name_ar': pop_info.name_ar,
                'body_weight': body_weight
            },
            'consumption_rate': consumption_rate,
            'pesticide_count': len(sample_results),
            'analyzed_count': len(hqc_results),
            'missing_adi_count': len(missing_adi),
            'missing_adi': missing_adi,
            'individual_risks': hqc_results,
            'hazard_index': hic_results,
            'highest_risk_pesticide': highest_risk,
            'highest_risk_group': highest_risk_group,
            'overall_interpretation': hic_results.get('interpretation', 'Unknown')
        }
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        pesticide_col: str = 'pesticide',
        concentration_col: str = 'reading',
        population_class: str = 'adults',
        consumption_rate: float = 0.1
    ) -> pd.DataFrame:
        """
        Analyze risk for a DataFrame of pesticide residue data.
        
        Parameters:
            df: DataFrame with pesticide data
            pesticide_col: Column name for pesticide names
            concentration_col: Column name for concentrations
            population_class: Population class for body weight
            consumption_rate: Food consumption rate in kg/day
        
        Returns:
            DataFrame with added risk columns
        """
        body_weight = get_population_weight(population_class)
        
        # Create result columns
        df = df.copy()
        df['adi'] = None
        df['edi'] = None
        df['hqc'] = None
        df['hqc_percent'] = None
        df['chemical_group'] = None
        df['risk_interpretation'] = None
        
        for idx, row in df.iterrows():
            pesticide = str(row.get(pesticide_col, ''))
            concentration = float(row.get(concentration_col, 0) or 0)
            
            if pesticide and concentration > 0:
                hqc_result = self.calculate_hqc(
                    pesticide,
                    concentration,
                    consumption_rate,
                    body_weight
                )
                
                if hqc_result:
                    df.at[idx, 'adi'] = hqc_result['adi']
                    df.at[idx, 'edi'] = hqc_result['edi']
                    df.at[idx, 'hqc'] = hqc_result['hqc']
                    df.at[idx, 'hqc_percent'] = hqc_result['hqc_percent']
                    df.at[idx, 'chemical_group'] = hqc_result['chemical_group']
                    df.at[idx, 'risk_interpretation'] = hqc_result['interpretation']
        
        return df


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

@lru_cache(maxsize=1)
def get_risk_service() -> RiskAssessmentService:
    """Get singleton instance of RiskAssessmentService."""
    return RiskAssessmentService()


def quick_hqc(
    pesticide: str,
    concentration: float,
    population: str = 'adults',
    consumption: float = 0.1
) -> Optional[Dict[str, Any]]:
    """
    Quick HQc calculation with default parameters.
    
    Parameters:
        pesticide: Pesticide name
        concentration: Residue concentration in mg/kg
        population: Population class (default: adults)
        consumption: Food consumption in kg/day
    
    Returns:
        HQc result dictionary or None
    """
    service = get_risk_service()
    body_weight = get_population_weight(population)
    return service.calculate_hqc(pesticide, concentration, consumption, body_weight)


def calculate_iqr(concentration: float, mrl: float) -> float:
    """
    Calculate Index of Quality for Residues (IqR).
    
    IqR = concentration / MRL
    
    Parameters:
        concentration: Residue concentration in mg/kg
        mrl: Maximum Residue Limit in mg/kg
    
    Returns:
        IqR value (ratio)
    """
    if mrl <= 0:
        return float('inf') if concentration > 0 else 0.0
    return concentration / mrl


def calculate_lars_metrics(
    residue_list: List[Dict[str, Any]],
    commodity: str = "Total Vegetables",
    target: str = "adult",
    body_weight: Optional[float] = None,
    ingestion_rate: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive LARS metrics for a list of pesticide residues.
    
    Implements Saudi-specific calculations with:
    - IqR (Index of Quality) = concentration / MRL
    - HQc (Hazard Quotient) = EDI / ADI
    - Chemical group statistics (Min, Max, Mean, Median)
    - Risk Driver identification
    
    Parameters:
        residue_list: List of dicts with keys:
            - 'name': pesticide name
            - 'concentration': residue concentration mg/kg
            - 'mrl': Maximum Residue Limit mg/kg
            - 'adi': Acceptable Daily Intake mg/kg bw/day (optional)
        commodity: Commodity name for IR lookup (e.g., "Tomato", "Cucumber")
        target: "adult" (53kg) or "child" (16kg)
        body_weight: Override body weight (uses exact value if provided)
        ingestion_rate: Override ingestion rate in kg/day (uses exact value if provided)
    
    Returns:
        Dictionary with:
            - stats: Group statistics {group: {min, max, mean, median, concentrations}}
            - iqr_total: Total IqR sum
            - hi_total: Total Hazard Index
            - risk_driver: Pesticide with highest HQc
            - iqr_driver: Pesticide with highest IqR
            - individual_results: Per-pesticide calculations
    """
    # Define body weight based on target OR use explicit parameter
    if body_weight is not None:
        bw = body_weight
    else:
        bw = 53.0 if target == "adult" else 16.0
    
    # Get ingestion rate for commodity OR use explicit parameter
    if ingestion_rate is not None:
        ir = ingestion_rate
    else:
        ir = get_saudi_ir(commodity)
    
    # Get risk service for ADI lookup
    service = get_risk_service()
    
    # Initialize analysis result
    analysis = {
        "stats": {},
        "iqr_total": 0.0,
        "hi_total": 0.0,
        "risk_driver": None,
        "iqr_driver": None,
        "individual_results": [],
        "population": {
            "target": target,
            "body_weight_kg": bw,
            "commodity": commodity,
            "ingestion_rate_kg_day": ir
        }
    }
    
    highest_hqc = 0.0
    highest_iqr = 0.0
    risk_driver_name = None
    iqr_driver_name = None
    
    for item in residue_list:
        p_name = item.get('name', '')
        concentration = item.get('concentration', 0)
        mrl = item.get('mrl', 0)
        adi = item.get('adi')
        
        # Get chemical group
        group = service.classifier.get_group(p_name) or "Other/Unknown"
        group_info = service.classifier.get_group_info(group)
        group_name = group_info.name_en if group_info else group
        
        # Calculate IqR
        iqr_val = calculate_iqr(concentration, mrl) if mrl > 0 else 0.0
        analysis["iqr_total"] += iqr_val
        
        # Track IqR driver
        if iqr_val > highest_iqr:
            highest_iqr = iqr_val
            iqr_driver_name = f"{p_name} ({group_name})"
        
        # Calculate HQc (need ADI)
        hqc = 0.0
        edi = 0.0
        if adi is None:
            # Try to look up ADI
            adi_info = service.adi_lookup.get_adi(p_name)
            if adi_info:
                adi = adi_info['value']
        
        if adi and adi > 0:
            edi = (concentration * ir) / bw
            hqc = edi / adi
            analysis["hi_total"] += hqc
            
            # Track HQc risk driver
            if hqc > highest_hqc:
                highest_hqc = hqc
                risk_driver_name = f"{p_name} ({group_name})"
        
        # Store individual result
        analysis["individual_results"].append({
            "pesticide": p_name,
            "chemical_group": group_name,
            "concentration": concentration,
            "mrl": mrl,
            "adi": adi,
            "iqr": iqr_val,
            "edi": edi,
            "hqc": hqc,
            "iqr_percent": iqr_val * 100 if iqr_val else 0,
            "hqc_percent": hqc * 100 if hqc else 0
        })
        
        # Aggregate group statistics
        if group_name not in analysis["stats"]:
            analysis["stats"][group_name] = {"concentrations": []}
        analysis["stats"][group_name]["concentrations"].append(concentration)
    
    # Set risk drivers
    analysis["risk_driver"] = risk_driver_name
    analysis["iqr_driver"] = iqr_driver_name
    
    # Calculate final statistics per group
    for group_name, data in analysis["stats"].items():
        vals = data["concentrations"]
        if vals:
            analysis["stats"][group_name] = {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "count": len(vals),
                "concentrations": vals  # Keep raw data
            }
    
    # Add risk interpretations
    analysis["hi_interpretation"] = "⚠️ Risk unacceptable" if analysis["hi_total"] >= 1 else "✅ Risk acceptable"
    analysis["iqr_interpretation"] = "⚠️ Quality concern" if analysis["iqr_total"] >= 1 else "✅ Quality acceptable"
    
    return analysis


def detect_query_type(query: str) -> Optional[str]:
    """
    Detect if query triggers Health Risk or Quality window.
    
    Parameters:
        query: User query text (Arabic or English)
    
    Returns:
        "[TRIGGER_UI: HEALTH_RISK_WINDOW]" for risk queries
        "[TRIGGER_UI: QUALITY_INDEX_WINDOW]" for quality queries
        None if no trigger detected
    """
    query_lower = query.lower()
    
    # Health Risk keywords (Arabic and English)
    risk_keywords = ['خطر', 'خطورة', 'hri', 'risk', 'health risk', 'hazard', 'هازارد']
    if any(kw in query_lower for kw in risk_keywords):
        return "[TRIGGER_UI: HEALTH_RISK_WINDOW]"
    
    # Quality Index keywords (Arabic and English)
    quality_keywords = ['جودة', 'iqr', 'pti', 'quality', 'index', 'مؤشر']
    if any(kw in query_lower for kw in quality_keywords):
        return "[TRIGGER_UI: QUALITY_INDEX_WINDOW]"
    
    return None




# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RISK ASSESSMENT SERVICE - TEST")
    print("=" * 80)
    
    # Initialize service
    service = RiskAssessmentService()
    
    # Test ADI lookup
    print("\n📊 Testing ADI Lookup:")
    for pesticide in ['chlorpyrifos', 'deltamethrin', 'imidacloprid', 'unknown_pest']:
        adi = service.adi_lookup.get_adi(pesticide)
        if adi:
            print(f"  ✅ {pesticide}: ADI = {adi['value']} {adi['unit']}")
        else:
            print(f"  ❌ {pesticide}: NOT FOUND")
    
    # Test chemical classification
    print("\n🧪 Testing Chemical Classification:")
    pesticides_to_classify = ['lambda-cyhalothrin', 'chlorpyrifos', 'carbendazim', 'unknown']
    for pest in pesticides_to_classify:
        group = service.classifier.get_group(pest)
        print(f"  {pest} → {group or 'Unknown'}")
    
    # Test HQc calculation
    print("\n🔬 Testing HQc Calculation:")
    result = service.calculate_hqc('chlorpyrifos', 0.1, 0.1, 70)
    if result:
        print(f"  Pesticide: {result['pesticide']}")
        print(f"  EDI: {result['edi']:.6f} mg/kg bw/day")
        print(f"  ADI: {result['adi']} mg/kg bw/day")
        print(f"  HQc: {result['hqc']:.4f} ({result['hqc_percent']:.2f}%)")
        print(f"  Interpretation: {result['interpretation']}")
    
    # Test sample analysis
    print("\n📋 Testing Sample Risk Analysis:")
    sample = [
        {'pesticide': 'lambda-cyhalothrin', 'concentration': 0.02},
        {'pesticide': 'chlorpyrifos', 'concentration': 0.1},
        {'pesticide': 'carbendazim', 'concentration': 0.05}
    ]
    
    analysis = service.analyze_sample_risk(sample, population_class='toddlers')
    print(f"  Population: {analysis['population']['name_en']} ({analysis['population']['body_weight']} kg)")
    print(f"  Analyzed: {analysis['analyzed_count']}/{analysis['pesticide_count']} pesticides")
    print(f"  Total HIc: {analysis['hazard_index']['total_hic']:.4f}")
    print(f"  Overall: {analysis['overall_interpretation']}")
    
    if analysis['highest_risk_group']:
        print(f"  Highest risk group: {analysis['highest_risk_group']['name_en']}")
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")
