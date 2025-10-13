# src/services/enhanced_risk_assessment_service.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

class EnhancedRiskAssessment:
    """Enhanced risk assessment with Monte Carlo and Bayesian methods"""
    
    def __init__(self, historical_data_path: str):
        """Initialize with historical data"""
        self.df = pd.read_excel('/app/data/processed_data_output.xlsx')
        self.df['document_date'] = pd.to_datetime(self.df['document_date'])
        
        # Define hazard multipliers based on exposure risk
        self.vegetable_multipliers = {
            'leafy_greens': 1.5,      # Higher exposure (eaten raw, large surface area)
            'peppers': 1.2,           # Moderate exposure
            'herbs': 1.4,             # High exposure (eaten raw)
            'root_vegetables': 0.8,   # Lower exposure (often peeled)
            'fruits': 1.0             # Baseline
        }
        
        # Bayesian priors for pesticide groups (base risk)
        self.pesticide_priors = {
            'phenylpyrazoles': 0.75,           # High base risk
            'pyrethroid': 0.70,                # High base risk
            'organophosphorus': 0.65,          # High base risk
            'neonicotinoid': 0.60,            # Moderate-high risk
            'chitin_synthesis_inhibitor': 0.55, # Moderate risk
            'strobilurin': 0.50,              # Moderate risk
            'other': 0.45                      # Lower base risk
        }
        
        # Group-specific critical thresholds
        self.critical_thresholds = {
            'chitin_synthesis_inhibitor': 3.0,
            'pyrethroid': 3.5,
            'organophosphorus': 3.0,
            'phenylpyrazoles': 3.2,
            'neonicotinoid': 3.3,
            'default': 4.0
        }
    
    def monte_carlo_exceedance_probability(
        self, 
        vegetable: str,
        pesticide_group: str,
        current_reading: float,
        limit: float,
        n_simulations: int = 10000
    ) -> Dict:
        """
        Monte Carlo simulation for probabilistic exceedance
        Samples from empirical distribution of subgroup readings
        """
        
        # Filter historical data for similar cases
        subgroup = self.df[
            (self.df['vegetable_english'] == vegetable) & 
            (self.df['pesticide_group'] == pesticide_group)
        ]['reading'].dropna()
        
        if len(subgroup) < 5:
            # Not enough data, use broader group
            subgroup = self.df[self.df['pesticide_group'] == pesticide_group]['reading'].dropna()
        
        if len(subgroup) < 5:
            # Still not enough, return simple calculation
            return {
                'exceedance_probability': 1.0 if current_reading > limit else 0.0,
                'confidence_interval': (0, 0),
                'n_samples': 0,
                'method': 'deterministic'
            }
        
        # Fit distribution to historical data
        mu = subgroup.mean()
        sigma = subgroup.std()
        
        # Monte Carlo simulation
        simulated_readings = np.random.normal(mu, sigma, n_simulations)
        
        # Calculate exceedance probability
        exceedances = (simulated_readings > limit).sum()
        prob_exceed = exceedances / n_simulations
        
        # 95% confidence interval
        ci_lower, ci_upper = stats.binom.interval(0.95, n_simulations, prob_exceed)
        ci_lower /= n_simulations
        ci_upper /= n_simulations
        
        return {
            'exceedance_probability': prob_exceed,
            'confidence_interval': (ci_lower, ci_upper),
            'n_samples': len(subgroup),
            'method': 'monte_carlo',
            'mean_historical': mu,
            'std_historical': sigma
        }
    
    def bayesian_risk_update(
        self,
        pesticide_group: str,
        exceedance_ratio: float,
        is_compliant: bool
    ) -> float:
        """
        Bayesian update of risk score using group priors
        Combines prior knowledge with observed data
        """
        
        # Get prior probability (base risk for pesticide group)
        prior = self.pesticide_priors.get(pesticide_group, 0.5)
        
        # Likelihood based on exceedance ratio
        if exceedance_ratio > 1:
            # Non-compliant case
            likelihood = min(0.95, 0.5 + (exceedance_ratio - 1) * 0.15)
        else:
            # Compliant case
            likelihood = max(0.05, 0.3 - exceedance_ratio * 0.1)
        
        # Bayesian update: P(risk|data) ∝ P(data|risk) * P(risk)
        posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
        
        return posterior
    
    def calculate_enhanced_risk_score(
        self,
        vegetable: str,
        vegetable_category: str,
        pesticide_group: str,
        reading: float,
        limit: float,
        is_compliant: bool
    ) -> Dict:
        """
        Enhanced risk calculation with all methods combined
        """
        
        # 1. Calculate exceedance ratio
        exceedance_ratio = reading / limit if limit > 0 else 0
        
        # 2. Monte Carlo exceedance probability
        mc_result = self.monte_carlo_exceedance_probability(
            vegetable=vegetable,
            pesticide_group=pesticide_group,
            current_reading=reading,
            limit=limit
        )
        
        # 3. Bayesian risk update
        bayesian_risk = self.bayesian_risk_update(
            pesticide_group=pesticide_group,
            exceedance_ratio=exceedance_ratio,
            is_compliant=is_compliant
        )
        
        # 4. Apply vegetable category multiplier (hazard model)
        veg_multiplier = self.vegetable_multipliers.get(vegetable_category, 1.0)
        
        # 5. Combine all factors into final risk score (0-5 scale)
        base_risk = min(5.0, exceedance_ratio)
        probabilistic_risk = mc_result['exceedance_probability'] * 2  # Scale to 0-2
        bayesian_component = bayesian_risk * 2  # Scale to 0-2
        
        # Weighted combination
        raw_risk = (
            base_risk * 0.4 +           # 40% from exceedance
            probabilistic_risk * 0.3 +   # 30% from probability
            bayesian_component * 0.3     # 30% from Bayesian
        ) * veg_multiplier               # Apply vegetable hazard multiplier
        
        final_risk_score = min(5.0, raw_risk)
        
        # 6. Determine risk level using group-specific thresholds
        threshold = self.critical_thresholds.get(pesticide_group, self.critical_thresholds['default'])
        
        if final_risk_score >= threshold:
            risk_level = 'CRITICAL'
        elif final_risk_score >= 3.0:
            risk_level = 'HIGH'
        elif final_risk_score >= 2.0:
            risk_level = 'MEDIUM'
        elif final_risk_score >= 1.0:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return {
            'risk_score': round(final_risk_score, 2),
            'risk_level': risk_level,
            'exceedance_probability': round(mc_result['exceedance_probability'], 3),
            'confidence_interval': mc_result['confidence_interval'],
            'bayesian_risk': round(bayesian_risk, 3),
            'vegetable_multiplier': veg_multiplier,
            'critical_threshold': threshold,
            'components': {
                'base_risk': round(base_risk, 2),
                'probabilistic_risk': round(probabilistic_risk, 2),
                'bayesian_component': round(bayesian_component, 2)
            }
        }
    
    def batch_assess_risks(self, test_cases: pd.DataFrame) -> pd.DataFrame:
        """Apply enhanced risk assessment to batch of samples"""
        
        results = []
        
        for _, row in test_cases.iterrows():
            risk_result = self.calculate_enhanced_risk_score(
                vegetable=row['vegetable_english'],
                vegetable_category=row['vegetable_category'],
                pesticide_group=row['pesticide_group'],
                reading=row['reading'],
                limit=row['limits'],
                is_compliant=row['is_compliant']
            )
            
            results.append({
                **row.to_dict(),
                **risk_result
            })
        
        return pd.DataFrame(results)


# Example usage and testing
def test_enhanced_risk_assessment():
    """Test the enhanced risk assessment"""
    
    # Initialize with historical data
    risk_model = EnhancedRiskAssessment('/app/data/processed_data_output.xlsx')
    
    # Test case 1: High risk scenario
    result1 = risk_model.calculate_enhanced_risk_score(
        vegetable='parsley',
        vegetable_category='leafy_greens',
        pesticide_group='organophosphorus',
        reading=347,
        limit=20,
        is_compliant=False
    )
    
    print("Test Case 1 (High Risk):")
    print(f"Risk Score: {result1['risk_score']}")
    print(f"Risk Level: {result1['risk_level']}")
    print(f"Exceedance Probability: {result1['exceedance_probability']}")
    print(f"Bayesian Risk: {result1['bayesian_risk']}")
    print()
    
    # Test case 2: Low risk scenario
    result2 = risk_model.calculate_enhanced_risk_score(
        vegetable='red_bell_pepper',
        vegetable_category='peppers',
        pesticide_group='chitin_synthesis_inhibitor',
        reading=40,
        limit=10,
        is_compliant=True
    )
    
    print("Test Case 2 (Moderate Risk):")
    print(f"Risk Score: {result2['risk_score']}")
    print(f"Risk Level: {result2['risk_level']}")
    print(f"Critical Threshold: {result2['critical_threshold']}")
    print()

if __name__ == "__main__":
    test_enhanced_risk_assessment()