from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Reference Table of Normal Ranges
# Contains the low/high bounds and thresholds for status classification.
REFERENCE_RANGES = {
    "Glucose": {
        "normal_min": 70.0,
        "normal_max": 100.0,
        "borderline_max": 125.0,
        "critical_min": 50.0,
        "critical_max": 250.0,
        "unit": "mg/dL",
        "description": "Fasting glucose levels indicate how the body metabolizes sugar. High levels can indicate prediabetes or diabetes."
    },
    "HbA1c": {
        "normal_min": 0.0,
        "normal_max": 5.6,
        "borderline_max": 6.4,
        "critical_min": None,
        "critical_max": 10.0,
        "unit": "%",
        "description": "HbA1c measures average blood sugar over the past 3 months. Used to diagnose and monitor diabetes."
    },
    "Creatinine": {
        "normal_min": 0.7,
        "normal_max": 1.3,
        "borderline_max": 1.5,
        "critical_min": 0.3,
        "critical_max": 2.0,
        "unit": "mg/dL",
        "description": "Creatinine is a waste product filtered by the kidneys. High levels suggest reduced kidney function."
    },
    "WBC": {
        "normal_min": 4500.0,
        "normal_max": 11000.0,
        "borderline_max": 15000.0,
        "critical_min": 1500.0,
        "critical_max": 20000.0,
        "unit": "/μL",
        "description": "White blood cells defend against infections. High counts suggest infection or inflammation; low counts suggest immune suppression."
    },
    "Hemoglobin": {
        "normal_min": 13.5,
        "normal_max": 17.5,
        "borderline_max": 18.5,
        "critical_min": 8.0,
        "critical_max": 20.0,
        "unit": "g/dL",
        "description": "Hemoglobin is the oxygen-carrying protein in red blood cells. Low values indicate anemia."
    },
    "Platelet": {
        "normal_min": 150000.0,
        "normal_max": 400000.0,
        "borderline_max": 500000.0,
        "critical_min": 50000.0,
        "critical_max": 1000000.0,
        "unit": "/μL",
        "description": "Platelets are critical for blood clotting. Low platelets increase bleeding risks; high platelets increase clotting risks."
    },
    "Systolic BP": {
        "normal_min": 90.0,
        "normal_max": 120.0,
        "borderline_max": 139.0,
        "critical_min": 80.0,
        "critical_max": 180.0,
        "unit": "mmHg",
        "description": "Systolic blood pressure measures pressure when the heart beats. High pressure strains blood vessels."
    },
    "Diastolic BP": {
        "normal_min": 60.0,
        "normal_max": 80.0,
        "borderline_max": 89.0,
        "critical_min": 50.0,
        "critical_max": 120.0,
        "unit": "mmHg",
        "description": "Diastolic blood pressure measures pressure when the heart rests between beats."
    },
    "Cholesterol Total": {
        "normal_min": 0.0,
        "normal_max": 199.0,
        "borderline_max": 239.0,
        "critical_min": None,
        "critical_max": 300.0,
        "unit": "mg/dL",
        "description": "Total cholesterol measures all cholesterol in your blood. Elevated levels raise cardiovascular risks."
    },
    "LDL": {
        "normal_min": 0.0,
        "normal_max": 99.0,
        "borderline_max": 159.0,
        "critical_min": None,
        "critical_max": 190.0,
        "unit": "mg/dL",
        "description": "LDL is 'bad' cholesterol that can build up in arterial walls, increasing cardiovascular risks."
    },
    "HDL": {
        "normal_min": 60.0,
        "normal_max": 150.0,
        "borderline_max": None,
        "critical_min": 30.0,
        "critical_max": None,
        "unit": "mg/dL",
        "description": "HDL is 'good' cholesterol that helps clear other fats from blood. Higher is generally better; low levels increase risk."
    }
}

def analyze_abnormalities(extracted_values: Dict[str, Tuple[float, str]]) -> List[Dict[str, Any]]:
    """
    Compares the extracted test values against the reference ranges and classifies status.
    Returns: [{test_name, value, unit, normal_range, status, explanation}]
    """
    results = []

    for test_name, (value, extracted_unit) in extracted_values.items():
        if test_name not in REFERENCE_RANGES:
            continue

        ref = REFERENCE_RANGES[test_name]
        normal_min = ref["normal_min"]
        normal_max = ref["normal_max"]
        borderline_max = ref.get("borderline_max")
        critical_min = ref.get("critical_min")
        critical_max = ref.get("critical_max")
        unit = ref["unit"] or extracted_unit

        # Construct normal range string representation
        if normal_min == 0.0 or normal_min is None:
            normal_range_str = f"< {normal_max} {unit}"
        elif normal_max is None:
            normal_range_str = f"> {normal_min} {unit}"
        else:
            normal_range_str = f"{normal_min} - {normal_max} {unit}"

        # Status Classification Logic
        status = "NORMAL"
        explanation = ""

        # Special logic for HDL (higher is better)
        if test_name == "HDL":
            if critical_min is not None and value < critical_min:
                status = "CRITICAL"
                explanation = f"Your HDL level ({value} mg/dL) is critically low. Good cholesterol protects the heart; very low levels indicate significant cardiovascular vulnerability."
            elif value < 40.0:
                status = "LOW"
                explanation = f"Your HDL level ({value} mg/dL) is low. Levels below 40 mg/dL are associated with an increased risk of heart disease."
            elif value < 60.0:
                status = "BORDERLINE"
                explanation = f"Your HDL level ({value} mg/dL) is borderline-low. Increasing physical activity and healthy fats can help raise this."
            else:
                status = "NORMAL"
                explanation = f"Your HDL level ({value} mg/dL) is normal and healthy, providing strong cardiovascular protection."
        
        # Standard values classification
        else:
            # Check Critical High
            if critical_max is not None and value >= critical_max:
                status = "CRITICAL"
                explanation = f"Your {test_name} value of {value} {unit} is critically high. This requires prompt clinical evaluation."
            # Check Critical Low
            elif critical_min is not None and value <= critical_min:
                status = "CRITICAL"
                explanation = f"Your {test_name} value of {value} {unit} is critically low. This requires prompt clinical evaluation."
            # Check High
            elif normal_max is not None and value > normal_max:
                if borderline_max is not None and value <= borderline_max:
                    status = "BORDERLINE"
                    explanation = f"Your {test_name} ({value} {unit}) is borderline high, just above the normal limit of {normal_max} {unit}."
                else:
                    status = "HIGH"
                    explanation = f"Your {test_name} ({value} {unit}) is high. This is above the recommended range of {normal_range_str}."
            # Check Low
            elif normal_min is not None and value < normal_min:
                status = "LOW"
                explanation = f"Your {test_name} ({value} {unit}) is low. This is below the recommended range of {normal_range_str}."
            # Normal
            else:
                status = "NORMAL"
                explanation = f"Your {test_name} ({value} {unit}) is within the normal reference range of {normal_range_str}."

        # Add global test description to the explanation
        base_desc = ref.get("description", "")
        full_explanation = f"{explanation} {base_desc}".strip()

        results.append({
            "test_name": test_name,
            "value": value,
            "unit": unit,
            "normal_range": normal_range_str,
            "status": status,
            "explanation": full_explanation
        })

    return results
