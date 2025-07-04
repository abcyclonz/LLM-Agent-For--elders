import json
from typing import List, Dict, Any, Union, Tuple

def get_health_from_file(file_path: str = "health_data.json") -> Dict[str, Any]:
    """Reads the latest health data from the JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️ Could not read health data file ('{file_path}'): {e}. Using default values.")
        return {
            "heart_rate": 80, "spo2": 98, "bp_systolic": 120,
            "bp_diastolic": 80, "ecg_status": "Normal",
            "fall_detected": False, "sleep_score": 75
        }

def check_health_data() -> Union[List[str], None]:
    """
    Checks health data against predefined thresholds and returns a list of issue strings.
    Returns None if no issues are found.
    """
    print("--- Checking Health Data ---")
    health_data = get_health_from_file()
    issues = []

    # Deterministic rule-based checks
    if health_data.get("heart_rate", 80) > 110 or health_data.get("heart_rate", 80) < 50:
        issues.append(f"Heart rate is at a critical level: {health_data['heart_rate']} BPM.")

    if health_data.get("spo2", 98) < 94:
        issues.append(f"Oxygen saturation (SpO2) is low: {health_data['spo2']}%.")

    if health_data.get("bp_systolic", 120) > 140 or health_data.get("bp_diastolic", 80) > 90:
        issues.append(f"Blood pressure is high: {health_data['bp_systolic']}/{health_data['bp_diastolic']} mmHg.")

    if health_data.get("ecg_status", "Normal").lower() != "normal":
        issues.append(f"ECG is showing an abnormal status: {health_data['ecg_status']}.")

    if health_data.get("fall_detected", False):
        # This is the most critical alert.
        issues.insert(0, "URGENT: A fall has been detected!")

    if health_data.get("sleep_score", 75) < 60:
        issues.append(f"Recent sleep quality was poor (Sleep Score: {health_data['sleep_score']}).")

    if issues:
        print(f"  Health issues detected: {issues}")
        return issues
    
    print("  No health issues detected.")
    return None