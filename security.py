import hashlib
import hmac
import json
import time
import os
from typing import Any, Dict
from datetime import datetime

import pandas as pd

import config


def hash_data(data: Any) -> str:
    """
    Return SHA-256 hex digest of the string representation of data.
    
    Args:
        data (Any): Arbitrary data to hash.
        
    Returns:
        str: SHA-256 hex digest.
    """
    data_str = str(data).encode('utf-8')
    return hashlib.sha256(data_str).hexdigest()


def generate_data_fingerprint(df: pd.DataFrame) -> str:
    """
    Generate SHA-256 fingerprint of a dataframe based on shape, columns, 
    and first/last rows to detect data tampering.
    
    Args:
        df (pd.DataFrame): Dataframe to fingerprint.
        
    Returns:
        str: SHA-256 hex digest.
    """
    shape_str = str(df.shape)
    cols_str = str(list(df.columns))
    
    # Convert first and last row to string
    if not df.empty:
        first_row_str = str(df.iloc[0].to_dict())
        last_row_str = str(df.iloc[-1].to_dict())
    else:
        first_row_str = ""
        last_row_str = ""
        
    serialized = f"{shape_str}|{cols_str}|{first_row_str}|{last_row_str}"
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def sign_alert(message: str, site: str, compound: str) -> Dict[str, Any]:
    """
    Create a signed alert simulating cryptographic cloud alert signing.
    Uses HMAC-SHA256 with RANDOM_SEED as key.
    
    Args:
        message (str): Alert message.
        site (str): Site name.
        compound (str): Compound name.
        
    Returns:
        Dict[str, Any]: Signed alert payload including the signature.
    """
    timestamp = time.time()
    payload = {
        "message": message,
        "site": site,
        "compound": compound,
        "timestamp": timestamp
    }
    
    # Create signature over deterministic JSON serialization
    payload_str = json.dumps(payload, sort_keys=True)
    key = str(config.RANDOM_SEED).encode('utf-8')
    signature = hmac.new(key, payload_str.encode('utf-8'), hashlib.sha256).hexdigest()
    
    # Add signature to return dict
    alert_dict = payload.copy()
    alert_dict["signature"] = signature
    return alert_dict


def verify_signature(signed_alert: Dict[str, Any]) -> bool:
    """
    Verify the signature of a signed alert.
    
    Args:
        signed_alert (Dict[str, Any]): The signed alert payload.
        
    Returns:
        bool: True if signature is valid, False otherwise.
    """
    alert_copy = signed_alert.copy()
    
    if "signature" not in alert_copy:
        return False
        
    provided_signature = alert_copy.pop("signature")
    
    payload_str = json.dumps(alert_copy, sort_keys=True)
    key = str(config.RANDOM_SEED).encode('utf-8')
    expected_signature = hmac.new(key, payload_str.encode('utf-8'), hashlib.sha256).hexdigest()
    
    return hmac.compare_digest(provided_signature, expected_signature)


def log_security_event(event_type: str, detail: str, log_dir: str) -> None:
    """
    Append a security event to security_log.txt.
    Format: [TIMESTAMP] [EVENT_TYPE] detail
    
    Args:
        event_type (str): Type of security event.
        detail (str): Event details/message.
        log_dir (str): Directory to store the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "security_log.txt")
    timestamp_str = datetime.now().isoformat()
    
    log_entry = f"[{timestamp_str}] [{event_type}] {detail}\n"
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def generate_run_id() -> str:
    """
    Generate a unique run ID.
    
    Returns:
        str: Unique run ID string in format run_{YYYYMMDD}_{HHMMSS}.
    """
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


if __name__ == "__main__":
    # Quick module tests
    print("Testing security module...")
    run_id = generate_run_id()
    print(f"Run ID: {run_id}")
    
    alert = sign_alert("PFAS Spike Detected", "Site_A", "PFOA")
    print(f"\nSigned Alert: {json.dumps(alert, indent=2)}")
    
    is_valid = verify_signature(alert)
    print(f"Signature Valid: {is_valid}")
    
    df_dummy = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    fp = generate_data_fingerprint(df_dummy)
    print(f"\nData Fingerprint: {fp}")
