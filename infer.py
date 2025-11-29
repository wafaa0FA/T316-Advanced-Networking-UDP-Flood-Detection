#!/usr/bin/env python3
"""
Real-time Network Traffic Classification Inference

Usage:
    python3 infer.py <csv_file>               # Classify from CSV file
    python3 infer.py --live <interface>       # Live capture and classify (requires root)
    python3 infer.py --pcap <pcap_file>       # Classify from PCAP file

Examples:
    python3 infer.py ~/adv316_project/new_sample.csv
    sudo python3 infer.py --live eth0
    python3 infer.py --pcap /path/to/capture.pcap
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

# Config
MODELS_DIR = Path.home() / 'tma_project' / 'models'
RESULTS_DIR = Path.home() / 'tma_project' / 'inference_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALERT_THRESHOLD = 0.7  # Probability threshold for attack alert

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Print script header."""
    print(f"\n{Colors.BOLD}{'='*70}")
    print("   NETWORK TRAFFIC CLASSIFICATION - INFERENCE ENGINE")
    print(f"{'='*70}{Colors.END}\n")

def load_models():
    """Load trained models and scaler."""
    print(f"{Colors.BLUE}[INFO]{Colors.END} Loading models from: {MODELS_DIR}")
    
    scaler_path = MODELS_DIR / 'scaler.joblib'
    rf_path = MODELS_DIR / 'rf_model.joblib'
    knn_path = MODELS_DIR / 'knn_model.joblib'
    
    if not scaler_path.exists():
        print(f"{Colors.RED}[ERROR]{Colors.END} Scaler not found at: {scaler_path}")
        print("Please train the models first using train_and_evaluate.py")
        sys.exit(1)
    
    if not rf_path.exists():
        print(f"{Colors.RED}[ERROR]{Colors.END} RandomForest model not found at: {rf_path}")
        sys.exit(1)
    
    scaler = joblib.load(scaler_path)
    rf_model = joblib.load(rf_path)
    
    # KNN is optional
    knn_model = None
    if knn_path.exists():
        knn_model = joblib.load(knn_path)
        print(f"{Colors.GREEN}[OK]{Colors.END} Loaded: Scaler, RandomForest, KNN")
    else:
        print(f"{Colors.GREEN}[OK]{Colors.END} Loaded: Scaler, RandomForest")
        print(f"{Colors.YELLOW}[WARN]{Colors.END} KNN model not found, using RF only")
    
    return scaler, rf_model, knn_model

def prepare_features(df, scaler):
    """Prepare features from dataframe."""
    # Remove non-numeric columns if any
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].copy()
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print(f"{Colors.YELLOW}[WARN]{Colors.END} Found missing values, filling with median")
        X = X.fillna(X.median())
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
    
    # Get expected features from scaler
    expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
    
    if expected_features is not None:
        # Check if all expected features are present
        missing_features = set(expected_features) - set(X.columns)
        if missing_features:
            print(f"{Colors.YELLOW}[WARN]{Colors.END} Missing features: {missing_features}")
            # Add missing features with zeros
            for feat in missing_features:
                X[feat] = 0
        
        # Select only expected features in correct order
        X = X[expected_features]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, X

def classify_sample(X_scaled, X_original, rf_model, knn_model=None):
    """Classify samples and return detailed results."""
    results = []
    
    for idx in range(len(X_scaled)):
        sample = X_scaled[idx:idx+1]
        
        # RandomForest prediction
        rf_pred = rf_model.predict(sample)[0]
        rf_proba = rf_model.predict_proba(sample)[0]
        rf_confidence = rf_proba[1] if rf_pred == 1 else rf_proba[0]
        
        # KNN prediction (if available)
        knn_pred = None
        knn_confidence = None
        if knn_model is not None:
            knn_pred = knn_model.predict(sample)[0]
            knn_proba = knn_model.predict_proba(sample)[0]
            knn_confidence = knn_proba[1] if knn_pred == 1 else knn_proba[0]
        
        # Ensemble prediction (if both models available)
        if knn_model is not None:
            ensemble_pred = 1 if (rf_proba[1] + knn_proba[1]) / 2 > ALERT_THRESHOLD else 0
            ensemble_confidence = (rf_proba[1] + knn_proba[1]) / 2
        else:
            ensemble_pred = rf_pred
            ensemble_confidence = rf_confidence
        
        results.append({
            'sample_id': idx,
            'rf_prediction': rf_pred,
            'rf_confidence': rf_confidence,
            'rf_attack_prob': rf_proba[1],
            'knn_prediction': knn_pred,
            'knn_confidence': knn_confidence,
            'ensemble_prediction': ensemble_pred,
            'ensemble_confidence': ensemble_confidence,
            'timestamp': datetime.now().isoformat()
        })
    
    return results

def print_results(results, save_to_file=True):
    """Print classification results in a nice format."""
    print(f"\n{Colors.BOLD}{'='*70}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*70}{Colors.END}\n")
    
    attack_count = 0
    normal_count = 0
    
    for res in results:
        sample_id = res['sample_id']
        pred = res['ensemble_prediction']
        confidence = res['ensemble_confidence']
        attack_prob = res['rf_attack_prob']
        
        if pred == 1:
            attack_count += 1
            status = f"{Colors.RED}{Colors.BOLD}ATTACK DETECTED{Colors.END}"
            action = f"{Colors.RED}→ Action: Raise alert / Throttle bandwidth{Colors.END}"
        else:
            normal_count += 1
            status = f"{Colors.GREEN}Normal Traffic{Colors.END}"
            action = f"{Colors.GREEN}→ Action: Allow{Colors.END}"
        
        print(f"Sample #{sample_id + 1}:")
        print(f"  Status:           {status}")
        print(f"  Confidence:       {confidence:.2%}")
        print(f"  Attack Prob:      {attack_prob:.2%}")
        print(f"  RF Prediction:    {'Attack' if res['rf_prediction'] == 1 else 'Normal'}")
        if res['knn_prediction'] is not None:
            print(f"  KNN Prediction:   {'Attack' if res['knn_prediction'] == 1 else 'Normal'}")
        print(f"  {action}")
        print()
    
    # Summary
    print(f"{Colors.BOLD}{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}{Colors.END}")
    print(f"Total samples analyzed:    {len(results)}")
    print(f"{Colors.GREEN}Normal traffic:{Colors.END}           {normal_count}")
    print(f"{Colors.RED}Attacks detected:{Colors.END}         {attack_count}")
    
    if attack_count > 0:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠ WARNING: {attack_count} attack(s) detected!{Colors.END}")
        print(f"{Colors.RED}Recommended actions:{Colors.END}")
        print(f"  1. Investigate source IPs")
        print(f"  2. Enable rate limiting")
        print(f"  3. Check firewall rules")
        print(f"  4. Monitor system logs")
    else:
        print(f"\n{Colors.GREEN}✓ All traffic appears normal{Colors.END}")
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    # Save to file
    if save_to_file:
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = RESULTS_DIR / f'inference_results_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"{Colors.BLUE}[INFO]{Colors.END} Results saved to: {output_file}")

def infer_from_csv(csv_path, scaler, rf_model, knn_model):
    """Run inference on CSV file."""
    print(f"{Colors.BLUE}[INFO]{Colors.END} Loading data from: {csv_path}")
    
    if not Path(csv_path).exists():
        print(f"{Colors.RED}[ERROR]{Colors.END} File not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"{Colors.GREEN}[OK]{Colors.END} Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Prepare features
    print(f"{Colors.BLUE}[INFO]{Colors.END} Preparing features...")
    X_scaled, X_original = prepare_features(df, scaler)
    
    # Classify
    print(f"{Colors.BLUE}[INFO]{Colors.END} Running classification...")
    results = classify_sample(X_scaled, X_original, rf_model, knn_model)
    
    # Print results
    print_results(results)

def infer_from_pcap(pcap_path, scaler, rf_model, knn_model):
    """Extract features from PCAP and classify."""
    print(f"{Colors.YELLOW}[TODO]{Colors.END} PCAP inference not yet implemented")
    print(f"You can use tshark or scapy to extract features first, then use CSV mode")
    print(f"\nExample workflow:")
    print(f"  1. Extract features: tshark -r {pcap_path} ... > features.csv")
    print(f"  2. Run inference: python3 infer.py features.csv")

def print_usage():
    """Print usage instructions."""
    print(__doc__)

def main():
    print_header()
    
    # Parse arguments
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    mode = sys.argv[1]
    
    # Load models
    scaler, rf_model, knn_model = load_models()
    
    # Run inference based on mode
    if mode == '--help' or mode == '-h':
        print_usage()
    elif mode == '--live':
        print(f"{Colors.YELLOW}[TODO]{Colors.END} Live capture mode not yet implemented")
        print("This would require capturing packets in real-time and classifying them")
    elif mode == '--pcap':
        if len(sys.argv) < 3:
            print(f"{Colors.RED}[ERROR]{Colors.END} PCAP file path required")
            print("Usage: python3 infer.py --pcap <pcap_file>")
            sys.exit(1)
        infer_from_pcap(sys.argv[2], scaler, rf_model, knn_model)
    else:
        # Assume it's a CSV file path
        infer_from_csv(mode, scaler, rf_model, knn_model)

if __name__ == "__main__":
    main()
