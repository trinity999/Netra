#!/usr/bin/env python3
"""
NETRA Model Retraining Script
============================

Retrains NETRA's ML models with the expanded knowledge base
for maximum intelligence and accuracy.
"""

import sqlite3
from subdomain_ai_enhanced import SubdomainAIEnhanced

def retrain_netra_model():
    """Retrain NETRA with the full knowledge base."""
    print("🔄 NETRA Model Retraining")
    print("=" * 30)
    
    # Initialize NETRA
    ai = SubdomainAIEnhanced()
    
    # Get all training data from knowledge base
    conn = sqlite3.connect(ai.kb.db_path)
    cursor = conn.cursor()
    
    print("[*] Loading training data from knowledge base...")
    cursor.execute('''
        SELECT domain, category 
        FROM subdomains 
        WHERE source IN ('synthetic', 'synthetic_advanced', 'manual', 'verified')
        ORDER BY RANDOM()
    ''')
    
    training_data = cursor.fetchall()
    conn.close()
    
    print(f"[+] Loaded {len(training_data):,} training samples")
    
    # Display category distribution
    from collections import Counter
    category_counts = Counter([item[1] for item in training_data])
    print(f"\n📊 Category Distribution:")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count:,}")
    
    # Train the model
    print(f"\n[*] Training ML model with {len(training_data):,} samples...")
    success = ai.classifier.train_model(training_data, test_size=0.15)
    
    if success:
        print("[+] Model retraining completed successfully!")
        print("[*] Running quick validation test...")
        
        # Test with sample subdomains
        test_subdomains = [
            'admin.company.com',
            'api-v2.company.com', 
            'auth-sso.company.com',
            'payment-gateway.company.com',
            'db-prod.company.com',
            'jenkins-ci.company.com',
            'cdn-static.company.com',
            'monitor-health.company.com'
        ]
        
        print(f"\n🧪 Validation Results:")
        for subdomain in test_subdomains:
            result = ai.classifier.classify_subdomain(subdomain)
            print(f"  {subdomain} → {result.predicted_category} ({result.confidence:.3f})")
        
        return True
    else:
        print("[-] Model retraining failed!")
        return False

if __name__ == "__main__":
    retrain_netra_model()
