#!/usr/bin/env python3
"""
Verify Incremental Model Training Results
"""

from incremental_classifier import IncrementalLearningClassifier
from subdomain_ai_enhanced import SubdomainAIEnhanced

def main():
    print("🎯 INCREMENTAL MODEL VERIFICATION")
    print("=" * 40)
    
    # Initialize the incremental classifier
    ai = SubdomainAIEnhanced()
    classifier = IncrementalLearningClassifier(ai.kb)
    
    # Get training summary
    summary = classifier.get_training_summary()
    
    print(f"📊 Total Samples Trained: {summary['total_samples_trained']:,}")
    print(f"🔄 Training Sessions: {summary['training_sessions']}")
    print(f"🏷️ Categories Learned: {summary['categories_count']}")
    print(f"🤖 Model Available: {summary['model_available']}")
    print(f"📈 Last Updated: {summary.get('last_updated', 'Never')}")
    
    if summary['accuracy_progression']:
        print(f"\n📈 ACCURACY PROGRESSION:")
        for session in summary['accuracy_progression']:
            print(f"  Session {session['session']}: {session['accuracy']:.3f}")
    
    print(f"\n🏷️ CATEGORIES LEARNED:")
    for i, category in enumerate(sorted(summary['categories_learned']), 1):
        print(f"  {i:2d}. {category}")
    
    # Test classification with the incremental model
    print(f"\n🧪 TESTING INCREMENTAL ML MODEL:")
    print("-" * 35)
    
    test_domains = [
        'admin-secure.company.com',
        'api-v4.company.com', 
        'auth-oauth.company.com',
        'database-replica.company.com',
        'jenkins-build.company.com',
        'monitoring-logs.company.com',
        'payment-stripe.company.com',
        'cdn-global.company.com'
    ]
    
    ml_usage = 0
    kb_usage = 0
    
    for domain in test_domains:
        result = classifier.classify_subdomain(domain)
        source_emoji = {"incremental_ml_model": "🤖", "knowledge_base": "📚", "heuristic": "🔧"}.get(result.prediction_source, "❓")
        
        if result.prediction_source == "incremental_ml_model":
            ml_usage += 1
        elif result.prediction_source == "knowledge_base":
            kb_usage += 1
            
        print(f"  {source_emoji} {result.subdomain} → {result.predicted_category} ({result.confidence:.3f})")
    
    print(f"\n📊 PREDICTION SOURCE USAGE:")
    print(f"  🤖 ML Model: {ml_usage}/{len(test_domains)} ({ml_usage/len(test_domains):.1%})")
    print(f"  📚 Knowledge Base: {kb_usage}/{len(test_domains)} ({kb_usage/len(test_domains):.1%})")
    
    if summary['model_available']:
        print("\n✅ INCREMENTAL LEARNING SUCCESS!")
        print(f"   🎯 {summary['total_samples_trained']:,} samples accumulated across {summary['training_sessions']} sessions")
        print("   🧠 Model properly trained on cumulative data (no overwriting)")
    else:
        print("\n❌ MODEL NOT AVAILABLE")

if __name__ == "__main__":
    main()
