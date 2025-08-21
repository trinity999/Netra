#!/usr/bin/env python3
"""
NETRA - Enhanced Subdomain Classification Demo
=============================================

Demonstrates the enhanced uncertainty detection and multi-category classification
without requiring ML dependencies.
"""

import random
from typing import List, Tuple, Dict

class MockEnhancedClassifier:
    """Mock enhanced classifier for demonstration."""
    
    def __init__(self):
        # Expanded categories with risk levels
        self.categories = {
            'Administrative / Management': {'risk': 'CRITICAL', 'keywords': ['admin', 'dashboard', 'console', 'portal', 'manage', 'control']},
            'APIs / Services': {'risk': 'HIGH', 'keywords': ['api', 'graphql', 'rest', 'service', 'endpoint']},
            'Development / Testing': {'risk': 'HIGH', 'keywords': ['staging', 'dev', 'test', 'qa', 'sandbox', 'beta']},
            'Authentication / Identity': {'risk': 'CRITICAL', 'keywords': ['auth', 'login', 'sso', 'accounts', 'identity']},
            'Financial / Payment': {'risk': 'CRITICAL', 'keywords': ['payment', 'billing', 'checkout', 'pay', 'shop']},
            'Content Delivery / Storage': {'risk': 'MEDIUM', 'keywords': ['cdn', 'static', 'media', 'uploads', 'files']},
            'Database / Data Services': {'risk': 'CRITICAL', 'keywords': ['db', 'sql', 'mysql', 'postgres', 'redis', 'data']},
            'Infrastructure / DevOps': {'risk': 'HIGH', 'keywords': ['jenkins', 'git', 'grafana', 'vpn', 'ci', 'ops']},
            'Marketing / Content': {'risk': 'LOW', 'keywords': ['blog', 'cms', 'wordpress', 'press', 'help']},
            'Mobile / Integration': {'risk': 'MEDIUM', 'keywords': ['mobile', 'app', 'partner', 'integration']},
            'Monitoring / Logging': {'risk': 'LOW', 'keywords': ['status', 'monitor', 'metrics', 'logs', 'health']},
            'Security / Network': {'risk': 'HIGH', 'keywords': ['vpn', 'firewall', 'waf', 'secure', 'ssl']},
            'Mail / Communication': {'risk': 'MEDIUM', 'keywords': ['mail', 'email', 'smtp', 'webmail']},
            'Regional / Localization': {'risk': 'VARIES', 'keywords': ['us', 'eu', 'uk', 'www2', 'www3']}
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.85,
            'medium': 0.70,
            'low': 0.55,
            'uncertain': 0.40
        }
    
    def classify_with_uncertainty(self, subdomain: str) -> Dict:
        """Enhanced classification with uncertainty detection."""
        subdomain_lower = subdomain.lower()
        
        # Find matching categories
        matches = []
        for category, info in self.categories.items():
            match_score = 0
            matched_keywords = []
            
            for keyword in info['keywords']:
                if keyword in subdomain_lower:
                    match_score += 1.0
                    matched_keywords.append(keyword)
            
            if match_score > 0:
                # Normalize score and add some randomness for demo
                confidence = min(0.95, max(0.3, match_score * 0.4 + random.uniform(0.1, 0.5)))
                matches.append((category, confidence, matched_keywords))
        
        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)
        
        if not matches:
            # Default fallback
            primary_category = "Marketing / Content"
            primary_confidence = 0.3
            alternatives = []
            matched_keywords = []
        else:
            primary_category, primary_confidence, matched_keywords = matches[0]
            alternatives = [(cat, conf) for cat, conf, _ in matches[1:3]]
        
        # Uncertainty analysis
        uncertainty_level = self._analyze_uncertainty(primary_confidence, alternatives)
        multi_category = self._check_multi_category(primary_confidence, alternatives)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(primary_confidence, uncertainty_level, multi_category)
        
        return {
            'subdomain': subdomain,
            'primary_category': primary_category,
            'primary_confidence': primary_confidence,
            'alternative_categories': alternatives,
            'uncertainty_level': uncertainty_level,
            'uncertainty_reason': self._get_uncertainty_reason(uncertainty_level, multi_category),
            'multi_category_possible': multi_category,
            'risk_level': self.categories.get(primary_category, {}).get('risk', 'MEDIUM'),
            'matched_keywords': matched_keywords,
            'recommendation': recommendation
        }
    
    def _analyze_uncertainty(self, confidence: float, alternatives: List[Tuple[str, float]]) -> str:
        """Analyze uncertainty level."""
        if confidence >= self.confidence_thresholds['high']:
            return 'low'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        elif confidence >= self.confidence_thresholds['low']:
            return 'high'
        else:
            return 'very_high'
    
    def _check_multi_category(self, primary_confidence: float, alternatives: List[Tuple[str, float]]) -> bool:
        """Check if multiple categories are possible."""
        if not alternatives:
            return False
        
        top_alternative_conf = alternatives[0][1]
        confidence_diff = primary_confidence - top_alternative_conf
        
        return confidence_diff <= 0.15  # Within 15%
    
    def _get_uncertainty_reason(self, uncertainty_level: str, multi_category: bool) -> str:
        """Get reason for uncertainty."""
        if uncertainty_level == 'very_high':
            return 'Very uncertain, manual review recommended'
        elif multi_category:
            return 'Multiple categories possible, close confidence scores'
        elif uncertainty_level == 'high':
            return 'Low confidence, consider alternatives'
        elif uncertainty_level == 'medium':
            return 'Moderate confidence, verify if needed'
        else:
            return 'High confidence in prediction'
    
    def _generate_recommendation(self, confidence: float, uncertainty_level: str, multi_category: bool) -> str:
        """Generate actionable recommendation."""
        if uncertainty_level == 'very_high':
            return "⚠️ MANUAL REVIEW REQUIRED: Very uncertain classification"
        elif multi_category:
            return "🤔 MULTI-CATEGORY POSSIBLE: Consider multiple classifications"
        elif uncertainty_level == 'high':
            return "⚡ LOW CONFIDENCE: Verify classification manually"
        elif confidence >= self.confidence_thresholds['high']:
            return "✅ HIGH CONFIDENCE: Classification reliable"
        else:
            return "✓ MODERATE CONFIDENCE: Classification likely correct"

def demo_enhanced_classification():
    """Demonstrate enhanced classification capabilities."""
    print("🧠 Enhanced Subdomain Classification Demo")
    print("=" * 50)
    print("Features demonstrated:")
    print("• Expanded categorization (14 categories)")
    print("• Uncertainty detection and honest reporting") 
    print("• Multi-category possibility detection")
    print("• Risk level assessment")
    print("• Actionable recommendations")
    print("• False positive prevention")
    print("=" * 50)
    
    # Sample subdomains for testing
    test_subdomains = [
        "admin.example.com",
        "api-staging.example.com", 
        "auth-sso.example.com",
        "payment-gateway.example.com",
        "db-prod.example.com",
        "jenkins-ci.example.com",
        "www-beta.example.com",
        "unknown-service.example.com",
        "mail-server.example.com",
        "cdn-static.example.com"
    ]
    
    classifier = MockEnhancedClassifier()
    
    print(f"\n🔍 Analyzing {len(test_subdomains)} test subdomains...\n")
    
    # Classify each subdomain
    results = []
    for subdomain in test_subdomains:
        result = classifier.classify_with_uncertainty(subdomain)
        results.append(result)
        
        # Print individual result
        print(f"📋 {subdomain}")
        print(f"   Category: {result['primary_category']} ({result['primary_confidence']:.3f})")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Uncertainty: {result['uncertainty_level']} - {result['uncertainty_reason']}")
        if result['alternative_categories']:
            alts = [f"{cat} ({conf:.3f})" for cat, conf in result['alternative_categories']]
            print(f"   Alternatives: {', '.join(alts)}")
        if result['matched_keywords']:
            print(f"   Keywords: {', '.join(result['matched_keywords'])}")
        print(f"   📝 {result['recommendation']}")
        print()
    
    # Summary statistics
    print("📊 ANALYSIS SUMMARY")
    print("-" * 30)
    
    # Count by uncertainty level
    uncertainty_counts = {}
    for result in results:
        level = result['uncertainty_level']
        uncertainty_counts[level] = uncertainty_counts.get(level, 0) + 1
    
    print("Uncertainty Distribution:")
    for level, count in uncertainty_counts.items():
        print(f"  {level.title()}: {count}")
    
    # Count by risk level
    risk_counts = {}
    for result in results:
        risk = result['risk_level']
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    print("\nRisk Distribution:")
    for risk, count in sorted(risk_counts.items(), key=lambda x: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'VARIES'].index(x[0])):
        print(f"  {risk}: {count}")
    
    # Multi-category possibilities
    multi_category_count = sum(1 for r in results if r['multi_category_possible'])
    manual_review_count = sum(1 for r in results if r['uncertainty_level'] == 'very_high')
    
    print(f"\nSpecial Cases:")
    print(f"  Multi-category possible: {multi_category_count}")
    print(f"  Manual review needed: {manual_review_count}")
    
    # High-priority targets
    high_priority = [r for r in results if r['risk_level'] in ['CRITICAL', 'HIGH']]
    print(f"\nHigh-Priority Targets: {len(high_priority)}")
    for result in high_priority:
        print(f"  🎯 {result['subdomain']} → {result['primary_category']} ({result['risk_level']})")

if __name__ == "__main__":
    demo_enhanced_classification()
