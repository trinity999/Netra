#!/usr/bin/env python3
"""
NETRA - Enhanced Subdomain Classifier with Uncertainty Detection
==============================================================

Addresses false positive prevention and honest uncertainty reporting
when multiple categories are possible or confidence is low.
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from subdomain_ai_enhanced import SubdomainAIEnhanced, ClassificationResult
import numpy as np

@dataclass
class EnhancedClassificationResult:
    """Enhanced classification result with uncertainty handling."""
    subdomain: str
    primary_category: str
    primary_confidence: float
    alternative_categories: List[Tuple[str, float]]
    uncertainty_level: str  # 'low', 'medium', 'high', 'very_high'
    uncertainty_reason: str
    multi_category_possible: bool
    confidence_threshold_met: bool
    prediction_source: str
    features_used: Dict
    recommendation: str  # Action recommendation for the user

class EnhancedSubdomainClassifier:
    """Enhanced classifier with uncertainty detection and expanded categorization."""
    
    def __init__(self):
        self.ai = SubdomainAIEnhanced()
        
        # Confidence thresholds to prevent false positives
        self.confidence_thresholds = {
            'high_confidence': 0.85,      # Very confident in prediction
            'medium_confidence': 0.70,    # Moderately confident
            'low_confidence': 0.55,       # Low confidence threshold
            'uncertain': 0.40            # Below this = highly uncertain
        }
        
        # Multi-category similarity threshold
        self.multi_category_threshold = 0.15  # If top 2 predictions are within 15%, flag as multi-category
        
        # Expanded categories with subcategories
        self.expanded_categories = self._define_expanded_categories()
        
        # Risk levels for each category
        self.category_risk_levels = self._define_risk_levels()
    
    def _define_expanded_categories(self) -> Dict[str, Dict]:
        """Define expanded categories with subcategories and indicators."""
        return {
            'Administrative / Management': {
                'subcategories': [
                    'Admin Panels',
                    'Management Dashboards', 
                    'Control Interfaces',
                    'Configuration Portals'
                ],
                'keywords': ['admin', 'dashboard', 'console', 'portal', 'cpanel', 'manage', 'control', 'panel', 'backend', 'backoffice'],
                'risk_level': 'CRITICAL'
            },
            'APIs / Services': {
                'subcategories': [
                    'REST APIs',
                    'GraphQL Endpoints',
                    'Microservices',
                    'Webhooks',
                    'Internal APIs'
                ],
                'keywords': ['api', 'graphql', 'rest', 'service', 'endpoint', 'microservice', 'webhook', 'ws', 'rpc'],
                'risk_level': 'HIGH'
            },
            'Development / Testing': {
                'subcategories': [
                    'Staging Environments',
                    'Development Servers',
                    'Testing Platforms',
                    'QA Systems',
                    'Demo Sites'
                ],
                'keywords': ['staging', 'dev', 'test', 'qa', 'sandbox', 'preprod', 'beta', 'alpha', 'demo', 'devel'],
                'risk_level': 'HIGH'
            },
            'Authentication / Identity': {
                'subcategories': [
                    'SSO Systems',
                    'Login Portals',
                    'Identity Providers',
                    'OAuth Services',
                    'SAML Endpoints'
                ],
                'keywords': ['auth', 'login', 'sso', 'accounts', 'idp', 'oauth', 'saml', 'identity', 'signin', 'authentication'],
                'risk_level': 'CRITICAL'
            },
            'Financial / Payment': {
                'subcategories': [
                    'Payment Gateways',
                    'Billing Systems',
                    'E-commerce',
                    'Financial APIs',
                    'Transaction Processing'
                ],
                'keywords': ['payment', 'billing', 'checkout', 'invoice', 'pay', 'stripe', 'paypal', 'shop', 'store', 'commerce'],
                'risk_level': 'CRITICAL'
            },
            'Content Delivery / Storage': {
                'subcategories': [
                    'CDN Endpoints',
                    'File Storage',
                    'Media Servers',
                    'Asset Delivery',
                    'Static Content'
                ],
                'keywords': ['cdn', 'static', 'media', 'uploads', 'files', 'assets', 'images', 'js', 'css', 'storage'],
                'risk_level': 'MEDIUM'
            },
            'Database / Data Services': {
                'subcategories': [
                    'Database Servers',
                    'Data APIs',
                    'Analytics Platforms',
                    'Search Services',
                    'Cache Systems'
                ],
                'keywords': ['db', 'database', 'sql', 'mysql', 'postgres', 'mongo', 'redis', 'elastic', 'data', 'analytics'],
                'risk_level': 'CRITICAL'
            },
            'Infrastructure / DevOps': {
                'subcategories': [
                    'CI/CD Systems',
                    'Monitoring Tools',
                    'VPN Gateways',
                    'Build Systems',
                    'Deployment Tools'
                ],
                'keywords': ['jira', 'jenkins', 'git', 'grafana', 'kibana', 'vpn', 'ci', 'build', 'deploy', 'infra', 'ops'],
                'risk_level': 'HIGH'
            },
            'Marketing / Content': {
                'subcategories': [
                    'CMS Systems',
                    'Blog Platforms',
                    'Marketing Sites',
                    'Corporate Pages',
                    'Help Systems'
                ],
                'keywords': ['blog', 'cms', 'wordpress', 'drupal', 'press', 'careers', 'about', 'help', 'support', 'news'],
                'risk_level': 'LOW'
            },
            'Mobile / Integration': {
                'subcategories': [
                    'Mobile APIs',
                    'App Backends',
                    'Partner Integration',
                    'Third-party Services',
                    'Platform Integration'
                ],
                'keywords': ['mobile', 'app', 'android', 'ios', 'partner', 'integration', 'm', 'platform', 'sdk'],
                'risk_level': 'MEDIUM'
            },
            'Monitoring / Logging': {
                'subcategories': [
                    'Status Pages',
                    'Health Checks',
                    'Metrics Collection',
                    'Log Aggregation',
                    'Performance Monitoring'
                ],
                'keywords': ['status', 'monitor', 'metrics', 'logs', 'uptime', 'health', 'ping', 'check', 'stats'],
                'risk_level': 'LOW'
            },
            'Security / Network': {
                'subcategories': [
                    'Security Services',
                    'Firewall Management',
                    'SSL/TLS Services',
                    'Network Security',
                    'Compliance Systems'
                ],
                'keywords': ['vpn', 'firewall', 'waf', 'secure', 'ssl', 'cert', 'security', 'tls', 'proxy'],
                'risk_level': 'HIGH'
            },
            'Mail / Communication': {
                'subcategories': [
                    'Email Servers',
                    'Webmail',
                    'Communication Tools',
                    'Messaging Systems',
                    'Collaboration Platforms'
                ],
                'keywords': ['mail', 'email', 'smtp', 'webmail', 'message', 'chat', 'communication', 'collaborate'],
                'risk_level': 'MEDIUM'
            },
            'Regional / Localization': {
                'subcategories': [
                    'Geographic Instances',
                    'Language Variants',
                    'Regional Services',
                    'Localized Content',
                    'Country-specific'
                ],
                'keywords': ['us', 'eu', 'asia', 'uk', 'de', 'fr', 'jp', 'cn', 'au', 'ca', 'www2', 'www3'],
                'risk_level': 'VARIES'
            }
        }
    
    def _define_risk_levels(self) -> Dict[str, str]:
        """Define risk levels for categories."""
        return {
            'CRITICAL': 'High-value target, immediate security review required',
            'HIGH': 'Significant security concern, priority review',
            'MEDIUM': 'Moderate risk, include in security assessment',
            'LOW': 'Lower risk, routine security practices apply',
            'VARIES': 'Risk depends on context and primary function'
        }
    
    def classify_with_uncertainty(self, subdomain: str) -> EnhancedClassificationResult:
        """Classify subdomain with enhanced uncertainty detection."""
        
        # Get base classification
        base_result = self.ai.classifier.classify_subdomain(subdomain)
        
        # Analyze uncertainty
        uncertainty_analysis = self._analyze_uncertainty(base_result, subdomain)
        
        # Check for multi-category possibilities
        multi_category_analysis = self._check_multi_category(base_result, subdomain)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(base_result, uncertainty_analysis, multi_category_analysis)
        
        return EnhancedClassificationResult(
            subdomain=subdomain,
            primary_category=base_result.predicted_category,
            primary_confidence=base_result.confidence,
            alternative_categories=base_result.alternative_categories,
            uncertainty_level=uncertainty_analysis['level'],
            uncertainty_reason=uncertainty_analysis['reason'],
            multi_category_possible=multi_category_analysis['possible'],
            confidence_threshold_met=base_result.confidence >= self.confidence_thresholds['medium_confidence'],
            prediction_source=base_result.prediction_source,
            features_used=base_result.features_used,
            recommendation=recommendation
        )
    
    def _analyze_uncertainty(self, result: ClassificationResult, subdomain: str) -> Dict[str, str]:
        """Analyze uncertainty level and reasons."""
        confidence = result.confidence
        
        if confidence >= self.confidence_thresholds['high_confidence']:
            return {'level': 'low', 'reason': 'High confidence in prediction'}
        elif confidence >= self.confidence_thresholds['medium_confidence']:
            return {'level': 'medium', 'reason': 'Moderate confidence, consider alternatives'}
        elif confidence >= self.confidence_thresholds['low_confidence']:
            return {'level': 'high', 'reason': 'Low confidence, multiple categories possible'}
        else:
            return {'level': 'very_high', 'reason': 'Very uncertain, manual review recommended'}
    
    def _check_multi_category(self, result: ClassificationResult, subdomain: str) -> Dict[str, bool]:
        """Check if subdomain could belong to multiple categories."""
        if len(result.alternative_categories) < 1:
            return {'possible': False, 'reason': 'No strong alternatives'}
        
        # Check if top alternative is close to primary prediction
        top_alternative = result.alternative_categories[0]
        confidence_diff = result.confidence - top_alternative[1]
        
        if confidence_diff <= self.multi_category_threshold:
            return {
                'possible': True, 
                'reason': f'Primary ({result.confidence:.3f}) and alternative ({top_alternative[1]:.3f}) are close'
            }
        
        return {'possible': False, 'reason': 'Clear primary category'}
    
    def _generate_recommendation(self, result: ClassificationResult, uncertainty: Dict, multi_category: Dict) -> str:
        """Generate actionable recommendation for the user."""
        confidence = result.confidence
        
        if uncertainty['level'] == 'very_high':
            return "⚠️ MANUAL REVIEW REQUIRED: Very uncertain classification"
        elif multi_category['possible']:
            alts = [f"{cat} ({conf:.3f})" for cat, conf in result.alternative_categories[:2]]
            return f"🤔 MULTI-CATEGORY POSSIBLE: Consider both {result.predicted_category} and {', '.join(alts)}"
        elif uncertainty['level'] == 'high':
            return "⚡ LOW CONFIDENCE: Verify classification manually"
        elif confidence >= self.confidence_thresholds['high_confidence']:
            return "✅ HIGH CONFIDENCE: Classification reliable"
        else:
            return "✓ MODERATE CONFIDENCE: Classification likely correct"
    
    def batch_classify_with_stats(self, subdomains: List[str]) -> Dict:
        """Classify batch of subdomains and return comprehensive stats."""
        results = []
        stats = {
            'total_processed': 0,
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0, 'very_low': 0},
            'uncertainty_distribution': {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0},
            'multi_category_count': 0,
            'manual_review_needed': 0,
            'category_distribution': {},
            'risk_distribution': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'VARIES': 0},
            'prediction_sources': {'knowledge_base': 0, 'ml_model': 0, 'heuristic': 0}
        }
        
        print(f"[*] Classifying {len(subdomains)} subdomains with uncertainty analysis...")
        
        for i, subdomain in enumerate(subdomains):
            result = self.classify_with_uncertainty(subdomain)
            results.append(result)
            
            # Update stats
            stats['total_processed'] += 1
            
            # Confidence distribution
            conf = result.primary_confidence
            if conf >= 0.85:
                stats['confidence_distribution']['high'] += 1
            elif conf >= 0.70:
                stats['confidence_distribution']['medium'] += 1  
            elif conf >= 0.55:
                stats['confidence_distribution']['low'] += 1
            else:
                stats['confidence_distribution']['very_low'] += 1
            
            # Uncertainty distribution
            stats['uncertainty_distribution'][result.uncertainty_level] += 1
            
            # Multi-category and manual review
            if result.multi_category_possible:
                stats['multi_category_count'] += 1
            if result.uncertainty_level == 'very_high':
                stats['manual_review_needed'] += 1
            
            # Category distribution
            category = result.primary_category
            stats['category_distribution'][category] = stats['category_distribution'].get(category, 0) + 1
            
            # Risk distribution
            risk_level = self._get_risk_level(category)
            stats['risk_distribution'][risk_level] += 1
            
            # Prediction source
            stats['prediction_sources'][result.prediction_source] += 1
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"[*] Processed {i + 1}/{len(subdomains)}...")
        
        return {'results': results, 'stats': stats}
    
    def _get_risk_level(self, category: str) -> str:
        """Get risk level for a category."""
        for cat_name, cat_info in self.expanded_categories.items():
            if category.startswith(cat_name) or cat_name in category:
                return cat_info['risk_level']
        return 'MEDIUM'  # Default
    
    def print_analysis_report(self, batch_result: Dict):
        """Print comprehensive analysis report."""
        stats = batch_result['stats']
        results = batch_result['results']
        
        print("\n" + "="*60)
        print("🔍 ENHANCED SUBDOMAIN ANALYSIS REPORT")
        print("="*60)
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"Total Subdomains: {stats['total_processed']:,}")
        print(f"Manual Review Needed: {stats['manual_review_needed']} ({stats['manual_review_needed']/stats['total_processed']*100:.1f}%)")
        print(f"Multi-Category Possible: {stats['multi_category_count']} ({stats['multi_category_count']/stats['total_processed']*100:.1f}%)")
        
        # Confidence Distribution
        print(f"\n🎯 CONFIDENCE DISTRIBUTION:")
        for level, count in stats['confidence_distribution'].items():
            percentage = count / stats['total_processed'] * 100
            print(f"  {level.title()}: {count} ({percentage:.1f}%)")
        
        # Uncertainty Distribution
        print(f"\n⚠️ UNCERTAINTY LEVELS:")
        for level, count in stats['uncertainty_distribution'].items():
            percentage = count / stats['total_processed'] * 100
            print(f"  {level.title()}: {count} ({percentage:.1f}%)")
        
        # Risk Distribution
        print(f"\n🚨 SECURITY RISK LEVELS:")
        for risk, count in sorted(stats['risk_distribution'].items(), key=lambda x: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'VARIES'].index(x[0])):
            percentage = count / stats['total_processed'] * 100
            print(f"  {risk}: {count} ({percentage:.1f}%)")
        
        # Top Categories
        print(f"\n📂 TOP CATEGORIES:")
        sorted_categories = sorted(stats['category_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
        for category, count in sorted_categories:
            percentage = count / stats['total_processed'] * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # High-Priority Targets
        critical_high_risk = [r for r in results if self._get_risk_level(r.primary_category) in ['CRITICAL', 'HIGH']]
        if critical_high_risk:
            print(f"\n🎯 HIGH-PRIORITY TARGETS ({len(critical_high_risk)}):")
            for result in critical_high_risk[:15]:  # Show top 15
                risk = self._get_risk_level(result.primary_category)
                print(f"  {result.subdomain} → {result.primary_category} ({risk}) [{result.primary_confidence:.3f}]")
        
        # Uncertain Classifications
        uncertain = [r for r in results if r.uncertainty_level in ['high', 'very_high']]
        if uncertain:
            print(f"\n❓ UNCERTAIN CLASSIFICATIONS ({len(uncertain)}):")
            for result in uncertain[:10]:  # Show top 10
                print(f"  {result.subdomain} → {result.primary_category} [{result.primary_confidence:.3f}] - {result.uncertainty_reason}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if stats['manual_review_needed'] > 0:
            print(f"  • Review {stats['manual_review_needed']} uncertain classifications manually")
        if stats['multi_category_count'] > 0:
            print(f"  • Consider {stats['multi_category_count']} subdomains for multiple categories")
        if len(critical_high_risk) > 0:
            print(f"  • Prioritize security review of {len(critical_high_risk)} critical/high-risk targets")
        if stats['confidence_distribution']['very_low'] > stats['total_processed'] * 0.1:
            print(f"  • High number of very low confidence predictions - consider more training data")

def main():
    """Main function for testing enhanced classifier."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Subdomain Classifier with Uncertainty Detection")
    parser.add_argument('--file', required=True, help='Subdomain file to analyze')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--limit', type=int, help='Limit number of subdomains to process')
    
    args = parser.parse_args()
    
    # Load subdomains
    with open(args.file, 'r') as f:
        subdomains = [line.strip() for line in f if line.strip()]
    
    if args.limit:
        subdomains = subdomains[:args.limit]
    
    print(f"🧠 Enhanced Subdomain Analysis Starting...")
    print(f"Processing {len(subdomains)} subdomains from {args.file}")
    
    # Initialize enhanced classifier
    classifier = EnhancedSubdomainClassifier()
    
    # Perform batch classification
    batch_result = classifier.batch_classify_with_stats(subdomains)
    
    # Print report
    classifier.print_analysis_report(batch_result)
    
    # Save results if requested
    if args.output:
        # Convert results to JSON-serializable format
        json_results = []
        for result in batch_result['results']:
            json_results.append({
                'subdomain': result.subdomain,
                'primary_category': result.primary_category,
                'primary_confidence': result.primary_confidence,
                'alternative_categories': result.alternative_categories,
                'uncertainty_level': result.uncertainty_level,
                'uncertainty_reason': result.uncertainty_reason,
                'multi_category_possible': result.multi_category_possible,
                'confidence_threshold_met': result.confidence_threshold_met,
                'prediction_source': result.prediction_source,
                'recommendation': result.recommendation
            })
        
        output_data = {
            'results': json_results,
            'stats': batch_result['stats']
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n[+] Results saved to: {args.output}")

if __name__ == "__main__":
    main()
