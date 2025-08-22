#!/usr/bin/env python3
"""
Knowledge Base Seeder for Subdomain AI Enhanced
==============================================

This script seeds the knowledge base with initial training data
to bootstrap the learning process.
"""

import random
import json
from typing import List, Tuple
from subdomain_ai_enhanced import SubdomainAIEnhanced, KnowledgeBaseManager

def generate_synthetic_training_data() -> List[Tuple[str, str]]:
    """Generate synthetic subdomain training data."""
    
    # Category patterns and examples
    category_patterns = {
        'Administrative / Management Interfaces': {
            'patterns': ['admin', 'dashboard', 'console', 'portal', 'cpanel', 'manage', 'control', 'panel'],
            'domains': ['example.com', 'company.org', 'site.net', 'app.io', 'service.co']
        },
        'APIs': {
            'patterns': ['api', 'graphql', 'rest', 'service', 'endpoint', 'microservice', 'webhook'],
            'domains': ['platform.com', 'service.io', 'app.dev', 'system.net', 'tech.co']
        },
        'Staging / Development / Testing': {
            'patterns': ['staging', 'dev', 'test', 'qa', 'sandbox', 'preprod', 'beta', 'alpha', 'demo'],
            'domains': ['company.com', 'product.io', 'service.dev', 'app.test', 'platform.net']
        },
        'Authentication / Identity': {
            'patterns': ['auth', 'login', 'sso', 'accounts', 'idp', 'oauth', 'saml', 'identity'],
            'domains': ['secure.com', 'identity.org', 'auth.io', 'login.net', 'sso.co']
        },
        'Payment / Transactional': {
            'patterns': ['payments', 'billing', 'checkout', 'invoice', 'pay', 'stripe', 'paypal'],
            'domains': ['commerce.com', 'shop.io', 'store.net', 'payment.co', 'billing.org']
        },
        'CDN / Storage / Assets': {
            'patterns': ['cdn', 'static', 'media', 'uploads', 'files', 'assets', 'images', 'js', 'css'],
            'domains': ['content.com', 'media.io', 'assets.net', 'cdn.co', 'static.org']
        },
        'Database / Data Services': {
            'patterns': ['db', 'sql', 'mysql', 'postgres', 'mongo', 'redis', 'elastic', 'data'],
            'domains': ['data.com', 'db.io', 'storage.net', 'database.co', 'analytics.org']
        },
        'Internal Tools / Infrastructure': {
            'patterns': ['jira', 'jenkins', 'git', 'grafana', 'kibana', 'vpn', 'ci', 'build', 'deploy'],
            'domains': ['internal.com', 'tools.io', 'infra.net', 'ops.co', 'devops.org']
        },
        'Marketing / Content / CMS': {
            'patterns': ['blog', 'cms', 'wordpress', 'drupal', 'press', 'careers', 'about', 'help'],
            'domains': ['website.com', 'blog.io', 'content.net', 'marketing.co', 'cms.org']
        },
        'Mobile / Partner / Integration': {
            'patterns': ['mobile', 'app', 'android', 'ios', 'partner', 'integration', 'm'],
            'domains': ['mobile.com', 'app.io', 'partner.net', 'integration.co', 'api.org']
        },
        'Monitoring / Logging': {
            'patterns': ['status', 'monitor', 'metrics', 'logs', 'uptime', 'health', 'ping'],
            'domains': ['monitor.com', 'status.io', 'metrics.net', 'health.co', 'uptime.org']
        },
        'Security Services': {
            'patterns': ['vpn', 'firewall', 'waf', 'secure', 'ssl', 'cert', 'security'],
            'domains': ['secure.com', 'security.io', 'vpn.net', 'ssl.co', 'firewall.org']
        }
    }
    
    training_data = []
    
    # Generate synthetic subdomains
    for category, data in category_patterns.items():
        patterns = data['patterns']
        domains = data['domains']
        
        # Generate base patterns
        for pattern in patterns:
            for domain in domains:
                # Basic pattern
                subdomain = f"{pattern}.{domain}"
                training_data.append((subdomain, category))
                
                # Pattern with numbers
                subdomain = f"{pattern}{random.randint(1, 5)}.{domain}"
                training_data.append((subdomain, category))
                
                # Pattern with environment
                env_suffix = random.choice(['prod', 'staging', 'dev', 'test'])
                subdomain = f"{pattern}-{env_suffix}.{domain}"
                training_data.append((subdomain, category))
                
                # Pattern with region
                region = random.choice(['us', 'eu', 'asia', 'west', 'east'])
                subdomain = f"{pattern}-{region}.{domain}"
                training_data.append((subdomain, category))
        
        # Generate compound patterns
        for i in range(5):
            pattern1 = random.choice(patterns)
            pattern2 = random.choice(['api', 'web', 'app', 'service'])
            domain = random.choice(domains)
            
            # Compound with hyphen
            subdomain = f"{pattern1}-{pattern2}.{domain}"
            training_data.append((subdomain, category))
            
            # Compound with number
            subdomain = f"{pattern1}{random.randint(1, 3)}{pattern2}.{domain}"
            training_data.append((subdomain, category))
    
    return training_data

def create_ground_truth_file(training_data: List[Tuple[str, str]], filename: str = "ground_truth.tsv"):
    """Create a ground truth file for benchmarking."""
    with open(filename, 'w') as f:
        for subdomain, category in training_data:
            f.write(f"{subdomain}\t{category}\n")
    
    print(f"[+] Ground truth file created: {filename} with {len(training_data)} entries")

def seed_knowledge_base():
    """Seed the knowledge base with initial training data."""
    print("🌱 Seeding Knowledge Base for Subdomain AI Enhanced")
    print("=" * 55)
    
    # Generate training data
    print("[*] Generating synthetic training data...")
    training_data = generate_synthetic_training_data()
    print(f"[+] Generated {len(training_data)} training samples")
    
    # Initialize AI system
    ai = SubdomainAIEnhanced()
    
    # Add data to knowledge base
    print("[*] Adding data to knowledge base...")
    for subdomain, category in training_data:
        ai.kb.add_subdomain(subdomain, category, confidence=0.9, source='synthetic')
    
    print("[+] Knowledge base seeded successfully!")
    
    # Train initial model
    print("[*] Training initial ML model...")
    success = ai.classifier.train_model(training_data, test_size=0.2)
    
    if success:
        print("[+] Initial model training completed!")
    else:
        print("[-] Model training failed - ML libraries may not be available")
    
    # Create test files
    print("[*] Creating test files...")
    
    # Split data for testing
    test_size = len(training_data) // 4
    test_data = random.sample(training_data, test_size)
    
    # Create subdomain test file
    with open("test_subdomains.txt", 'w') as f:
        for subdomain, _ in test_data:
            f.write(f"{subdomain}\n")
    
    # Create ground truth file
    create_ground_truth_file(test_data, "ground_truth.tsv")
    
    print(f"[+] Test files created:")
    print(f"    - test_subdomains.txt ({len(test_data)} subdomains)")
    print(f"    - ground_truth.tsv ({len(test_data)} entries)")
    
    return ai

def demo_analysis():
    """Demonstrate the analysis capabilities."""
    print("\n🔍 Demo Analysis")
    print("=" * 20)
    
    ai = SubdomainAIEnhanced()
    
    # Demo subdomains
    demo_subdomains = [
        "api-v2.example.com",
        "admin-dashboard.company.io",
        "staging-db.service.net",
        "auth-sso.platform.co",
        "payment-gateway.shop.org",
        "cdn-static.media.io",
        "jenkins-ci.internal.com",
        "blog-cms.website.net",
        "mobile-api.app.co",
        "status-monitor.uptime.org"
    ]
    
    print(f"[*] Analyzing {len(demo_subdomains)} demo subdomains...")
    results = ai.classify_subdomains(demo_subdomains)
    
    print("\n[+] Analysis Results:")
    print("-" * 80)
    print(f"{'Subdomain':<25} {'Category':<35} {'Confidence':<10} {'Source':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result.subdomain:<25} {result.predicted_category:<35} "
              f"{result.confidence:<10.3f} {result.prediction_source:<15}")
    
    # Summary
    sources = {}
    for result in results:
        sources[result.prediction_source] = sources.get(result.prediction_source, 0) + 1
    
    print("\n[+] Prediction Sources:")
    for source, count in sources.items():
        print(f"    {source}: {count}")

def main():
    """Main seeding function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed knowledge base for Subdomain AI Enhanced")
    parser.add_argument('--demo', action='store_true', help='Run demo analysis')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark test')
    
    args = parser.parse_args()
    
    # Seed knowledge base
    ai = seed_knowledge_base()
    
    if args.demo:
        demo_analysis()
    
    if args.benchmark:
        print("\n📊 Running Benchmark")
        print("=" * 22)
        metrics = ai.benchmark("test_subdomains.txt", "ground_truth.tsv")
        
        print(f"\n[+] Final Benchmark Results:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.3f}")

if __name__ == "__main__":
    main()
