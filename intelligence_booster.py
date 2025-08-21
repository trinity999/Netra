#!/usr/bin/env python3
"""
NETRA Intelligence Booster
=========================

Maximizes NETRA's intelligence by:
- Generating diverse synthetic training data
- Training on realistic subdomain patterns
- Optimizing model parameters
- Building comprehensive knowledge base
"""

import sqlite3
import random
import re
from typing import List, Dict, Tuple
from collections import defaultdict

class IntelligenceBooster:
    """Boosts NETRA's intelligence with comprehensive training data."""
    
    def __init__(self, db_path: str = "subdomain_knowledge.db"):
        self.db_path = db_path
        
        # Advanced subdomain patterns by category
        self.advanced_patterns = {
            'Administrative / Management Interfaces': [
                'admin', 'console', 'dashboard', 'panel', 'control', 'manage', 'mgmt',
                'cpanel', 'plesk', 'webmin', 'phpmyadmin', 'adminer', 'administrator',
                'backend', 'control-panel', 'admin-panel', 'management', 'supervisor',
                'director', 'operator', 'commander', 'master', 'chief', 'super',
                'root', 'sys', 'system', 'config', 'configuration', 'settings'
            ],
            
            'APIs': [
                'api', 'rest', 'graphql', 'soap', 'rpc', 'service', 'services',
                'microservice', 'microservices', 'endpoint', 'endpoints', 'gateway',
                'proxy', 'bridge', 'connector', 'interface', 'webhook', 'webhooks',
                'callback', 'callbacks', 'v1', 'v2', 'v3', 'version', 'latest',
                'stable', 'beta-api', 'alpha-api', 'public-api', 'private-api'
            ],
            
            'Staging / Development / Testing': [
                'dev', 'development', 'staging', 'stage', 'test', 'testing', 'qa',
                'quality', 'sandbox', 'demo', 'preview', 'pre', 'preprod', 'beta',
                'alpha', 'canary', 'experimental', 'lab', 'playground', 'try',
                'sample', 'example', 'prototype', 'pilot', 'trial', 'temp',
                'temporary', 'scratch', 'draft', 'work', 'wip'
            ],
            
            'Authentication / Identity': [
                'auth', 'login', 'signin', 'sso', 'oauth', 'saml', 'ldap', 'ad',
                'identity', 'id', 'accounts', 'account', 'user', 'users', 'member',
                'members', 'profile', 'profiles', 'session', 'sessions', 'token',
                'tokens', 'jwt', 'keycloak', 'okta', 'pingfederate', 'adfs',
                'cas', 'kerberos', 'radius', 'tacacs', 'duo', 'mfa', '2fa'
            ],
            
            'Payment / Transactional': [
                'payment', 'payments', 'pay', 'billing', 'invoice', 'invoices',
                'checkout', 'cart', 'shop', 'store', 'ecommerce', 'commerce',
                'transaction', 'transactions', 'stripe', 'paypal', 'square',
                'adyen', 'braintree', 'worldpay', 'sage', 'authorize', 'cybersource',
                'gateway', 'processor', 'merchant', 'pos', 'wallet', 'finance'
            ],
            
            'CDN / Storage / Assets': [
                'cdn', 'static', 'assets', 'media', 'files', 'uploads', 'images',
                'img', 'pics', 'photos', 'videos', 'audio', 'documents', 'docs',
                's3', 'blob', 'storage', 'bucket', 'repository', 'repo', 'archive',
                'backup', 'backups', 'resources', 'content', 'cache', 'cached',
                'dl', 'download', 'downloads', 'public', 'shared'
            ],
            
            'Database / Data Services': [
                'db', 'database', 'sql', 'mysql', 'postgres', 'postgresql', 'oracle',
                'mssql', 'mongodb', 'mongo', 'redis', 'elastic', 'elasticsearch',
                'solr', 'cassandra', 'couchdb', 'neo4j', 'influxdb', 'timeseries',
                'data', 'analytics', 'warehouse', 'lake', 'mart', 'cube', 'olap',
                'etl', 'pipeline', 'stream', 'kafka', 'rabbit', 'queue'
            ],
            
            'Internal Tools / Infrastructure': [
                'jenkins', 'bamboo', 'teamcity', 'travis', 'circleci', 'gitlab',
                'github', 'bitbucket', 'git', 'svn', 'mercurial', 'ci', 'cd',
                'build', 'deploy', 'deployment', 'artifact', 'artifactory', 'nexus',
                'docker', 'k8s', 'kubernetes', 'helm', 'terraform', 'ansible',
                'puppet', 'chef', 'salt', 'consul', 'vault', 'nomad', 'packer'
            ],
            
            'Marketing / Content / CMS': [
                'blog', 'news', 'press', 'about', 'contact', 'careers', 'jobs',
                'help', 'support', 'faq', 'docs', 'wiki', 'knowledge', 'learn',
                'training', 'education', 'course', 'courses', 'tutorial', 'guide',
                'cms', 'wordpress', 'drupal', 'joomla', 'contentful', 'strapi',
                'ghost', 'medium', 'mailchimp', 'hubspot', 'salesforce', 'marketo'
            ],
            
            'Mobile / Partner / Integration': [
                'mobile', 'app', 'apps', 'android', 'ios', 'partner', 'partners',
                'integration', 'integrations', 'connect', 'sync', 'federation',
                'federated', 'bridge', 'link', 'portal', 'portals', 'external',
                'third-party', 'vendor', 'suppliers', 'clients', 'customers',
                'm', 'wap', 'amp', 'pwa', 'cordova', 'phonegap', 'ionic'
            ],
            
            'Monitoring / Logging': [
                'monitor', 'monitoring', 'metrics', 'stats', 'statistics', 'analytics',
                'logs', 'logging', 'log', 'syslog', 'audit', 'trace', 'tracing',
                'apm', 'performance', 'health', 'status', 'uptime', 'ping', 'check',
                'grafana', 'kibana', 'splunk', 'datadog', 'newrelic', 'dynatrace',
                'prometheus', 'jaeger', 'zipkin', 'elk', 'fluentd', 'logstash'
            ],
            
            'Security Services': [
                'security', 'sec', 'firewall', 'waf', 'ids', 'ips', 'siem',
                'vpn', 'ssl', 'tls', 'cert', 'certificate', 'ca', 'pki',
                'scan', 'scanner', 'vulnerability', 'pentest', 'audit', 'compliance',
                'policy', 'policies', 'acl', 'rbac', 'authorization', 'permission',
                'fortinet', 'paloalto', 'checkpoint', 'symantec', 'mcafee', 'sophos'
            ]
        }
        
        # Common TLDs and domain patterns
        self.tlds = ['com', 'org', 'net', 'edu', 'gov', 'mil', 'io', 'co', 'ai', 'tech', 'dev']
        self.domain_bases = ['example', 'company', 'corp', 'enterprise', 'business', 'org', 'tech', 'startup']
        
        # Environment indicators
        self.environments = ['dev', 'test', 'qa', 'staging', 'prod', 'production', 'demo', 'sandbox']
        
        # Geographic indicators
        self.regions = ['us', 'eu', 'asia', 'apac', 'emea', 'na', 'sa', 'africa', 'au', 'uk', 'de', 'fr', 'jp', 'cn']
        
    def generate_intelligent_training_data(self, samples_per_category: int = 1000) -> List[Tuple[str, str]]:
        """Generate intelligent, realistic training data."""
        training_data = []
        
        print(f"[*] Generating {samples_per_category} samples per category...")
        
        for category, patterns in self.advanced_patterns.items():
            category_samples = []
            
            for _ in range(samples_per_category):
                subdomain = self._generate_realistic_subdomain(patterns, category)
                category_samples.append((subdomain, category))
            
            training_data.extend(category_samples)
            print(f"[+] Generated {len(category_samples)} samples for: {category}")
        
        # Add some random/edge cases
        edge_cases = self._generate_edge_cases(500)
        training_data.extend(edge_cases)
        
        print(f"[+] Total training samples generated: {len(training_data)}")
        return training_data
    
    def _generate_realistic_subdomain(self, patterns: List[str], category: str) -> str:
        """Generate a realistic subdomain based on patterns."""
        # Choose base pattern
        base_pattern = random.choice(patterns)
        
        # Add variations
        variations = [
            base_pattern,  # Simple
            f"{base_pattern}-{random.choice(self.environments)}",  # With environment
            f"{random.choice(self.regions)}-{base_pattern}",  # With region
            f"{base_pattern}{random.randint(1, 9)}",  # With number
            f"{base_pattern}-{random.choice(['web', 'service', 'server', 'app'])}",  # With suffix
            f"new-{base_pattern}",  # With prefix
            f"{base_pattern}-{random.choice(['old', 'legacy', 'v2', 'next'])}",  # With version
        ]
        
        # Choose domain base and TLD
        domain_base = random.choice(self.domain_bases)
        tld = random.choice(self.tlds)
        
        # Sometimes add subdirectories or ports (but keep it realistic)
        subdomain = random.choice(variations)
        
        # Add some randomness for multi-part subdomains
        if random.random() < 0.3:  # 30% chance
            if random.random() < 0.5:
                subdomain = f"{random.choice(['app', 'web', 'service'])}.{subdomain}"
            else:
                subdomain = f"{subdomain}.{random.choice(['internal', 'external', 'public', 'private'])}"
        
        return f"{subdomain}.{domain_base}.{tld}"
    
    def _generate_edge_cases(self, count: int) -> List[Tuple[str, str]]:
        """Generate edge cases and tricky subdomains."""
        edge_cases = []
        
        # Long subdomains
        for _ in range(count // 5):
            long_subdomain = "-".join([random.choice(['very', 'long', 'subdomain', 'name', 'with', 'many', 'parts'])
                                     for _ in range(random.randint(3, 8))])
            subdomain = f"{long_subdomain}.example.com"
            category = random.choice(list(self.advanced_patterns.keys()))
            edge_cases.append((subdomain, category))
        
        # Numeric subdomains
        for _ in range(count // 5):
            numeric = f"{random.randint(1, 999)}.example.com"
            category = 'CDN / Storage / Assets'  # Often numeric
            edge_cases.append((numeric, category))
        
        # Mixed case and special characters
        for _ in range(count // 5):
            mixed = f"Test-{random.randint(1, 99)}_Server.example.com"
            category = 'Staging / Development / Testing'
            edge_cases.append((mixed, category))
        
        # Common false positives
        for _ in range(count // 5):
            false_positives = ['www', 'mail', 'ftp', 'ns1', 'ns2', 'mx1', 'mx2']
            fp_subdomain = f"{random.choice(false_positives)}.example.com"
            category = 'Marketing / Content / CMS'  # Often these are just standard
            edge_cases.append((fp_subdomain, category))
        
        # Remaining random
        for _ in range(count - len(edge_cases)):
            random_pattern = random.choice(list(self.advanced_patterns.values()))
            random_word = random.choice(random_pattern)
            subdomain = f"{random_word}-random.example.com"
            category = random.choice(list(self.advanced_patterns.keys()))
            edge_cases.append((subdomain, category))
        
        return edge_cases
    
    def boost_intelligence(self, samples_per_category: int = 2000):
        """Main intelligence boosting function."""
        print("🚀 NETRA Intelligence Booster")
        print("=" * 40)
        
        # Generate training data
        training_data = self.generate_intelligent_training_data(samples_per_category)
        
        # Add to knowledge base
        print("[*] Adding training data to knowledge base...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        added_count = 0
        for domain, category in training_data:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO subdomains (domain, category, confidence, source)
                    VALUES (?, ?, ?, ?)
                ''', (domain, category, 0.85, 'synthetic_advanced'))
                
                if cursor.rowcount > 0:
                    added_count += 1
                    
            except Exception as e:
                print(f"[-] Error adding {domain}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        print(f"[+] Added {added_count} new training samples to knowledge base")
        
        # Generate statistics
        self._print_statistics()
        
        print("\n🧠 Intelligence boost complete!")
        print("Run enhanced_classifier.py to retrain the model with new data.")
    
    def _print_statistics(self):
        """Print knowledge base statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM subdomains')
        total_subdomains = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT category, COUNT(*) 
            FROM subdomains 
            GROUP BY category 
            ORDER BY COUNT(*) DESC
        ''')
        category_counts = cursor.fetchall()
        
        cursor.execute('''
            SELECT source, COUNT(*) 
            FROM subdomains 
            GROUP BY source 
            ORDER BY COUNT(*) DESC
        ''')
        source_counts = cursor.fetchall()
        
        conn.close()
        
        print(f"\n📊 Knowledge Base Statistics:")
        print(f"Total Subdomains: {total_subdomains:,}")
        
        print(f"\n📂 By Category:")
        for category, count in category_counts:
            print(f"  {category}: {count:,}")
        
        print(f"\n📈 By Source:")
        for source, count in source_counts:
            print(f"  {source}: {count:,}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NETRA Intelligence Booster")
    parser.add_argument('--samples', type=int, default=1500, 
                       help='Samples per category (default: 1500)')
    parser.add_argument('--db', default='subdomain_knowledge.db',
                       help='Knowledge base database path')
    
    args = parser.parse_args()
    
    booster = IntelligenceBooster(args.db)
    booster.boost_intelligence(args.samples)


if __name__ == "__main__":
    main()
