#!/usr/bin/env python3
"""
NETRA Mass H1 Subdomain Analyzer
================================

Analyzes subdomain data from HackerOne bug bounty programs 
to build comprehensive intelligence and feed NETRA's knowledge base.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from enhanced_classifier import EnhancedSubdomainClassifier
from subdomain_ai_enhanced import SubdomainAIEnhanced
import sqlite3

class MassH1Analyzer:
    """Analyzes multiple H1 bug bounty programs for subdomain intelligence."""
    
    def __init__(self, h1_base_path: str):
        self.h1_base_path = Path(h1_base_path)
        self.classifier = EnhancedSubdomainClassifier()
        self.ai = SubdomainAIEnhanced()
        self.results = {}
        
        # High-value targets for priority analysis
        self.priority_companies = [
            'github', 'slack', 'spotify', 'uber', 'airbnb', 'stripe', 'paypal',
            'amazon', 'google', 'microsoft', 'cloudflare', 'twitter', 'reddit',
            'shopify', 'gitlab', 'alibaba', 'tiktok', 'snapchat', 'instagram'
        ]
    
    def discover_programs(self) -> List[Tuple[str, int]]:
        """Discover all H1 programs and their subdomain counts."""
        programs = []
        
        for program_dir in self.h1_base_path.iterdir():
            if not program_dir.is_dir():
                continue
                
            assets_file = program_dir / "assets.txt"
            if assets_file.exists():
                # Count lines in assets file
                try:
                    with open(assets_file, 'r', encoding='utf-8', errors='ignore') as f:
                        subdomain_count = sum(1 for line in f if line.strip())
                    programs.append((program_dir.name, subdomain_count))
                except Exception as e:
                    print(f"[-] Error reading {assets_file}: {e}")
                    continue
        
        # Sort by subdomain count (descending)
        programs.sort(key=lambda x: x[1], reverse=True)
        return programs
    
    def analyze_program(self, program_name: str, limit: int = 500) -> Dict:
        """Analyze a single bug bounty program."""
        program_dir = self.h1_base_path / program_name
        assets_file = program_dir / "assets.txt"
        
        if not assets_file.exists():
            return {'error': f'Assets file not found for {program_name}'}
        
        print(f"\n🎯 Analyzing {program_name}...")
        
        # Load subdomains
        subdomains = []
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                with open(assets_file, 'r', encoding=encoding, errors='ignore') as f:
                    subdomains = [line.strip() for line in f if line.strip() and not line.startswith('#')][:limit]
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if not subdomains:
            return {'error': f'Could not read subdomains for {program_name}'}
        
        print(f"[*] Loaded {len(subdomains)} subdomains for analysis")
        
        # Perform enhanced analysis
        try:
            batch_result = self.classifier.batch_classify_with_stats(subdomains)
            
            # Add program metadata
            analysis_result = {
                'program_name': program_name,
                'total_subdomains': len(subdomains),
                'analysis_timestamp': time.time(),
                'stats': batch_result['stats'],
                'top_categories': sorted(batch_result['stats']['category_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5],
                'high_risk_count': batch_result['stats']['risk_distribution']['CRITICAL'] + 
                                 batch_result['stats']['risk_distribution']['HIGH'],
                'critical_subdomains': [r for r in batch_result['results'] 
                                      if self.classifier._get_risk_level(r.primary_category) == 'CRITICAL'][:10],
                'uncertain_subdomains': [r for r in batch_result['results'] 
                                       if r.uncertainty_level in ['high', 'very_high']][:5]
            }
            
            return analysis_result
            
        except Exception as e:
            return {'error': f'Analysis failed for {program_name}: {str(e)}'}
    
    def mass_analyze(self, max_programs: int = 20, subdomains_per_program: int = 300) -> Dict:
        """Perform mass analysis of multiple programs."""
        print("🚀 NETRA Mass H1 Analysis Starting")
        print("=" * 50)
        
        # Discover programs
        programs = self.discover_programs()
        print(f"[+] Discovered {len(programs)} bug bounty programs")
        
        # Analyze top programs
        analyzed_count = 0
        total_subdomains = 0
        
        for program_name, subdomain_count in programs[:max_programs]:
            if analyzed_count >= max_programs:
                break
                
            print(f"\n[{analyzed_count + 1}/{max_programs}] {program_name} ({subdomain_count:,} subdomains)")
            
            result = self.analyze_program(program_name, subdomains_per_program)
            
            if 'error' in result:
                print(f"[-] {result['error']}")
                continue
            
            self.results[program_name] = result
            total_subdomains += result['total_subdomains']
            analyzed_count += 1
            
            # Add to knowledge base for training
            self._add_to_knowledge_base(program_name, result)
            
            print(f"[+] Analyzed {result['total_subdomains']} subdomains")
            print(f"    Critical/High Risk: {result['high_risk_count']}")
            print(f"    Top Category: {result['top_categories'][0] if result['top_categories'] else 'N/A'}")
            
            # Brief pause to avoid overwhelming the system
            time.sleep(0.5)
        
        # Generate comprehensive report
        intelligence_report = self._generate_intelligence_report(total_subdomains)
        
        print(f"\n✅ Mass analysis complete!")
        print(f"   Programs analyzed: {analyzed_count}")
        print(f"   Total subdomains: {total_subdomains:,}")
        
        return {
            'programs_analyzed': analyzed_count,
            'total_subdomains': total_subdomains,
            'results': self.results,
            'intelligence_report': intelligence_report
        }
    
    def _add_to_knowledge_base(self, program_name: str, result: Dict):
        """Add high-confidence results to NETRA's knowledge base."""
        try:
            conn = sqlite3.connect(self.ai.kb.db_path)
            cursor = conn.cursor()
            
            # Add top critical subdomains as training data
            added_count = 0
            for critical_result in result.get('critical_subdomains', [])[:5]:
                subdomain = critical_result.subdomain
                category = critical_result.primary_category
                confidence = critical_result.primary_confidence
                
                if confidence >= 0.8:  # Only high-confidence predictions
                    cursor.execute('''
                        INSERT OR IGNORE INTO subdomains (domain, category, confidence, source)
                        VALUES (?, ?, ?, ?)
                    ''', (subdomain, category, confidence, f'h1_{program_name}'))
                    
                    if cursor.rowcount > 0:
                        added_count += 1
            
            conn.commit()
            conn.close()
            
            if added_count > 0:
                print(f"[+] Added {added_count} high-confidence subdomains to knowledge base")
                
        except Exception as e:
            print(f"[-] Error adding to knowledge base: {e}")
    
    def _generate_intelligence_report(self, total_subdomains: int) -> Dict:
        """Generate comprehensive intelligence report."""
        
        # Aggregate statistics
        total_critical = sum(r['stats']['risk_distribution']['CRITICAL'] for r in self.results.values())
        total_high = sum(r['stats']['risk_distribution']['HIGH'] for r in self.results.values())
        
        # Category aggregation
        category_totals = {}
        for result in self.results.values():
            for category, count in result['stats']['category_distribution'].items():
                category_totals[category] = category_totals.get(category, 0) + count
        
        # Top risky programs
        risky_programs = sorted(
            [(name, r['high_risk_count'], r['total_subdomains']) for name, r in self.results.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        # Interesting patterns
        patterns = self._detect_patterns()
        
        return {
            'summary': {
                'total_programs': len(self.results),
                'total_subdomains_analyzed': total_subdomains,
                'critical_risk_targets': total_critical,
                'high_risk_targets': total_high,
                'risk_percentage': (total_critical + total_high) / total_subdomains * 100
            },
            'top_categories_global': sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:10],
            'riskiest_programs': risky_programs,
            'intelligence_patterns': patterns
        }
    
    def _detect_patterns(self) -> Dict:
        """Detect interesting patterns across programs."""
        patterns = {
            'common_admin_patterns': [],
            'dev_staging_prevalence': 0,
            'api_endpoint_trends': [],
            'security_service_usage': 0
        }
        
        dev_keywords = ['dev', 'staging', 'test', 'qa', 'sandbox', 'preprod']
        admin_keywords = ['admin', 'dashboard', 'console', 'portal', 'manage']
        
        for program_name, result in self.results.items():
            # Check for dev/staging prevalence
            dev_count = sum(
                count for category, count in result['stats']['category_distribution'].items()
                if any(keyword in category.lower() for keyword in dev_keywords)
            )
            if dev_count > 0:
                patterns['dev_staging_prevalence'] += 1
            
            # Check admin interfaces
            admin_count = result['stats']['category_distribution'].get('Administrative / Management Interfaces', 0)
            if admin_count > 0:
                patterns['common_admin_patterns'].append((program_name, admin_count))
        
        # Sort admin patterns
        patterns['common_admin_patterns'].sort(key=lambda x: x[1], reverse=True)
        patterns['common_admin_patterns'] = patterns['common_admin_patterns'][:10]
        
        return patterns
    
    def save_report(self, output_file: str = "h1_intelligence_report.json"):
        """Save comprehensive intelligence report."""
        if not self.results:
            print("[-] No analysis results to save")
            return
        
        # Create detailed report
        report = {
            'metadata': {
                'generated_at': time.time(),
                'analysis_tool': 'NETRA Mass H1 Analyzer',
                'version': '2.0.0',
                'total_programs': len(self.results)
            },
            'executive_summary': self._generate_intelligence_report(
                sum(r['total_subdomains'] for r in self.results.values())
            ),
            'program_details': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"[+] Intelligence report saved to: {output_file}")
        
        # Also create a summary CSV
        csv_file = output_file.replace('.json', '_summary.csv')
        with open(csv_file, 'w') as f:
            f.write("Program,Total Subdomains,Critical Risk,High Risk,Top Category\n")
            for name, result in self.results.items():
                critical = result['stats']['risk_distribution']['CRITICAL']
                high = result['stats']['risk_distribution']['HIGH']
                top_cat = result['top_categories'][0][0] if result['top_categories'] else 'Unknown'
                f.write(f"{name},{result['total_subdomains']},{critical},{high},{top_cat}\n")
        
        print(f"[+] Summary CSV saved to: {csv_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NETRA Mass H1 Subdomain Analyzer")
    parser.add_argument('--h1-path', default=r"C:\Users\abhij\bb\H1", 
                       help='Path to H1 bug bounty data')
    parser.add_argument('--max-programs', type=int, default=15,
                       help='Maximum programs to analyze')
    parser.add_argument('--subdomains-per-program', type=int, default=400,
                       help='Max subdomains per program')
    parser.add_argument('--output', default='h1_intelligence_report.json',
                       help='Output report file')
    
    args = parser.parse_args()
    
    analyzer = MassH1Analyzer(args.h1_path)
    
    # Perform mass analysis
    results = analyzer.mass_analyze(args.max_programs, args.subdomains_per_program)
    
    # Save comprehensive report
    analyzer.save_report(args.output)
    
    print(f"\n🎉 Analysis Complete!")
    print(f"   Intelligence gathered from {results['programs_analyzed']} programs")
    print(f"   {results['total_subdomains']:,} subdomains analyzed")
    print(f"   Report saved to: {args.output}")


if __name__ == "__main__":
    main()
