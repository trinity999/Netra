#!/usr/bin/env python3
"""
NETRA Ultimate Bug Bounty Analyzer
==================================

Analyzes subdomain data from ALL bug bounty platforms to build
the most comprehensive subdomain intelligence database possible.

Supports: HackerOne (H1), Bugcrowd (BC), Intigriti (IG), Microsoft (MS), Synack (SH)
"""

import os
import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
from enhanced_classifier import EnhancedSubdomainClassifier
from subdomain_ai_enhanced import SubdomainAIEnhanced

class UltimateBugBountyAnalyzer:
    """Analyzes ALL bug bounty platforms for maximum intelligence gathering."""
    
    def __init__(self, bb_base_path: str):
        self.bb_base_path = Path(bb_base_path)
        self.classifier = EnhancedSubdomainClassifier()
        self.ai = SubdomainAIEnhanced()
        self.results = {}
        self.global_stats = defaultdict(int)
        
        # Platform mapping
        self.platforms = {
            'H1': 'HackerOne',
            'BC': 'Bugcrowd', 
            'IG': 'Intigriti',
            'MS': 'Microsoft',
            'SH': 'Synack'
        }
        
        # Intelligence tracking
        self.intelligence_db = {}
        self.pattern_frequency = Counter()
        self.company_intelligence = {}
    
    def discover_all_programs(self) -> Dict[str, List[Tuple[str, int]]]:
        """Discover programs across ALL bug bounty platforms."""
        all_platforms = {}
        
        for platform_dir in self.bb_base_path.iterdir():
            if not platform_dir.is_dir() or platform_dir.name not in self.platforms:
                continue
            
            platform_name = self.platforms[platform_dir.name]
            print(f"🔍 Discovering {platform_name} programs...")
            
            programs = []
            program_count = 0
            
            for program_dir in platform_dir.iterdir():
                if not program_dir.is_dir():
                    continue
                
                assets_file = program_dir / "assets.txt"
                if assets_file.exists():
                    try:
                        # Count subdomains
                        with open(assets_file, 'r', encoding='utf-8', errors='ignore') as f:
                            subdomain_count = sum(1 for line in f if line.strip() and not line.startswith('#'))
                        
                        if subdomain_count > 0:
                            programs.append((program_dir.name, subdomain_count))
                            program_count += 1
                            
                    except Exception as e:
                        print(f"[-] Error reading {assets_file}: {e}")
                        continue
            
            # Sort by subdomain count
            programs.sort(key=lambda x: x[1], reverse=True)
            all_platforms[platform_dir.name] = programs
            
            total_subdomains = sum(count for _, count in programs)
            print(f"[+] {platform_name}: {program_count} programs, {total_subdomains:,} total subdomains")
        
        return all_platforms
    
    def analyze_program(self, platform: str, program_name: str, limit: int = 300) -> Dict:
        """Analyze a single program with enhanced intelligence extraction."""
        platform_dir = self.bb_base_path / platform / program_name
        assets_file = platform_dir / "assets.txt"
        
        if not assets_file.exists():
            return {'error': f'Assets file not found for {platform}/{program_name}'}
        
        # Load subdomains with robust encoding handling
        subdomains = self._load_subdomains(assets_file, limit)
        if not subdomains:
            return {'error': f'Could not read subdomains for {platform}/{program_name}'}
        
        print(f"[*] Analyzing {len(subdomains)} subdomains...")
        
        try:
            # Perform enhanced batch analysis
            batch_result = self.classifier.batch_classify_with_stats(subdomains)
            
            # Extract intelligence patterns
            intelligence = self._extract_intelligence_patterns(subdomains, batch_result)
            
            # Build comprehensive result
            result = {
                'platform': self.platforms.get(platform, platform),
                'platform_code': platform,
                'program_name': program_name,
                'total_subdomains': len(subdomains),
                'analysis_timestamp': time.time(),
                'stats': batch_result['stats'],
                'intelligence': intelligence,
                'top_categories': sorted(batch_result['stats']['category_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5],
                'high_risk_count': batch_result['stats']['risk_distribution']['CRITICAL'] + 
                                 batch_result['stats']['risk_distribution']['HIGH'],
                'critical_subdomains': [r for r in batch_result['results'] 
                                      if self.classifier._get_risk_level(r.primary_category) == 'CRITICAL'][:10],
                'unique_patterns': intelligence['unique_patterns'][:10]
            }
            
            # Add to global intelligence
            self._update_global_intelligence(platform, program_name, result, batch_result['results'])
            
            return result
            
        except Exception as e:
            return {'error': f'Analysis failed for {platform}/{program_name}: {str(e)}'}
    
    def _load_subdomains(self, assets_file: Path, limit: int) -> List[str]:
        """Load subdomains with robust encoding support."""
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
        
        for encoding in encodings_to_try:
            try:
                with open(assets_file, 'r', encoding=encoding, errors='ignore') as f:
                    subdomains = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '.' in line:
                            # Clean up wildcard patterns
                            if line.startswith('*.'):
                                line = line[2:]
                            subdomains.append(line)
                        
                        if len(subdomains) >= limit:
                            break
                    
                return subdomains
                
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        return []
    
    def _extract_intelligence_patterns(self, subdomains: List[str], batch_result: Dict) -> Dict:
        """Extract advanced intelligence patterns from subdomain analysis."""
        patterns = {
            'unique_patterns': [],
            'admin_interfaces': [],
            'dev_staging_count': 0,
            'api_endpoints': [],
            'geographic_indicators': [],
            'technology_stack': [],
            'service_patterns': {},
            'domain_depth_analysis': {},
            'naming_conventions': []
        }
        
        # Common technology and service indicators
        tech_indicators = {
            'aws': 'Amazon Web Services',
            'azure': 'Microsoft Azure', 
            'gcp': 'Google Cloud Platform',
            'cloudflare': 'Cloudflare',
            'fastly': 'Fastly CDN',
            'akamai': 'Akamai',
            'heroku': 'Heroku',
            'docker': 'Docker',
            'k8s': 'Kubernetes',
            'jenkins': 'Jenkins CI/CD'
        }
        
        geographic_codes = ['us', 'eu', 'asia', 'uk', 'de', 'fr', 'jp', 'cn', 'au', 'ca', 'br', 'in']
        
        for i, subdomain in enumerate(subdomains):
            result = batch_result['results'][i] if i < len(batch_result['results']) else None
            
            # Analyze subdomain structure
            parts = subdomain.lower().split('.')
            subdomain_part = parts[0] if parts else ''
            
            # Technology stack detection
            for tech, name in tech_indicators.items():
                if tech in subdomain.lower():
                    patterns['technology_stack'].append((subdomain, name))
            
            # Geographic indicators
            for geo in geographic_codes:
                if geo in subdomain_part or f'-{geo}' in subdomain_part:
                    patterns['geographic_indicators'].append((subdomain, geo.upper()))
            
            # Admin interfaces (high-value targets)
            if result and result.primary_category == 'Administrative / Management Interfaces':
                patterns['admin_interfaces'].append(subdomain)
            
            # API endpoints
            if 'api' in subdomain.lower():
                patterns['api_endpoints'].append(subdomain)
            
            # Development/staging
            dev_keywords = ['dev', 'staging', 'test', 'qa', 'sandbox', 'preprod', 'beta']
            if any(keyword in subdomain.lower() for keyword in dev_keywords):
                patterns['dev_staging_count'] += 1
            
            # Domain depth analysis
            depth = len(parts)
            patterns['domain_depth_analysis'][depth] = patterns['domain_depth_analysis'].get(depth, 0) + 1
            
            # Extract unique naming patterns
            if len(subdomain_part) > 3 and not subdomain_part.isdigit():
                # Look for interesting patterns
                if '-' in subdomain_part:
                    pattern_parts = subdomain_part.split('-')
                    if len(pattern_parts) >= 2:
                        patterns['unique_patterns'].append(f"{pattern_parts[0]}-*-{pattern_parts[-1]}")
        
        # Clean up and deduplicate
        patterns['unique_patterns'] = list(set(patterns['unique_patterns']))
        patterns['technology_stack'] = list(set(patterns['technology_stack']))
        patterns['geographic_indicators'] = list(set(patterns['geographic_indicators']))
        patterns['admin_interfaces'] = patterns['admin_interfaces'][:20]  # Top 20
        patterns['api_endpoints'] = patterns['api_endpoints'][:20]  # Top 20
        
        return patterns
    
    def _update_global_intelligence(self, platform: str, program_name: str, result: Dict, detailed_results: List):
        """Update global intelligence database."""
        # Update global stats
        self.global_stats['total_programs'] += 1
        self.global_stats['total_subdomains'] += result['total_subdomains']
        self.global_stats[f'{platform}_programs'] += 1
        
        # Track pattern frequency
        for pattern in result['intelligence']['unique_patterns']:
            self.pattern_frequency[pattern] += 1
        
        # Company intelligence (for well-known companies)
        key_companies = [
            'google', 'microsoft', 'amazon', 'apple', 'facebook', 'twitter', 'github',
            'slack', 'spotify', 'uber', 'airbnb', 'netflix', 'adobe', 'salesforce'
        ]
        
        program_lower = program_name.lower()
        for company in key_companies:
            if company in program_lower:
                if company not in self.company_intelligence:
                    self.company_intelligence[company] = {
                        'programs': [],
                        'total_subdomains': 0,
                        'platforms': set(),
                        'risk_profile': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
                    }
                
                self.company_intelligence[company]['programs'].append(f"{platform}/{program_name}")
                self.company_intelligence[company]['total_subdomains'] += result['total_subdomains']
                self.company_intelligence[company]['platforms'].add(platform)
                
                # Update risk profile
                for risk_level, count in result['stats']['risk_distribution'].items():
                    if risk_level.lower() in self.company_intelligence[company]['risk_profile']:
                        self.company_intelligence[company]['risk_profile'][risk_level.lower()] += count
        
        # Add high-confidence results to knowledge base
        self._add_to_knowledge_base(platform, program_name, result)
    
    def _add_to_knowledge_base(self, platform: str, program_name: str, result: Dict):
        """Add high-confidence results to NETRA's knowledge base."""
        try:
            conn = sqlite3.connect(self.ai.kb.db_path)
            cursor = conn.cursor()
            
            added_count = 0
            source_tag = f"{platform.lower()}_{program_name}"
            
            # Add critical subdomains with high confidence
            for critical_result in result.get('critical_subdomains', [])[:10]:
                subdomain = critical_result.subdomain
                category = critical_result.primary_category
                confidence = critical_result.primary_confidence
                
                if confidence >= 0.75:  # High confidence threshold for real-world data
                    cursor.execute('''
                        INSERT OR IGNORE INTO subdomains (domain, category, confidence, source)
                        VALUES (?, ?, ?, ?)
                    ''', (subdomain, category, confidence, source_tag))
                    
                    if cursor.rowcount > 0:
                        added_count += 1
            
            conn.commit()
            conn.close()
            
            if added_count > 0:
                print(f"[+] Added {added_count} high-confidence patterns to knowledge base")
                
        except Exception as e:
            print(f"[-] Error updating knowledge base: {e}")
    
    def ultimate_analysis(self, max_programs_per_platform: int = 30, subdomains_per_program: int = 400) -> Dict:
        """Perform ultimate analysis across ALL bug bounty platforms."""
        print("🚀 NETRA ULTIMATE BUG BOUNTY ANALYSIS")
        print("=" * 60)
        print("Analyzing ALL bug bounty platforms for maximum intelligence...")
        
        # Discover all programs
        all_platforms = self.discover_all_programs()
        
        total_programs_found = sum(len(programs) for programs in all_platforms.values())
        print(f"\n[+] Total programs discovered: {total_programs_found}")
        print(f"[+] Platforms: {', '.join(all_platforms.keys())}")
        
        # Analyze programs from each platform
        analyzed_count = 0
        total_subdomains = 0
        platform_results = {}
        
        for platform_code, programs in all_platforms.items():
            if not programs:
                continue
                
            platform_name = self.platforms[platform_code]
            print(f"\n🎯 Analyzing {platform_name} ({len(programs)} programs)...")
            
            platform_analyzed = 0
            platform_subdomains = 0
            
            for program_name, subdomain_count in programs[:max_programs_per_platform]:
                if platform_analyzed >= max_programs_per_platform:
                    break
                
                print(f"\n[{analyzed_count + 1}] {platform_code}/{program_name} ({subdomain_count:,} subdomains)")
                
                result = self.analyze_program(platform_code, program_name, subdomains_per_program)
                
                if 'error' in result:
                    print(f"[-] {result['error']}")
                    continue
                
                # Store result
                if platform_code not in platform_results:
                    platform_results[platform_code] = []
                platform_results[platform_code].append(result)
                
                # Update counters
                analyzed_count += 1
                platform_analyzed += 1
                total_subdomains += result['total_subdomains']
                platform_subdomains += result['total_subdomains']
                
                # Progress info
                print(f"[+] Analyzed: {result['total_subdomains']} subdomains")
                print(f"    Critical/High Risk: {result['high_risk_count']}")
                print(f"    Top Category: {result['top_categories'][0] if result['top_categories'] else 'N/A'}")
                print(f"    Unique Patterns: {len(result['intelligence']['unique_patterns'])}")
                
                # Brief pause
                time.sleep(0.3)
            
            print(f"\n✅ {platform_name} Complete: {platform_analyzed} programs, {platform_subdomains:,} subdomains")
        
        self.results = platform_results
        
        # Generate ultimate intelligence report
        ultimate_report = self._generate_ultimate_report(total_subdomains, analyzed_count)
        
        print(f"\n🎉 ULTIMATE ANALYSIS COMPLETE!")
        print(f"   Platforms analyzed: {len(platform_results)}")
        print(f"   Total programs: {analyzed_count}")
        print(f"   Total subdomains: {total_subdomains:,}")
        
        return {
            'platforms_analyzed': len(platform_results),
            'programs_analyzed': analyzed_count,
            'total_subdomains': total_subdomains,
            'platform_results': platform_results,
            'ultimate_intelligence_report': ultimate_report,
            'global_stats': dict(self.global_stats),
            'company_intelligence': {k: {**v, 'platforms': list(v['platforms'])} 
                                   for k, v in self.company_intelligence.items()}
        }
    
    def _generate_ultimate_report(self, total_subdomains: int, total_programs: int) -> Dict:
        """Generate the ultimate intelligence report."""
        # Aggregate statistics across all platforms
        total_critical = sum(
            sum(result['stats']['risk_distribution']['CRITICAL'] for result in platform_results)
            for platform_results in self.results.values()
        )
        
        total_high = sum(
            sum(result['stats']['risk_distribution']['HIGH'] for result in platform_results)  
            for platform_results in self.results.values()
        )
        
        # Global category distribution
        global_categories = Counter()
        for platform_results in self.results.values():
            for result in platform_results:
                for category, count in result['stats']['category_distribution'].items():
                    global_categories[category] += count
        
        # Platform comparison
        platform_comparison = {}
        for platform_code, platform_results in self.results.items():
            platform_name = self.platforms[platform_code]
            platform_subdomains = sum(r['total_subdomains'] for r in platform_results)
            platform_critical = sum(r['stats']['risk_distribution']['CRITICAL'] for r in platform_results)
            
            platform_comparison[platform_name] = {
                'programs': len(platform_results),
                'subdomains': platform_subdomains,
                'critical_risk': platform_critical,
                'risk_percentage': (platform_critical / platform_subdomains * 100) if platform_subdomains > 0 else 0
            }
        
        # Most common patterns across all platforms
        top_patterns = self.pattern_frequency.most_common(20)
        
        # Intelligence insights
        insights = self._generate_intelligence_insights()
        
        return {
            'summary': {
                'platforms_analyzed': len(self.results),
                'total_programs': total_programs,
                'total_subdomains_analyzed': total_subdomains,
                'critical_risk_targets': total_critical,
                'high_risk_targets': total_high,
                'overall_risk_percentage': ((total_critical + total_high) / total_subdomains * 100) if total_subdomains > 0 else 0
            },
            'platform_comparison': platform_comparison,
            'global_category_distribution': global_categories.most_common(15),
            'most_common_patterns': top_patterns,
            'company_intelligence': self.company_intelligence,
            'intelligence_insights': insights
        }
    
    def _generate_intelligence_insights(self) -> Dict:
        """Generate advanced intelligence insights."""
        insights = {
            'cross_platform_patterns': [],
            'high_value_targets': [],
            'technology_trends': Counter(),
            'geographic_distribution': Counter(),
            'security_observations': []
        }
        
        # Analyze patterns across platforms
        all_patterns = []
        all_tech_indicators = []
        all_geo_indicators = []
        
        for platform_results in self.results.values():
            for result in platform_results:
                intelligence = result.get('intelligence', {})
                all_patterns.extend(intelligence.get('unique_patterns', []))
                all_tech_indicators.extend([tech for _, tech in intelligence.get('technology_stack', [])])
                all_geo_indicators.extend([geo for _, geo in intelligence.get('geographic_indicators', [])])
        
        insights['technology_trends'] = Counter(all_tech_indicators)
        insights['geographic_distribution'] = Counter(all_geo_indicators)
        
        # Cross-platform pattern analysis
        pattern_counter = Counter(all_patterns)
        insights['cross_platform_patterns'] = [
            {'pattern': pattern, 'frequency': freq, 'platforms': self._count_pattern_platforms(pattern)}
            for pattern, freq in pattern_counter.most_common(10)
            if freq >= 3  # Appears in at least 3 programs
        ]
        
        # Security observations
        insights['security_observations'] = [
            f"Critical risk subdomains represent {(insights['technology_trends']['Amazon Web Services'] / max(sum(insights['technology_trends'].values()), 1)) * 100:.1f}% of AWS infrastructure",
            f"Development/staging environments found in {len([r for platform in self.results.values() for r in platform if r['intelligence']['dev_staging_count'] > 0])} programs",
            f"Geographic distribution shows strongest presence in: {insights['geographic_distribution'].most_common(3) if insights['geographic_distribution'] else 'No data'}"
        ]
        
        return insights
    
    def _count_pattern_platforms(self, pattern: str) -> int:
        """Count how many platforms a pattern appears in."""
        platforms_with_pattern = set()
        for platform_code, platform_results in self.results.items():
            for result in platform_results:
                if pattern in result.get('intelligence', {}).get('unique_patterns', []):
                    platforms_with_pattern.add(platform_code)
        return len(platforms_with_pattern)
    
    def save_ultimate_report(self, output_file: str = "ultimate_bb_intelligence.json"):
        """Save the ultimate bug bounty intelligence report."""
        if not self.results:
            print("[-] No analysis results to save")
            return
        
        # Create comprehensive report
        report = {
            'metadata': {
                'generated_at': time.time(),
                'analysis_tool': 'NETRA Ultimate Bug Bounty Analyzer',
                'version': '3.0.0',
                'platforms_analyzed': list(self.platforms.values())
            },
            'executive_summary': self._generate_ultimate_report(
                self.global_stats['total_subdomains'],
                self.global_stats['total_programs']
            ),
            'detailed_results': self.results,
            'global_statistics': dict(self.global_stats),
            'company_intelligence': {k: {**v, 'platforms': list(v['platforms'])} 
                                   for k, v in self.company_intelligence.items()}
        }
        
        # Save main report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"[+] Ultimate intelligence report saved to: {output_file}")
        
        # Create summary CSV
        csv_file = output_file.replace('.json', '_summary.csv')
        with open(csv_file, 'w') as f:
            f.write("Platform,Program,Total Subdomains,Critical Risk,High Risk,Top Category,Unique Patterns\n")
            for platform_code, platform_results in self.results.items():
                platform_name = self.platforms[platform_code]
                for result in platform_results:
                    critical = result['stats']['risk_distribution']['CRITICAL']
                    high = result['stats']['risk_distribution']['HIGH']
                    top_cat = result['top_categories'][0][0] if result['top_categories'] else 'Unknown'
                    patterns = len(result['intelligence']['unique_patterns'])
                    f.write(f"{platform_name},{result['program_name']},{result['total_subdomains']},{critical},{high},{top_cat},{patterns}\n")
        
        print(f"[+] Summary CSV saved to: {csv_file}")
        
        # Create intelligence insights report
        insights_file = output_file.replace('.json', '_insights.txt')
        with open(insights_file, 'w') as f:
            report_data = report['executive_summary']
            f.write("NETRA ULTIMATE BUG BOUNTY INTELLIGENCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            summary = report_data['summary']
            f.write(f"EXECUTIVE SUMMARY:\n")
            f.write(f"- Platforms Analyzed: {summary['platforms_analyzed']}\n")
            f.write(f"- Programs Analyzed: {summary['total_programs']}\n") 
            f.write(f"- Subdomains Analyzed: {summary['total_subdomains_analyzed']:,}\n")
            f.write(f"- Critical Risk Targets: {summary['critical_risk_targets']:,}\n")
            f.write(f"- Overall Risk Percentage: {summary['overall_risk_percentage']:.1f}%\n\n")
            
            # Platform comparison
            f.write("PLATFORM COMPARISON:\n")
            for platform, data in report_data['platform_comparison'].items():
                f.write(f"- {platform}: {data['programs']} programs, {data['subdomains']:,} subdomains, {data['risk_percentage']:.1f}% critical risk\n")
            
            f.write("\nTOP GLOBAL CATEGORIES:\n")
            for category, count in report_data['global_category_distribution'][:10]:
                f.write(f"- {category}: {count:,} instances\n")
            
            f.write("\nCOMPANY INTELLIGENCE:\n")
            for company, intel in report_data.get('company_intelligence', {}).items():
                f.write(f"- {company.title()}: {intel['total_subdomains']:,} subdomains across {len(intel['platforms'])} platforms\n")
        
        print(f"[+] Intelligence insights saved to: {insights_file}")


def main():
    """Main function for ultimate bug bounty analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NETRA Ultimate Bug Bounty Intelligence Analyzer")
    parser.add_argument('--bb-path', default=r"C:\Users\abhij\bb", 
                       help='Path to bug bounty data directory')
    parser.add_argument('--max-programs-per-platform', type=int, default=25,
                       help='Maximum programs per platform to analyze')
    parser.add_argument('--subdomains-per-program', type=int, default=500,
                       help='Max subdomains per program to analyze')
    parser.add_argument('--output', default='ultimate_bb_intelligence.json',
                       help='Output report file')
    
    args = parser.parse_args()
    
    # Initialize ultimate analyzer
    analyzer = UltimateBugBountyAnalyzer(args.bb_path)
    
    # Perform ultimate analysis
    print(f"🎯 Starting ultimate analysis of ALL bug bounty platforms...")
    results = analyzer.ultimate_analysis(
        args.max_programs_per_platform, 
        args.subdomains_per_program
    )
    
    # Save comprehensive reports
    analyzer.save_ultimate_report(args.output)
    
    print(f"\n🎉 ULTIMATE ANALYSIS COMPLETE!")
    print(f"   Platforms: {results['platforms_analyzed']}")
    print(f"   Programs: {results['programs_analyzed']}")
    print(f"   Subdomains: {results['total_subdomains']:,}")
    print(f"   Report: {args.output}")
    
    # Display top insights
    summary = results['ultimate_intelligence_report']['summary']
    print(f"\n📊 KEY INSIGHTS:")
    print(f"   Critical Risk: {summary['critical_risk_targets']:,} ({summary['overall_risk_percentage']:.1f}%)")
    print(f"   Cross-Platform Intelligence: Maximum achieved!")


if __name__ == "__main__":
    main()
