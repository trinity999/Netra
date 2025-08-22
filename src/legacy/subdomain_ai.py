#!/usr/bin/env python3
"""
Subdomain AI - Security Research Tool
====================================

A comprehensive tool for collecting and analyzing subdomains with AI-powered risk assessment.

Features:
- Multi-tool subdomain collection (Subfinder, Amass, Assetfinder)
- AI-powered security analysis and categorization
- Modular architecture for easy extension
- Comprehensive error handling and reporting

Author: Security Research Team
Version: 1.0.0
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import tempfile
import time

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class SubdomainCollector:
    """Handles subdomain collection using various tools."""
    
    def __init__(self):
        self.tools = {
            'subfinder': self._run_subfinder,
            'amass': self._run_amass,
            'assetfinder': self._run_assetfinder
        }
        self.available_tools = self._check_tool_availability()
    
    def _check_tool_availability(self) -> Dict[str, bool]:
        """Check which subdomain enumeration tools are available."""
        availability = {}
        
        for tool in self.tools.keys():
            try:
                # Check if tool is available in PATH
                result = subprocess.run([tool, '--help'], 
                                      capture_output=True, 
                                      timeout=10)
                availability[tool] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                availability[tool] = False
                
        return availability
    
    def _run_subfinder(self, domain: str) -> Set[str]:
        """Run Subfinder to collect subdomains."""
        subdomains = set()
        try:
            print(f"[*] Running Subfinder for {domain}...")
            cmd = ['subfinder', '-d', domain, '-silent']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                subdomains = set(line.strip() for line in result.stdout.splitlines() if line.strip())
                print(f"[+] Subfinder found {len(subdomains)} subdomains")
            else:
                print(f"[-] Subfinder failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("[-] Subfinder timeout - continuing with other tools")
        except Exception as e:
            print(f"[-] Subfinder error: {e}")
            
        return subdomains
    
    def _run_amass(self, domain: str) -> Set[str]:
        """Run Amass (passive mode) to collect subdomains."""
        subdomains = set()
        try:
            print(f"[*] Running Amass (passive) for {domain}...")
            cmd = ['amass', 'enum', '-passive', '-d', domain]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                subdomains = set(line.strip() for line in result.stdout.splitlines() if line.strip())
                print(f"[+] Amass found {len(subdomains)} subdomains")
            else:
                print(f"[-] Amass failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("[-] Amass timeout - continuing with other tools")
        except Exception as e:
            print(f"[-] Amass error: {e}")
            
        return subdomains
    
    def _run_assetfinder(self, domain: str) -> Set[str]:
        """Run Assetfinder to collect subdomains."""
        subdomains = set()
        try:
            print(f"[*] Running Assetfinder for {domain}...")
            cmd = ['assetfinder', '--subs-only', domain]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                subdomains = set(line.strip() for line in result.stdout.splitlines() if line.strip())
                print(f"[+] Assetfinder found {len(subdomains)} subdomains")
            else:
                print(f"[-] Assetfinder failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("[-] Assetfinder timeout - continuing with other tools")
        except Exception as e:
            print(f"[-] Assetfinder error: {e}")
            
        return subdomains
    
    def collect_subdomains(self, domain: str, output_file: str = "all_subs.txt") -> int:
        """
        Collect subdomains using all available tools.
        
        Args:
            domain: Target domain to enumerate
            output_file: Output file path for collected subdomains
            
        Returns:
            Number of unique subdomains found
        """
        print(f"\n[*] Starting subdomain collection for: {domain}")
        print(f"[*] Available tools: {[tool for tool, available in self.available_tools.items() if available]}")
        
        if not any(self.available_tools.values()):
            print("[-] No subdomain enumeration tools are available!")
            print("[-] Please install: subfinder, amass, and/or assetfinder")
            return 0
        
        all_subdomains = set()
        
        # Run each available tool
        for tool_name, tool_func in self.tools.items():
            if self.available_tools.get(tool_name, False):
                try:
                    subs = tool_func(domain)
                    all_subdomains.update(subs)
                except Exception as e:
                    print(f"[-] Error running {tool_name}: {e}")
            else:
                print(f"[-] Skipping {tool_name} (not available)")
        
        # Save results
        unique_count = len(all_subdomains)
        if unique_count > 0:
            with open(output_file, 'w') as f:
                for subdomain in sorted(all_subdomains):
                    f.write(f"{subdomain}\n")
            
            print(f"\n[+] Collection complete!")
            print(f"[+] Found {unique_count} unique subdomains")
            print(f"[+] Saved to: {output_file}")
        else:
            print("\n[-] No subdomains found!")
            
        return unique_count


class SubdomainAnalyzer:
    """Handles AI-powered subdomain analysis and risk assessment."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if OPENAI_AVAILABLE and self.api_key:
            openai.api_key = self.api_key
        
        # Predefined categories for classification
        self.categories = [
            "Administrative / Management Interfaces",
            "APIs",
            "Staging / Development / Testing", 
            "Authentication / Identity",
            "Payment / Transactional",
            "CDN / Storage / Assets",
            "Database / Data Services",
            "Internal Tools / Infrastructure",
            "Marketing / Content / CMS",
            "Mobile / Partner / Integration",
            "Monitoring / Logging",
            "Security Services"
        ]
    
    def _load_subdomains(self, file_path: str) -> List[str]:
        """Load subdomains from file."""
        try:
            with open(file_path, 'r') as f:
                subdomains = [line.strip() for line in f if line.strip()]
            return subdomains
        except FileNotFoundError:
            print(f"[-] Subdomain file not found: {file_path}")
            return []
        except Exception as e:
            print(f"[-] Error reading subdomain file: {e}")
            return []
    
    def _create_analysis_prompt(self, subdomains: List[str]) -> str:
        """Create the prompt for AI analysis."""
        prompt = f"""You are a security researcher. Analyze the following list of subdomains and classify each into attack surface categories.

### Instructions:
1. For each subdomain:
   - Identify the most likely category (from the list below).
   - Suggest possible risks/attack vectors based on security research knowledge.
   - Be precise and concise — only list realistic risks.
   - If a subdomain fits multiple categories, tag both.

2. Categories:
{chr(10).join(f'- {cat}' for cat in self.categories)}

3. Output ONLY valid JSON in this format:
[
  {{
    "subdomain": "example.subdomain.com",
    "categories": ["Category Name"],
    "possible_risks": ["Risk 1", "Risk 2", "Risk 3"]
  }}
]

### Subdomains to analyze:
{chr(10).join(subdomains[:50])}  

Respond with ONLY the JSON array, no other text."""
        
        return prompt
    
    def _call_openai_api(self, prompt: str) -> Optional[str]:
        """Make API call to OpenAI."""
        if not OPENAI_AVAILABLE:
            print("[-] OpenAI library not available. Install with: pip install openai")
            return None
            
        if not self.api_key:
            print("[-] OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            return None
        
        try:
            print("[*] Sending subdomains to AI for analysis...")
            
            response = openai.ChatCompletion.create(
                model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper analysis
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a cybersecurity expert specializing in attack surface analysis."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=4000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[-] OpenAI API error: {e}")
            return None
    
    def _mock_analysis(self, subdomains: List[str]) -> List[Dict]:
        """Provide mock analysis when AI is not available."""
        print("[*] Using mock analysis (AI not available)")
        
        mock_results = []
        
        for subdomain in subdomains:
            sub_lower = subdomain.lower()
            
            # Simple heuristic-based categorization
            categories = []
            risks = []
            
            if any(keyword in sub_lower for keyword in ['admin', 'dashboard', 'console', 'portal', 'cpanel', 'manage']):
                categories.append("Administrative / Management Interfaces")
                risks = ["Weak authentication", "Exposed management interface", "Privilege escalation"]
                
            elif any(keyword in sub_lower for keyword in ['api', 'graphql', 'rest']):
                categories.append("APIs")
                risks = ["Broken authentication", "Exposed endpoints", "Rate limiting bypass", "Data exposure"]
                
            elif any(keyword in sub_lower for keyword in ['staging', 'dev', 'test', 'qa', 'sandbox', 'preprod']):
                categories.append("Staging / Development / Testing")
                risks = ["Debug info leaks", "Weak authentication", "Outdated versions", "Test data exposure"]
                
            elif any(keyword in sub_lower for keyword in ['auth', 'login', 'sso', 'accounts', 'idp', 'oauth']):
                categories.append("Authentication / Identity")
                risks = ["Authentication bypass", "Session hijacking", "OAuth misconfigurations"]
                
            elif any(keyword in sub_lower for keyword in ['payments', 'billing', 'checkout', 'invoice']):
                categories.append("Payment / Transactional")
                risks = ["Payment bypass", "Financial data exposure", "Transaction manipulation"]
                
            elif any(keyword in sub_lower for keyword in ['cdn', 'static', 'media', 'uploads', 'files', 'assets']):
                categories.append("CDN / Storage / Assets")
                risks = ["File upload vulnerabilities", "Directory traversal", "Sensitive file exposure"]
                
            elif any(keyword in sub_lower for keyword in ['db', 'sql', 'mysql', 'postgres', 'mongo', 'redis']):
                categories.append("Database / Data Services")
                risks = ["Database exposure", "Injection vulnerabilities", "Data breach potential"]
                
            elif any(keyword in sub_lower for keyword in ['jira', 'jenkins', 'git', 'grafana', 'kibana', 'vpn', 'ci']):
                categories.append("Internal Tools / Infrastructure")
                risks = ["Internal system access", "Source code exposure", "Infrastructure compromise"]
                
            else:
                categories.append("Marketing / Content / CMS")
                risks = ["Information disclosure", "CMS vulnerabilities", "SEO poisoning"]
            
            mock_results.append({
                "subdomain": subdomain,
                "categories": categories,
                "possible_risks": risks
            })
        
        return mock_results
    
    def analyze_subdomains(self, file_path: str, use_ai: bool = True) -> Tuple[bool, str, str]:
        """
        Analyze subdomains and generate reports.
        
        Args:
            file_path: Path to subdomain list file
            use_ai: Whether to use AI analysis or mock analysis
            
        Returns:
            Tuple of (success, json_file_path, report_file_path)
        """
        print(f"\n[*] Starting subdomain analysis from: {file_path}")
        
        # Load subdomains
        subdomains = self._load_subdomains(file_path)
        if not subdomains:
            return False, "", ""
        
        print(f"[*] Loaded {len(subdomains)} subdomains for analysis")
        
        # Get analysis results
        if use_ai and OPENAI_AVAILABLE and self.api_key:
            # Process in batches if too many subdomains
            batch_size = 50
            all_results = []
            
            for i in range(0, len(subdomains), batch_size):
                batch = subdomains[i:i + batch_size]
                prompt = self._create_analysis_prompt(batch)
                response = self._call_openai_api(prompt)
                
                if response:
                    try:
                        # Clean response and parse JSON
                        response_clean = response.strip()
                        if response_clean.startswith('```json'):
                            response_clean = response_clean[7:]
                        if response_clean.endswith('```'):
                            response_clean = response_clean[:-3]
                        
                        batch_results = json.loads(response_clean.strip())
                        all_results.extend(batch_results)
                        
                    except json.JSONDecodeError as e:
                        print(f"[-] Failed to parse AI response as JSON: {e}")
                        print("[-] Falling back to mock analysis")
                        all_results.extend(self._mock_analysis(batch))
                
                # Rate limiting delay
                time.sleep(1)
                
            analysis_results = all_results
        else:
            # Use mock analysis
            analysis_results = self._mock_analysis(subdomains)
        
        # Save JSON results
        json_file = "analysis.json"
        try:
            with open(json_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            print(f"[+] JSON analysis saved to: {json_file}")
        except Exception as e:
            print(f"[-] Error saving JSON: {e}")
            return False, "", ""
        
        # Generate human-readable report
        report_file = "analysis_report.txt"
        try:
            self._generate_report(analysis_results, report_file)
            print(f"[+] Analysis report saved to: {report_file}")
        except Exception as e:
            print(f"[-] Error generating report: {e}")
            return False, json_file, ""
        
        # Print summary
        self._print_summary(analysis_results)
        
        return True, json_file, report_file
    
    def _generate_report(self, results: List[Dict], output_file: str):
        """Generate human-readable analysis report."""
        category_counts = defaultdict(int)
        category_subdomains = defaultdict(list)
        
        # Organize results by category
        for result in results:
            subdomain = result['subdomain']
            categories = result['categories']
            risks = result['possible_risks']
            
            for category in categories:
                category_counts[category] += 1
                category_subdomains[category].append({
                    'subdomain': subdomain,
                    'risks': risks
                })
        
        # Write report
        with open(output_file, 'w') as f:
            f.write("SUBDOMAIN SECURITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Subdomains Analyzed: {len(results)}\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Category summary
            f.write("CATEGORY SUMMARY\n")
            f.write("-" * 20 + "\n")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{category}: {count} subdomains\n")
            
            f.write("\n")
            
            # Detailed analysis by category
            for category in sorted(category_subdomains.keys()):
                f.write(f"\n{category.upper()}\n")
                f.write("=" * len(category) + "\n")
                
                for item in category_subdomains[category]:
                    f.write(f"\n• {item['subdomain']}\n")
                    for risk in item['risks']:
                        f.write(f"  - {risk}\n")
            
            # High-priority targets
            f.write("\n\nHIGH-PRIORITY TARGETS\n")
            f.write("=" * 25 + "\n")
            
            high_priority_categories = [
                "Administrative / Management Interfaces",
                "APIs",
                "Database / Data Services",
                "Authentication / Identity",
                "Payment / Transactional"
            ]
            
            for category in high_priority_categories:
                if category in category_subdomains:
                    f.write(f"\n{category}:\n")
                    for item in category_subdomains[category][:5]:  # Top 5 per category
                        f.write(f"  • {item['subdomain']}\n")
    
    def _print_summary(self, results: List[Dict]):
        """Print analysis summary to console."""
        category_counts = defaultdict(int)
        high_value_targets = []
        
        high_value_categories = [
            "Administrative / Management Interfaces",
            "APIs", 
            "Database / Data Services",
            "Authentication / Identity",
            "Payment / Transactional"
        ]
        
        for result in results:
            for category in result['categories']:
                category_counts[category] += 1
                if category in high_value_categories:
                    high_value_targets.append(result['subdomain'])
        
        print(f"\n[+] Analysis Summary:")
        print(f"[+] Total subdomains analyzed: {len(results)}")
        print(f"[+] Categories found:")
        
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {category}: {count}")
        
        if high_value_targets:
            print(f"\n[!] High-value targets identified: {len(set(high_value_targets))}")
            for target in sorted(set(high_value_targets))[:10]:  # Show top 10
                print(f"    - {target}")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Subdomain AI - Collect and analyze subdomains with AI-powered risk assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python subdomain_ai.py --collect example.com
  python subdomain_ai.py --analyze subdomains.txt
  python subdomain_ai.py --both example.com
  python subdomain_ai.py --collect example.com --output my_subs.txt
        """
    )
    
    parser.add_argument('--collect', metavar='DOMAIN', 
                       help='Collect subdomains for the specified domain')
    parser.add_argument('--analyze', metavar='FILE',
                       help='Analyze subdomains from the specified file')
    parser.add_argument('--both', metavar='DOMAIN',
                       help='Collect and then analyze subdomains for the domain')
    parser.add_argument('--output', metavar='FILE', default='all_subs.txt',
                       help='Output file for collected subdomains (default: all_subs.txt)')
    parser.add_argument('--no-ai', action='store_true',
                       help='Skip AI analysis and use heuristic-based analysis')
    parser.add_argument('--api-key', metavar='KEY',
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Check if at least one main action is specified
    if not any([args.collect, args.analyze, args.both]):
        parser.print_help()
        sys.exit(1)
    
    print("🔍 Subdomain AI - Security Research Tool")
    print("=" * 50)
    
    collector = SubdomainCollector()
    analyzer = SubdomainAnalyzer(api_key=args.api_key)
    
    # Collection mode
    if args.collect or args.both:
        domain = args.collect or args.both
        print(f"\n[*] Collection Mode: {domain}")
        
        subdomain_count = collector.collect_subdomains(domain, args.output)
        
        if subdomain_count == 0:
            print("[-] No subdomains collected. Exiting.")
            sys.exit(1)
    
    # Analysis mode
    if args.analyze or args.both:
        file_path = args.analyze or args.output
        print(f"\n[*] Analysis Mode: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"[-] File not found: {file_path}")
            sys.exit(1)
        
        use_ai = not args.no_ai
        success, json_file, report_file = analyzer.analyze_subdomains(file_path, use_ai)
        
        if success:
            print(f"\n[+] Analysis complete!")
            print(f"[+] Results saved to: {json_file}")
            if report_file:
                print(f"[+] Report saved to: {report_file}")
        else:
            print("[-] Analysis failed.")
            sys.exit(1)
    
    print(f"\n✅ Task completed successfully!")


if __name__ == "__main__":
    main()
