#!/usr/bin/env python3
"""
Learning Curve Analyzer for Subdomain AI Enhanced
================================================

Analyzes learning curves, detects saturation points, and provides
insights for optimal training data requirements.
"""

import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from subdomain_ai_enhanced import SubdomainAIEnhanced
import random

class LearningCurveAnalyzer:
    """Analyzes learning curves and detects saturation."""
    
    def __init__(self, ai_system: SubdomainAIEnhanced):
        self.ai = ai_system
        self.results_history = []
        
    def run_comprehensive_analysis(self, test_file: str, max_training_size: int = 100000) -> Dict:
        """Run comprehensive learning curve analysis."""
        print("📈 Learning Curve Analysis")
        print("=" * 30)
        
        # Define training sizes (logarithmic scale)
        sizes = self._generate_training_sizes(max_training_size)
        
        results = {
            'training_sizes': [],
            'accuracies': [],
            'confidences': [],
            'kb_usage': [],
            'ml_usage': [],
            'training_times': [],
            'saturation_point': None,
            'recommendations': []
        }
        
        print(f"[*] Testing with training sizes: {sizes}")
        
        for size in sizes:
            print(f"\n[*] Testing with {size:,} training samples...")
            
            # Get training data from knowledge base
            training_data = self._get_training_data(size)
            if not training_data:
                print(f"[-] Insufficient training data for size {size}")
                continue
            
            # Time the training
            start_time = datetime.now()
            
            # Train model
            success = self.ai.classifier.train_model(training_data, test_size=0.1)
            if not success:
                print(f"[-] Training failed for size {size}")
                continue
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Run benchmark
            metrics = self.ai.benchmark(test_file)
            
            # Store results
            results['training_sizes'].append(size)
            results['accuracies'].append(metrics['accuracy'])
            results['confidences'].append(metrics['average_confidence'])
            results['kb_usage'].append(metrics['knowledge_base_usage'])
            results['ml_usage'].append(metrics['ml_model_usage'])
            results['training_times'].append(training_time)
            
            print(f"    Accuracy: {metrics['accuracy']:.3f}")
            print(f"    Confidence: {metrics['average_confidence']:.3f}")
            print(f"    Training Time: {training_time:.1f}s")
        
        # Analyze saturation
        saturation_info = self._detect_saturation(results['training_sizes'], results['accuracies'])
        results['saturation_point'] = saturation_info
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results['recommendations'] = recommendations
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _generate_training_sizes(self, max_size: int) -> List[int]:
        """Generate logarithmically spaced training sizes."""
        # Start with small sizes and grow exponentially
        sizes = [100, 250, 500, 1000, 2500, 5000]
        
        # Add logarithmic progression
        current = 10000
        while current <= max_size:
            sizes.append(current)
            current = int(current * 1.5)  # 50% increase each time
        
        return sorted(list(set(sizes)))
    
    def _get_training_data(self, size: int) -> List[Tuple[str, str]]:
        """Get training data from knowledge base."""
        conn = sqlite3.connect(self.ai.kb.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT domain, category FROM subdomains 
            WHERE source IN ('synthetic', 'manual', 'verified')
            ORDER BY RANDOM() LIMIT ?
        ''', (size,))
        
        training_data = cursor.fetchall()
        conn.close()
        
        return training_data
    
    def _detect_saturation(self, sizes: List[int], accuracies: List[float]) -> Optional[Dict]:
        """Detect saturation point in learning curve."""
        if len(accuracies) < 3:
            return None
        
        # Calculate improvement rates
        improvements = []
        for i in range(1, len(accuracies)):
            improvement = accuracies[i] - accuracies[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            improvement_rate = improvement / size_ratio
            improvements.append(improvement_rate)
        
        # Find saturation point (where improvement rate becomes very small)
        saturation_threshold = 0.001  # 0.1% improvement per size doubling
        
        for i, rate in enumerate(improvements):
            if rate < saturation_threshold and i >= 2:  # Need at least 3 points
                return {
                    'detected': True,
                    'saturation_size': sizes[i + 1],
                    'saturation_accuracy': accuracies[i + 1],
                    'improvement_rate': rate,
                    'point_index': i + 1
                }
        
        return {
            'detected': False,
            'latest_improvement_rate': improvements[-1] if improvements else 0,
            'trend': 'still_improving' if improvements and improvements[-1] > saturation_threshold else 'plateauing'
        }
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        if not results['accuracies']:
            return ["Insufficient data for recommendations"]
        
        max_accuracy = max(results['accuracies'])
        latest_accuracy = results['accuracies'][-1]
        saturation = results['saturation_point']
        
        # Accuracy recommendations
        if max_accuracy < 0.7:
            recommendations.append("🔴 Low accuracy detected. Consider improving feature extraction or adding more diverse training data.")
        elif max_accuracy < 0.85:
            recommendations.append("🟡 Moderate accuracy achieved. Focus on quality over quantity of training data.")
        else:
            recommendations.append("🟢 Good accuracy achieved. System is performing well.")
        
        # Saturation recommendations
        if saturation and saturation.get('detected'):
            size = saturation['saturation_size']
            recommendations.append(f"📊 Saturation detected at {size:,} samples. Additional data beyond this point shows diminishing returns.")
            recommendations.append(f"💡 Optimal training size: {size:,} samples for best efficiency.")
        else:
            recommendations.append("📈 No clear saturation detected. System may benefit from more training data.")
        
        # Usage pattern recommendations
        if results['ml_usage']:
            avg_ml_usage = sum(results['ml_usage']) / len(results['ml_usage'])
            if avg_ml_usage < 0.5:
                recommendations.append("⚙️ ML model is underutilized. Consider improving model confidence or knowledge base coverage.")
            else:
                recommendations.append("✅ Good balance between ML model and knowledge base usage.")
        
        # Training time recommendations
        if results['training_times']:
            max_time = max(results['training_times'])
            if max_time > 300:  # 5 minutes
                recommendations.append("⏱️ Long training times detected. Consider feature selection or model optimization.")
        
        return recommendations
    
    def _save_results(self, results: Dict):
        """Save analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"learning_curve_analysis_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (list, tuple)):
                serializable_results[key] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in value]
            else:
                serializable_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n[+] Analysis results saved to: {filename}")
    
    def create_visualizations(self, results: Dict):
        """Create visualization plots."""
        if not results['accuracies']:
            print("[-] No data to visualize")
            return
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        sizes = results['training_sizes']
        
        # 1. Learning Curve (Accuracy vs Training Size)
        ax1.semilogx(sizes, results['accuracies'], 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Training Size')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Learning Curve: Accuracy vs Training Size')
        ax1.grid(True, alpha=0.3)
        
        # Mark saturation point if detected
        saturation = results['saturation_point']
        if saturation and saturation.get('detected'):
            idx = saturation['point_index']
            ax1.axvline(x=sizes[idx], color='red', linestyle='--', alpha=0.7, label='Saturation Point')
            ax1.legend()
        
        # 2. Confidence vs Training Size
        ax2.semilogx(sizes, results['confidences'], 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Training Size')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Confidence vs Training Size')
        ax2.grid(True, alpha=0.3)
        
        # 3. Usage Patterns
        ax3.semilogx(sizes, results['ml_usage'], 'r-o', label='ML Model Usage', linewidth=2)
        ax3.semilogx(sizes, results['kb_usage'], 'b-o', label='Knowledge Base Usage', linewidth=2)
        ax3.set_xlabel('Training Size')
        ax3.set_ylabel('Usage Ratio')
        ax3.set_title('Prediction Source Usage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Training Time vs Size
        if results['training_times']:
            ax4.loglog(sizes, results['training_times'], 'purple', marker='o', linewidth=2, markersize=6)
            ax4.set_xlabel('Training Size')
            ax4.set_ylabel('Training Time (seconds)')
            ax4.set_title('Training Time vs Size')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"learning_curves_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[+] Visualizations saved to: {filename}")
        
        plt.show()

class PerformanceMonitor:
    """Monitor system performance over time."""
    
    def __init__(self, ai_system: SubdomainAIEnhanced):
        self.ai = ai_system
    
    def monitor_realtime_performance(self, test_subdomains: List[str]) -> Dict:
        """Monitor real-time classification performance."""
        print("📊 Real-time Performance Monitor")
        print("=" * 35)
        
        results = []
        source_counts = {'knowledge_base': 0, 'ml_model': 0, 'heuristic': 0}
        confidence_scores = []
        processing_times = []
        
        for i, subdomain in enumerate(test_subdomains):
            start_time = datetime.now()
            
            result = self.ai.classifier.classify_subdomain(subdomain)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
            
            results.append(result)
            source_counts[result.prediction_source] += 1
            confidence_scores.append(result.confidence)
            processing_times.append(processing_time)
            
            # Print progress every 100 items
            if (i + 1) % 100 == 0:
                avg_time = sum(processing_times[-100:]) / min(100, len(processing_times))
                print(f"[*] Processed {i + 1}/{len(test_subdomains)} - Avg time: {avg_time:.2f}ms")
        
        # Calculate metrics
        total_count = len(results)
        avg_confidence = sum(confidence_scores) / total_count
        avg_processing_time = sum(processing_times) / total_count
        
        performance_metrics = {
            'total_processed': total_count,
            'average_confidence': avg_confidence,
            'average_processing_time_ms': avg_processing_time,
            'source_distribution': {
                source: count / total_count for source, count in source_counts.items()
            },
            'confidence_distribution': {
                'high_confidence': sum(1 for c in confidence_scores if c > 0.8) / total_count,
                'medium_confidence': sum(1 for c in confidence_scores if 0.5 <= c <= 0.8) / total_count,
                'low_confidence': sum(1 for c in confidence_scores if c < 0.5) / total_count
            }
        }
        
        self._print_performance_report(performance_metrics)
        return performance_metrics
    
    def _print_performance_report(self, metrics: Dict):
        """Print formatted performance report."""
        print("\n[+] Performance Report:")
        print("-" * 40)
        print(f"Total Processed: {metrics['total_processed']:,}")
        print(f"Average Confidence: {metrics['average_confidence']:.3f}")
        print(f"Average Processing Time: {metrics['average_processing_time_ms']:.2f} ms")
        
        print("\nPrediction Sources:")
        for source, ratio in metrics['source_distribution'].items():
            print(f"  {source}: {ratio:.2%}")
        
        print("\nConfidence Distribution:")
        for level, ratio in metrics['confidence_distribution'].items():
            print(f"  {level}: {ratio:.2%}")

def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Learning Curve Analyzer for Subdomain AI")
    parser.add_argument('--test-file', default='test_subdomains.txt', help='Test subdomains file')
    parser.add_argument('--max-size', type=int, default=50000, help='Maximum training size to test')
    parser.add_argument('--visualize', action='store_true', help='Create visualization plots')
    parser.add_argument('--monitor', action='store_true', help='Run real-time performance monitoring')
    
    args = parser.parse_args()
    
    # Initialize AI system
    ai = SubdomainAIEnhanced()
    
    if args.monitor:
        # Load test subdomains for monitoring
        try:
            with open(args.test_file, 'r') as f:
                test_subs = [line.strip() for line in f if line.strip()][:1000]  # Limit for demo
            
            monitor = PerformanceMonitor(ai)
            monitor.monitor_realtime_performance(test_subs)
        except FileNotFoundError:
            print(f"[-] Test file not found: {args.test_file}")
    else:
        # Run learning curve analysis
        analyzer = LearningCurveAnalyzer(ai)
        results = analyzer.run_comprehensive_analysis(args.test_file, args.max_size)
        
        if args.visualize:
            try:
                analyzer.create_visualizations(results)
            except ImportError:
                print("[-] Matplotlib not available for visualization")
        
        # Print summary
        print("\n" + "="*60)
        print("📋 ANALYSIS SUMMARY")
        print("="*60)
        
        if results['recommendations']:
            print("\n🎯 Recommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"{i}. {rec}")

if __name__ == "__main__":
    main()
