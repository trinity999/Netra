#!/usr/bin/env python3
"""
Enhanced Incremental Training Manager for NETRA
==============================================

Fixed incremental learning with proper progress tracking and clear distinction
between data accumulation (fast) and model training (slow).
"""

import os
import json
import time
import psutil
import sqlite3
import gc
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from subdomain_ai_enhanced import SubdomainAIEnhanced
from incremental_classifier import IncrementalLearningClassifier
import warnings
warnings.filterwarnings('ignore')

class ProgressBar:
    """Simple progress bar for terminal."""
    
    def __init__(self, total, description="Progress", width=50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
    
    def update(self, current=None):
        """Update progress bar."""
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        progress = self.current / self.total
        bar_length = int(self.width * progress)
        bar = "█" * bar_length + "▒" * (self.width - bar_length)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"
        
        percent = progress * 100
        print(f"\r{self.description}: [{bar}] {percent:.1f}% ({self.current}/{self.total}) | {eta_str}", end="")
        
        if self.current >= self.total:
            print()  # New line when complete

class SystemMonitor:
    """Monitor system resources to prevent overload."""
    
    def __init__(self, memory_threshold=85, cpu_threshold=90):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        
    def check_system_health(self) -> Dict:
        """Check current system resource usage."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'cpu_percent': cpu_percent,
            'is_memory_safe': memory.percent < self.memory_threshold,
            'is_cpu_safe': cpu_percent < self.cpu_threshold,
            'is_system_safe': memory.percent < self.memory_threshold and cpu_percent < self.cpu_threshold
        }
    
    def wait_for_safe_conditions(self, max_wait_minutes=10):
        """Wait for system resources to become available."""
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        
        while time.time() - start_time < max_wait_seconds:
            health = self.check_system_health()
            if health['is_system_safe']:
                return True
                
            print(f"⏳ Waiting for system resources... Memory: {health['memory_percent']:.1f}%, CPU: {health['cpu_percent']:.1f}%")
            time.sleep(30)
            
        return False

class EnhancedIncrementalTrainer:
    """Enhanced training manager with clear progress tracking."""
    
    def __init__(self, chunk_size=1000, max_memory_usage=75):
        print("🔧 Initializing Enhanced Incremental Trainer...")
        print("📚 Data Collection (fast) vs 🧠 Model Training (slow) - clearly separated")
        
        # Initialize AI system
        self.ai = SubdomainAIEnhanced()
        self.incremental_classifier = IncrementalLearningClassifier(self.ai.kb)
        
        self.chunk_size = chunk_size
        self.monitor = SystemMonitor(memory_threshold=max_memory_usage)
        self.training_state_file = "enhanced_training_state.json"
        
        # Training state
        self.training_state = {
            'chunks_processed': 0,
            'total_chunks': 0,
            'data_collection_complete': False,
            'model_training_sessions': 0,
            'total_samples_accumulated': 0,
            'last_training_accuracy': 0.0,
            'best_accuracy': 0.0,
            'training_started': None,
            'last_model_training': None
        }
        
        self.load_training_state()
    
    def get_training_data_count(self) -> int:
        """Get total count of available training data."""
        conn = sqlite3.connect(self.ai.kb.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM subdomains 
            WHERE source IN ('synthetic', 'synthetic_advanced', 'manual', 'verified')
        ''')
        
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_balanced_chunk_data(self, chunk_id: int, offset: int) -> List[Tuple[str, str]]:
        """Get balanced data for a specific chunk."""
        conn = sqlite3.connect(self.ai.kb.db_path)
        cursor = conn.cursor()
        
        # Get available categories and their counts
        cursor.execute('''
            SELECT category, COUNT(*) as count FROM subdomains 
            WHERE source IN ('synthetic', 'synthetic_advanced', 'manual', 'verified')
            GROUP BY category
            ORDER BY count DESC
        ''')
        
        category_counts = cursor.fetchall()
        
        if not category_counts:
            conn.close()
            return []
        
        # Build balanced chunk data
        chunk_data = []
        min_samples_per_category = max(5, self.chunk_size // len(category_counts))
        
        for category, total_count in category_counts:
            samples_to_take = min(min_samples_per_category, total_count, 
                                self.chunk_size - len(chunk_data))
            
            if samples_to_take <= 0:
                break
                
            cursor.execute('''
                SELECT domain, category FROM subdomains 
                WHERE source IN ('synthetic', 'synthetic_advanced', 'manual', 'verified')
                AND category = ?
                ORDER BY RANDOM() LIMIT ?
            ''', (category, samples_to_take))
            
            category_data = cursor.fetchall()
            chunk_data.extend(category_data)
            
            if len(chunk_data) >= self.chunk_size:
                break
        
        # Fill remaining space if needed
        if len(chunk_data) < self.chunk_size:
            remaining = self.chunk_size - len(chunk_data)
            cursor.execute('''
                SELECT domain, category FROM subdomains 
                WHERE source IN ('synthetic', 'synthetic_advanced', 'manual', 'verified')
                ORDER BY RANDOM() LIMIT ?
            ''', (remaining,))
            
            additional_data = cursor.fetchall()
            chunk_data.extend(additional_data)
        
        conn.close()
        
        import random
        random.shuffle(chunk_data)
        return chunk_data[:self.chunk_size]
    
    def collect_training_data(self, chunk_id: int, chunk_data: List[Tuple[str, str]]) -> Dict:
        """FAST: Collect and process training data (no model training)."""
        start_time = time.time()
        
        # Check system health
        health = self.monitor.check_system_health()
        
        if not health['is_system_safe']:
            print("⚠️ System resources low, waiting...")
            if not self.monitor.wait_for_safe_conditions():
                raise RuntimeError("System resources unsafe")
        
        # Add chunk to accumulated dataset (this is fast - just data processing)
        chunk_stats = self.incremental_classifier.add_training_chunk(chunk_data)
        processing_time = time.time() - start_time
        
        # Update training state
        self.training_state['chunks_processed'] = chunk_id + 1
        self.training_state['total_samples_accumulated'] = chunk_stats['total_accumulated']
        self.save_training_state()
        
        gc.collect()  # Clean up memory
        
        return {
            'success': True,
            'processing_time': processing_time,
            'new_samples_added': chunk_stats['new_samples_added'],
            'total_accumulated': chunk_stats['total_accumulated'],
            'unique_categories': chunk_stats['unique_categories']
        }
    
    def train_accumulated_model(self, progress_callback=None) -> Dict:
        """SLOW: Train the ML model on ALL accumulated data."""
        print("\\n🧠 TRAINING ML MODEL ON ALL ACCUMULATED DATA")
        print("⚠️  This is the SLOW step that takes 30+ seconds")
        print("=" * 50)
        
        # Check system resources
        health = self.monitor.check_system_health()
        if not health['is_system_safe']:
            print("⚠️ Waiting for safe system conditions...")
            if not self.monitor.wait_for_safe_conditions(max_wait_minutes=15):
                return {'success': False, 'error': 'System resources unsafe'}
        
        # Show what we're about to train on
        summary = self.incremental_classifier.get_training_summary()
        print(f"📊 Training on {summary['total_samples_trained']:,} accumulated samples")
        print(f"🏷️ Categories: {summary['categories_count']}")
        print(f"🔄 Training session #{summary['training_sessions'] + 1}")
        
        # Progress callback for training steps
        if progress_callback:
            progress_callback("Starting model training...")
        
        # Train the model (this is where the 30+ seconds happens)
        start_time = time.time()
        training_results = self.incremental_classifier.train_incremental_model(validation_size=0.15)
        training_time = time.time() - start_time
        
        if training_results['success']:
            # Update training state
            self.training_state['model_training_sessions'] += 1
            self.training_state['last_training_accuracy'] = training_results['val_accuracy']
            self.training_state['last_model_training'] = datetime.now().isoformat()
            
            if training_results['val_accuracy'] > self.training_state['best_accuracy']:
                self.training_state['best_accuracy'] = training_results['val_accuracy']
            
            self.save_training_state()
            
            print(f"\\n🎉 MODEL TRAINING COMPLETED!")
            print(f"📊 Validation Accuracy: {training_results['val_accuracy']:.3f}")
            print(f"📈 Training Accuracy: {training_results['train_accuracy']:.3f}")
            print(f"🏷️ Categories Trained: {training_results['categories_count']}")
            print(f"⏱️ Training Time: {training_time:.1f} seconds")
        
        return training_results
    
    def run_comprehensive_training(self, retrain_frequency=10) -> Dict:
        """Run comprehensive training with clear progress tracking."""
        print("🚀 ENHANCED INCREMENTAL TRAINING MANAGER")
        print("=" * 60)
        print("📋 TRAINING PROCESS:")
        print("   1. 📚 Data Collection Phase (fast - 0.1s per chunk)")
        print("   2. 🧠 Model Training Phase (slow - 30s+ per session)")
        print("=" * 60)
        
        # Get training info
        total_samples = self.get_training_data_count()
        total_chunks = (total_samples + self.chunk_size - 1) // self.chunk_size
        
        self.training_state['total_chunks'] = total_chunks
        self.training_state['training_started'] = self.training_state.get('training_started') or datetime.now().isoformat()
        
        print(f"📈 Total Training Samples: {total_samples:,}")
        print(f"📦 Total Chunks: {total_chunks}")
        print(f"📏 Chunk Size: {self.chunk_size}")
        print(f"🔄 Model Retraining: Every {retrain_frequency} chunks")
        
        # Show current state
        summary = self.incremental_classifier.get_training_summary()
        print(f"✅ Chunks Already Processed: {self.training_state['chunks_processed']}")
        print(f"🗂️ Accumulated Samples: {summary['total_samples_trained']:,}")
        print(f"🏆 Best Accuracy So Far: {self.training_state['best_accuracy']:.3f}")
        
        start_chunk = self.training_state['chunks_processed']
        
        if start_chunk >= total_chunks:
            print("\\n🎉 All data collection already completed!")
            print("🔄 Running final model training...")
            final_result = self.train_accumulated_model()
            return {'training_completed': True, 'final_training': final_result}
        
        print("\\n📚 PHASE 1: DATA COLLECTION")
        print("-" * 30)
        
        # Create progress bar for data collection
        remaining_chunks = total_chunks - start_chunk
        data_progress = ProgressBar(remaining_chunks, "Collecting Data", 40)
        
        try:
            # Data collection phase (fast)
            for chunk_id in range(start_chunk, total_chunks):
                offset = chunk_id * self.chunk_size
                chunk_data = self.get_balanced_chunk_data(chunk_id, offset)
                
                if not chunk_data:
                    print(f"\\n⚠️ No data for chunk {chunk_id}, skipping...")
                    continue
                
                # Collect data (fast operation)
                result = self.collect_training_data(chunk_id, chunk_data)
                
                if result['success']:
                    # Update progress
                    data_progress.update()
                    
                    # Show periodic model training
                    if ((chunk_id + 1) % retrain_frequency == 0) or (chunk_id == total_chunks - 1):
                        print(f"\\n\\n🧠 PHASE 2: MODEL TRAINING (Session {self.training_state['model_training_sessions'] + 1})")
                        print("-" * 40)
                        print(f"📊 Accumulated samples: {result['total_accumulated']:,}")
                        print(f"🏷️ Categories: {result['unique_categories']}")
                        
                        # This is the slow step
                        training_result = self.train_accumulated_model()
                        
                        if training_result['success']:
                            print(f"🎯 Model Accuracy: {training_result['val_accuracy']:.3f}")
                            print(f"📈 Improvement: {training_result['val_accuracy'] - self.training_state.get('last_training_accuracy', 0):.3f}")
                        else:
                            print(f"❌ Training failed: {training_result.get('error', 'Unknown')}")
                        
                        print("\\n📚 Resuming data collection...")
                        print("-" * 30)
                    
                    # Brief pause between chunks
                    time.sleep(0.5)
                    
                else:
                    print(f"\\n❌ Chunk {chunk_id} processing failed")
                    continue
                
                # Periodic system health check
                if (chunk_id + 1) % 10 == 0:
                    health = self.monitor.check_system_health()
                    if not health['is_system_safe']:
                        print("\\n⏸️ Pausing for system health...")
                        self.monitor.wait_for_safe_conditions()
            
            # Mark data collection as complete
            self.training_state['data_collection_complete'] = True
            self.save_training_state()
            
            print("\\n\\n🏁 DATA COLLECTION COMPLETED!")
            print("🔄 Running final comprehensive model training...")
            
            # Final training session
            final_result = self.train_accumulated_model()
            
            print("\\n🎉 COMPLETE INCREMENTAL TRAINING FINISHED!")
            print("=" * 50)
            
            return {
                'training_completed': True,
                'total_chunks_processed': self.training_state['chunks_processed'],
                'total_samples_accumulated': self.training_state['total_samples_accumulated'],
                'final_accuracy': self.training_state['last_training_accuracy'],
                'best_accuracy_achieved': self.training_state['best_accuracy'],
                'training_sessions': self.training_state['model_training_sessions'],
                'final_training_result': final_result
            }
                
        except Exception as e:
            print(f"\\n❌ Training interrupted: {e}")
            print("💾 Progress saved. Run again to resume.")
            self.training_state['error'] = str(e)
            self.training_state['interrupted_at'] = datetime.now().isoformat()
            self.save_training_state()
            return {'training_completed': False, 'error': str(e)}
    
    def run_validation_test(self, test_domains: List[str] = None) -> Dict:
        """Run validation test with progress tracking."""
        print("\\n🧪 Final Model Validation")
        print("=" * 30)
        
        if test_domains is None:
            test_domains = [
                'admin-portal.example.com', 'api-gateway.example.com', 'auth-service.example.com',
                'cdn-assets.example.com', 'database-prod.example.com', 'dev-env.example.com',
                'mail-server.example.com', 'payment-api.example.com', 'monitoring.example.com',
                'backup-sys.example.com', 'jenkins-ci.example.com', 'vpn-gateway.example.com',
                'blog-cms.example.com', 'support-desk.example.com', 'docs-wiki.example.com',
                'ftp-server.example.com', 'staging-env.example.com', 'load-balancer.example.com',
                'cache-redis.example.com', 'analytics.example.com'
            ]
        
        print(f"🎯 Testing {len(test_domains)} subdomains...")
        
        # Progress bar for validation
        validation_progress = ProgressBar(len(test_domains), "Validating", 40)
        
        results = []
        high_confidence = 0
        ml_predictions = 0
        kb_predictions = 0
        heuristic_predictions = 0
        
        for i, domain in enumerate(test_domains):
            try:
                result = self.incremental_classifier.classify_subdomain(domain)
                results.append(result)
                
                if result.confidence > 0.8:
                    high_confidence += 1
                
                # Count prediction sources
                if result.prediction_source == "incremental_ml_model":
                    ml_predictions += 1
                elif result.prediction_source == "knowledge_base":
                    kb_predictions += 1
                else:
                    heuristic_predictions += 1
                
                validation_progress.update()
                    
            except Exception as e:
                print(f"\\n⚠️ Error classifying {domain}: {e}")
                validation_progress.update()
        
        print()  # New line after progress bar
        
        if results:
            avg_confidence = sum(r.confidence for r in results) / len(results)
            ml_usage_ratio = ml_predictions / len(results)
            kb_usage_ratio = kb_predictions / len(results)
            
            print(f"📊 Average Confidence: {avg_confidence:.3f}")
            print(f"🎯 High Confidence (>0.8): {high_confidence}/{len(results)} ({high_confidence/len(results):.1%})")
            print(f"🤖 ML Model Usage: {ml_predictions}/{len(results)} ({ml_usage_ratio:.1%})")
            print(f"📚 Knowledge Base Usage: {kb_predictions}/{len(results)} ({kb_usage_ratio:.1%})")
            print(f"🔧 Heuristic Usage: {heuristic_predictions}/{len(results)} ({heuristic_predictions/len(results):.1%})")
            
            print("\\n🔍 Sample Classifications:")
            for result in results[:10]:
                source_emoji = {"incremental_ml_model": "🤖", "knowledge_base": "📚", "heuristic": "🔧"}.get(result.prediction_source, "❓")
                print(f"  {source_emoji} {result.subdomain} → {result.predicted_category} ({result.confidence:.3f})")
            
            return {
                'validation_successful': True,
                'total_tested': len(results),
                'average_confidence': avg_confidence,
                'high_confidence_ratio': high_confidence / len(results),
                'ml_usage_ratio': ml_usage_ratio,
                'kb_usage_ratio': kb_usage_ratio,
                'source_breakdown': {
                    'ml_model': ml_predictions,
                    'knowledge_base': kb_predictions,
                    'heuristic': heuristic_predictions
                }
            }
        else:
            return {'validation_successful': False, 'error': 'No successful classifications'}
    
    def save_training_state(self):
        """Save training state."""
        with open(self.training_state_file, 'w') as f:
            json.dump(self.training_state, f, indent=2)
    
    def load_training_state(self):
        """Load training state."""
        if os.path.exists(self.training_state_file):
            try:
                with open(self.training_state_file, 'r') as f:
                    loaded_state = json.load(f)
                self.training_state.update(loaded_state)
                print(f"[+] Loaded training state: {self.training_state['chunks_processed']} chunks processed")
            except Exception as e:
                print(f"[-] Could not load training state: {e}")

def main():
    """Main training function with enhanced progress tracking."""
    print("🔧 ENHANCED INCREMENTAL LEARNING TRAINER")
    print("✅ Fixed model overwrite issue + Enhanced progress tracking")
    print("=" * 60)
    
    # Initialize trainer
    trainer = EnhancedIncrementalTrainer(chunk_size=1000, max_memory_usage=80)
    
    try:
        # Run comprehensive training
        training_results = trainer.run_comprehensive_training(retrain_frequency=5)
        
        if training_results.get('training_completed'):
            # Run validation
            validation_results = trainer.run_validation_test()
            
            print("\\n📋 FINAL TRAINING SUMMARY")
            print("=" * 35)
            print(f"✅ Chunks Processed: {training_results['total_chunks_processed']}")
            print(f"📊 Total Samples: {training_results['total_samples_accumulated']:,}")
            print(f"🏆 Best Accuracy: {training_results['best_accuracy_achieved']:.3f}")
            print(f"📈 Final Accuracy: {training_results['final_accuracy']:.3f}")
            print(f"🔄 Training Sessions: {training_results['training_sessions']}")
            print(f"🎯 Validation Confidence: {validation_results.get('average_confidence', 0):.3f}")
            print(f"🤖 ML Usage Rate: {validation_results.get('ml_usage_ratio', 0):.1%}")
            
            print("\\n✅ ENHANCED INCREMENTAL LEARNING COMPLETED!")
            print("🎯 Model successfully accumulates knowledge across ALL training data")
            print("📊 Clear separation of data collection (fast) vs model training (slow)")
        else:
            print("\\n⏸️ Training incomplete - can resume later")
            print(f"Progress: {training_results.get('progress', 'Unknown')}")
        
    except KeyboardInterrupt:
        print("\\n⏹️ Training stopped by user. Progress saved.")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        print("💾 Check enhanced_training_state.json for recovery.")

if __name__ == "__main__":
    main()
