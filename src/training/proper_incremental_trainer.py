#!/usr/bin/env python3
"""
Proper Incremental Training Manager for NETRA
============================================

This fixes the critical issue where the model was being overwritten instead
of accumulating knowledge across training sessions.
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

class ProperIncrementalTrainer:
    """Training manager that properly accumulates knowledge instead of overwriting."""
    
    def __init__(self, chunk_size=1000, max_memory_usage=75):
        print("🔧 Initializing Proper Incremental Trainer...")
        
        # Initialize AI system with incremental classifier
        self.ai = SubdomainAIEnhanced()
        
        # Replace the standard classifier with our incremental one
        self.incremental_classifier = IncrementalLearningClassifier(self.ai.kb)
        
        self.chunk_size = chunk_size
        self.monitor = SystemMonitor(memory_threshold=max_memory_usage)
        self.training_state_file = "incremental_training_state.json"
        
        # Training state
        self.training_state = {
            'chunks_processed': 0,
            'total_chunks': 0,
            'last_session': None,
            'training_started': None,
            'sessions_completed': 0,
            'total_samples_accumulated': 0,
            'last_accuracy': 0.0,
            'best_accuracy': 0.0
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
        
        # Ensure minimum samples per category
        min_samples_per_category = max(5, self.chunk_size // len(category_counts))
        
        # Build balanced chunk data
        chunk_data = []
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
        
        # Fill remaining space randomly if needed
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
        
        # Shuffle and limit
        import random
        random.shuffle(chunk_data)
        return chunk_data[:self.chunk_size]
    
    def process_training_chunk(self, chunk_id: int, chunk_data: List[Tuple[str, str]]) -> Dict:
        """Process a single chunk by adding it to accumulated data."""
        print(f"\\n📥 Processing Training Chunk {chunk_id}")
        print(f"📊 New Samples: {len(chunk_data)}")
        
        # Check system health
        health = self.monitor.check_system_health()
        print(f"💾 Memory: {health['memory_percent']:.1f}% | 🖥️ CPU: {health['cpu_percent']:.1f}%")
        
        if not health['is_system_safe']:
            print("⚠️ System resources low, waiting...")
            if not self.monitor.wait_for_safe_conditions():
                raise RuntimeError("System resources unsafe")
        
        # Add chunk to accumulated training data
        start_time = time.time()
        chunk_stats = self.incremental_classifier.add_training_chunk(chunk_data)
        processing_time = time.time() - start_time
        
        # Force garbage collection
        gc.collect()
        
        print(f"✅ Chunk {chunk_id} processed in {processing_time:.1f}s")
        print(f"📈 Total Accumulated: {chunk_stats['total_accumulated']:,} samples")
        print(f"🏷️ Categories: {chunk_stats['unique_categories']}")
        
        return {
            'success': True,
            'processing_time': processing_time,
            'new_samples_added': chunk_stats['new_samples_added'],
            'total_accumulated': chunk_stats['total_accumulated'],
            'unique_categories': chunk_stats['unique_categories']
        }
    
    def train_accumulated_model(self) -> Dict:
        """Train the model on ALL accumulated data."""
        print("\\n🧠 TRAINING ACCUMULATED MODEL")
        print("=" * 40)
        
        # Check system resources before training
        health = self.monitor.check_system_health()
        if not health['is_system_safe']:
            print("⚠️ Waiting for safe system conditions before training...")
            if not self.monitor.wait_for_safe_conditions(max_wait_minutes=15):
                return {'success': False, 'error': 'System resources unsafe for training'}
        
        # Train on all accumulated data
        training_results = self.incremental_classifier.train_incremental_model(validation_size=0.15)
        
        if training_results['success']:
            # Update training state
            self.training_state['sessions_completed'] += 1
            self.training_state['last_accuracy'] = training_results['val_accuracy']
            self.training_state['total_samples_accumulated'] = training_results['total_samples']
            
            if training_results['val_accuracy'] > self.training_state['best_accuracy']:
                self.training_state['best_accuracy'] = training_results['val_accuracy']
            
            self.training_state['last_session'] = datetime.now().isoformat()
            self.save_training_state()
            
            print("\\n🎉 INCREMENTAL TRAINING COMPLETED!")
            print(f"📊 Validation Accuracy: {training_results['val_accuracy']:.3f}")
            print(f"📈 Total Samples Used: {training_results['total_samples']:,}")
            print(f"🏷️ Categories: {training_results['categories_count']}")
        
        return training_results
    
    def run_comprehensive_training(self, retrain_frequency=10) -> Dict:
        """Run comprehensive training with proper accumulation."""
        print("🚀 PROPER INCREMENTAL TRAINING MANAGER")
        print("✅ Fixing model overwrite issue - using TRUE incremental learning")
        print("=" * 60)
        
        # Get training data info
        total_samples = self.get_training_data_count()
        total_chunks = (total_samples + self.chunk_size - 1) // self.chunk_size
        
        self.training_state['total_chunks'] = total_chunks
        self.training_state['training_started'] = self.training_state.get('training_started') or datetime.now().isoformat()
        
        print(f"📈 Total Training Samples: {total_samples:,}")
        print(f"📦 Total Chunks: {total_chunks}")
        print(f"🔄 Retrain Frequency: Every {retrain_frequency} chunks")
        print(f"✅ Chunks Already Processed: {self.training_state['chunks_processed']}")
        
        # Show current accumulated state
        summary = self.incremental_classifier.get_training_summary()
        print(f"🗂️ Accumulated Samples: {summary['total_samples_trained']:,}")
        print(f"🏷️ Categories Learned: {summary['categories_count']}")
        print(f"🏆 Best Accuracy So Far: {self.training_state['best_accuracy']:.3f}")
        
        start_chunk = self.training_state['chunks_processed']
        accumulated_results = []
        
        try:
            # Process remaining chunks
            for chunk_id in range(start_chunk, total_chunks):
                offset = chunk_id * self.chunk_size
                chunk_data = self.get_balanced_chunk_data(chunk_id, offset)
                
                if not chunk_data:
                    print(f"⚠️ No data for chunk {chunk_id}, skipping...")
                    continue
                
                # Process chunk (add to accumulated data)
                result = self.process_training_chunk(chunk_id, chunk_data)
                
                if result['success']:
                    self.training_state['chunks_processed'] = chunk_id + 1
                    self.save_training_state()
                    accumulated_results.append(result)
                    
                    # Retrain model periodically
                    if (chunk_id + 1) % retrain_frequency == 0 or chunk_id == total_chunks - 1:
                        print(f"\\n🔄 RETRAINING MODEL (after {chunk_id + 1} chunks)...")
                        training_result = self.train_accumulated_model()
                        
                        if training_result['success']:
                            print(f"🎯 New Model Accuracy: {training_result['val_accuracy']:.3f}")
                            print(f"📊 Trained on {training_result['total_samples']:,} total samples")
                        else:
                            print(f"❌ Training failed: {training_result.get('error', 'Unknown error')}")
                    
                    # Brief pause between chunks
                    time.sleep(1)
                    
                else:
                    print(f"❌ Chunk {chunk_id} processing failed")
                    continue
                
                # Check system health periodically
                if (chunk_id + 1) % 5 == 0:
                    health = self.monitor.check_system_health()
                    if not health['is_system_safe']:
                        print("⏸️ Pausing for system health...")
                        self.monitor.wait_for_safe_conditions()
            
            # Final training if not done yet
            if self.training_state['chunks_processed'] >= total_chunks:
                print("\\n🏁 FINAL MODEL TRAINING")
                final_result = self.train_accumulated_model()
                
                print("\\n🎉 COMPLETE INCREMENTAL TRAINING FINISHED!")
                print("=" * 50)
                
                return {
                    'training_completed': True,
                    'total_chunks_processed': self.training_state['chunks_processed'],
                    'total_samples_accumulated': self.training_state['total_samples_accumulated'],
                    'final_accuracy': self.training_state['last_accuracy'],
                    'best_accuracy_achieved': self.training_state['best_accuracy'],
                    'training_sessions': self.training_state['sessions_completed'],
                    'chunks_results': accumulated_results
                }
            else:
                return {
                    'training_completed': False,
                    'progress': f"{self.training_state['chunks_processed']}/{total_chunks}",
                    'can_resume': True
                }
                
        except Exception as e:
            print(f"\\n❌ Training interrupted: {e}")
            print("💾 Progress saved. Run again to resume.")
            self.training_state['error'] = str(e)
            self.training_state['interrupted_at'] = datetime.now().isoformat()
            self.save_training_state()
            return {'training_completed': False, 'error': str(e)}
    
    def run_final_validation(self, test_domains: List[str] = None) -> Dict:
        """Run final validation using the incremental classifier."""
        print("\\n🧪 Final Incremental Model Validation")
        print("=" * 40)
        
        if test_domains is None:
            test_domains = [
                'admin-portal.company.com', 'api-v3.company.com', 'auth-service.company.com',
                'cdn-assets.company.com', 'database-prod.company.com', 'dev-environment.company.com',
                'mail-server.company.com', 'payment-gateway.company.com', 'monitoring-sys.company.com',
                'backup-system.company.com', 'jenkins-ci.company.com', 'vpn-gateway.company.com',
                'blog-cms.company.com', 'support-portal.company.com', 'docs-wiki.company.com'
            ]
        
        print(f"🎯 Testing with {len(test_domains)} subdomains...")
        
        results = []
        high_confidence = 0
        ml_predictions = 0
        
        for domain in test_domains:
            try:
                result = self.incremental_classifier.classify_subdomain(domain)
                results.append(result)
                
                if result.confidence > 0.8:
                    high_confidence += 1
                
                if result.prediction_source == "incremental_ml_model":
                    ml_predictions += 1
                    
            except Exception as e:
                print(f"⚠️ Error classifying {domain}: {e}")
        
        if results:
            avg_confidence = sum(r.confidence for r in results) / len(results)
            ml_usage_ratio = ml_predictions / len(results)
            
            print(f"📊 Average Confidence: {avg_confidence:.3f}")
            print(f"🎯 High Confidence Predictions: {high_confidence}/{len(results)} ({high_confidence/len(results):.1%})")
            print(f"🤖 ML Model Usage: {ml_predictions}/{len(results)} ({ml_usage_ratio:.1%})")
            
            print("\\n🔍 Sample Classifications:")
            for result in results[:10]:
                source_emoji = "🤖" if result.prediction_source == "incremental_ml_model" else "📚" if result.prediction_source == "knowledge_base" else "🔧"
                print(f"  {source_emoji} {result.subdomain} → {result.predicted_category} ({result.confidence:.3f})")
            
            return {
                'validation_successful': True,
                'total_tested': len(results),
                'average_confidence': avg_confidence,
                'high_confidence_ratio': high_confidence / len(results),
                'ml_usage_ratio': ml_usage_ratio,
                'sample_results': results[:5]
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
    """Main training function with proper incremental learning."""
    print("🔧 PROPER INCREMENTAL LEARNING TRAINER")
    print("Fixing the model overwrite issue!")
    print("=" * 50)
    
    # Initialize with conservative settings
    trainer = ProperIncrementalTrainer(chunk_size=750, max_memory_usage=80)
    
    try:
        # Run comprehensive training
        training_results = trainer.run_comprehensive_training(retrain_frequency=8)
        
        if training_results.get('training_completed'):
            # Run final validation
            validation_results = trainer.run_final_validation()
            
            print("\\n📋 FINAL TRAINING SUMMARY")
            print("=" * 35)
            print(f"✅ Chunks Processed: {training_results['total_chunks_processed']}")
            print(f"📊 Total Samples: {training_results['total_samples_accumulated']:,}")
            print(f"🏆 Best Accuracy: {training_results['best_accuracy_achieved']:.3f}")
            print(f"📈 Final Accuracy: {training_results['final_accuracy']:.3f}")
            print(f"🔄 Training Sessions: {training_results['training_sessions']}")
            print(f"🎯 Validation Confidence: {validation_results.get('average_confidence', 0):.3f}")
            print(f"🤖 ML Usage Rate: {validation_results.get('ml_usage_ratio', 0):.1%}")
            
            print("\\n✅ PROPER INCREMENTAL LEARNING COMPLETED!")
            print("🎯 Model now truly accumulates knowledge across all training data")
        else:
            print("\\n⏸️ Training incomplete - can resume later")
            print(f"Progress: {training_results.get('progress', 'Unknown')}")
        
    except KeyboardInterrupt:
        print("\\n⏹️ Training stopped by user. Progress saved.")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        print("💾 Check incremental_training_state.json for recovery.")

if __name__ == "__main__":
    main()
