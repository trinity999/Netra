#!/usr/bin/env python3
"""
Safe Training Manager for NETRA System
=====================================

Provides memory-aware, crash-resistant training with system monitoring
and automatic recovery capabilities.
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
import warnings
warnings.filterwarnings('ignore')

class SystemMonitor:
    """Monitor system resources to prevent overload."""
    
    def __init__(self, memory_threshold=85, cpu_threshold=90):
        self.memory_threshold = memory_threshold  # % of total memory
        self.cpu_threshold = cpu_threshold        # % CPU usage
        self.initial_memory = psutil.virtual_memory().percent
        
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
            time.sleep(30)  # Check every 30 seconds
            
        return False

class SafeTrainingManager:
    """Manages safe, chunked training with crash recovery."""
    
    def __init__(self, chunk_size=1000, max_memory_usage=70):
        self.ai = SubdomainAIEnhanced()
        self.chunk_size = chunk_size
        self.monitor = SystemMonitor(memory_threshold=max_memory_usage)
        self.training_state_file = "training_state.json"
        self.checkpoint_dir = "training_checkpoints"
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
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
    
    def load_training_state(self) -> Dict:
        """Load previous training state if exists."""
        if os.path.exists(self.training_state_file):
            try:
                with open(self.training_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load training state: {e}")
        
        return {
            'completed_chunks': 0,
            'total_chunks': 0,
            'last_checkpoint': None,
            'training_started': datetime.now().isoformat(),
            'current_accuracy': 0.0,
            'best_accuracy': 0.0,
            'training_samples_processed': 0
        }
    
    def save_training_state(self, state: Dict):
        """Save current training state."""
        state['last_updated'] = datetime.now().isoformat()
        with open(self.training_state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def create_checkpoint(self, chunk_id: int, accuracy: float) -> str:
        """Create a training checkpoint."""
        checkpoint_name = f"checkpoint_chunk_{chunk_id:03d}_{accuracy:.3f}.json"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint_data = {
            'chunk_id': chunk_id,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
            'system_info': self.monitor.check_system_health()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        return checkpoint_path
    
    def get_chunk_data(self, chunk_id: int, offset: int) -> List[Tuple[str, str]]:
        """Get data for a specific chunk with balanced sampling."""
        conn = sqlite3.connect(self.ai.kb.db_path)
        cursor = conn.cursor()
        
        # First, get available categories and their counts
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
        
        # Calculate samples per category to ensure minimum 2 samples per class
        min_samples_per_category = max(2, self.chunk_size // len(category_counts))
        
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
        
        # If still need more data, fill randomly
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
        
        # Shuffle the final dataset
        import random
        random.shuffle(chunk_data)
        
        return chunk_data[:self.chunk_size]
    
    def train_chunk(self, chunk_id: int, chunk_data: List[Tuple[str, str]]) -> Dict:
        """Train on a single chunk of data."""
        print(f"\n🔄 Training Chunk {chunk_id}")
        print(f"📊 Samples: {len(chunk_data)}")
        
        # Check system health before training
        health = self.monitor.check_system_health()
        print(f"💾 Memory: {health['memory_percent']:.1f}% | 🖥️ CPU: {health['cpu_percent']:.1f}%")
        
        if not health['is_system_safe']:
            print("⚠️ System resources low, waiting for safe conditions...")
            if not self.monitor.wait_for_safe_conditions():
                raise RuntimeError("System resources remained unsafe for too long")
        
        # Train the model
        start_time = time.time()
        success = self.ai.classifier.train_model(chunk_data, test_size=0.1)
        training_time = time.time() - start_time
        
        if not success:
            return {'success': False, 'error': 'Training failed'}
        
        # Quick validation
        test_samples = chunk_data[:min(100, len(chunk_data))]
        correct = 0
        
        for domain, expected_category in test_samples:
            try:
                result = self.ai.classifier.classify_subdomain(domain)
                if result.predicted_category.lower() == expected_category.lower():
                    correct += 1
            except Exception:
                pass
        
        accuracy = correct / len(test_samples) if test_samples else 0.0
        
        # Force garbage collection
        gc.collect()
        
        return {
            'success': True,
            'accuracy': accuracy,
            'training_time': training_time,
            'samples_trained': len(chunk_data),
            'system_health': self.monitor.check_system_health()
        }
    
    def resume_training(self) -> Dict:
        """Resume or start training from last checkpoint."""
        print("🚀 NETRA Safe Training Manager")
        print("=" * 40)
        
        # Load training state
        state = self.load_training_state()
        total_samples = self.get_training_data_count()
        total_chunks = (total_samples + self.chunk_size - 1) // self.chunk_size
        
        state['total_chunks'] = total_chunks
        
        print(f"📈 Total Training Samples: {total_samples:,}")
        print(f"📦 Total Chunks: {total_chunks}")
        print(f"✅ Completed Chunks: {state['completed_chunks']}")
        print(f"🎯 Best Accuracy So Far: {state['best_accuracy']:.3f}")
        
        if state['completed_chunks'] >= total_chunks:
            print("🎉 Training already completed!")
            return state
        
        # Continue from where we left off
        start_chunk = state['completed_chunks']
        results = []
        
        try:
            for chunk_id in range(start_chunk, total_chunks):
                offset = chunk_id * self.chunk_size
                chunk_data = self.get_chunk_data(chunk_id, offset)
                
                if not chunk_data:
                    print(f"⚠️ No data for chunk {chunk_id}, skipping...")
                    continue
                
                # Train this chunk
                result = self.train_chunk(chunk_id, chunk_data)
                
                if result['success']:
                    accuracy = result['accuracy']
                    
                    # Update state
                    state['completed_chunks'] = chunk_id + 1
                    state['current_accuracy'] = accuracy
                    state['training_samples_processed'] += result['samples_trained']
                    
                    if accuracy > state['best_accuracy']:
                        state['best_accuracy'] = accuracy
                        # Create checkpoint for best model
                        checkpoint_path = self.create_checkpoint(chunk_id, accuracy)
                        state['last_checkpoint'] = checkpoint_path
                        print(f"🏆 New best accuracy: {accuracy:.3f} - Checkpoint saved")
                    
                    # Save state after each chunk
                    self.save_training_state(state)
                    
                    print(f"✅ Chunk {chunk_id} completed - Accuracy: {accuracy:.3f} - Time: {result['training_time']:.1f}s")
                    results.append(result)
                    
                    # Brief pause between chunks
                    time.sleep(2)
                    
                else:
                    print(f"❌ Chunk {chunk_id} failed: {result.get('error', 'Unknown error')}")
                    # Continue with next chunk rather than stopping
                    continue
                
                # Check if we should pause for system health
                health = self.monitor.check_system_health()
                if not health['is_system_safe']:
                    print("⏸️ Pausing training due to system resource constraints...")
                    self.monitor.wait_for_safe_conditions()
            
            print("\n🎉 Training completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Training interrupted: {e}")
            print("💾 Progress saved. Run again to resume from last checkpoint.")
            state['error'] = str(e)
            state['interrupted_at'] = datetime.now().isoformat()
            self.save_training_state(state)
        
        return state
    
    def run_final_validation(self, test_file: str = None) -> Dict:
        """Run comprehensive validation of the trained model."""
        print("\n🧪 Final Model Validation")
        print("=" * 30)
        
        # Use built-in test samples if no file provided
        test_subdomains = [
            'admin.company.com', 'api.company.com', 'auth.company.com',
            'cdn.company.com', 'db.company.com', 'dev.company.com',
            'ftp.company.com', 'mail.company.com', 'staging.company.com',
            'test.company.com', 'www.company.com', 'blog.company.com',
            'shop.company.com', 'support.company.com', 'docs.company.com'
        ]
        
        if test_file and os.path.exists(test_file):
            with open(test_file, 'r') as f:
                file_subdomains = [line.strip() for line in f if line.strip()]
                if file_subdomains:
                    test_subdomains = file_subdomains[:100]  # Limit for validation
        
        print(f"🎯 Testing with {len(test_subdomains)} subdomains...")
        
        results = []
        high_confidence = 0
        
        for subdomain in test_subdomains:
            try:
                result = self.ai.classifier.classify_subdomain(subdomain)
                results.append(result)
                if result.confidence > 0.8:
                    high_confidence += 1
                    
            except Exception as e:
                print(f"⚠️ Error classifying {subdomain}: {e}")
        
        if results:
            avg_confidence = sum(r.confidence for r in results) / len(results)
            print(f"📊 Average Confidence: {avg_confidence:.3f}")
            print(f"🎯 High Confidence Predictions: {high_confidence}/{len(results)} ({high_confidence/len(results):.1%})")
            
            # Show sample results
            print("\n🔍 Sample Classifications:")
            for result in results[:10]:
                print(f"  {result.subdomain} → {result.predicted_category} ({result.confidence:.3f})")
            
            return {
                'total_tested': len(results),
                'average_confidence': avg_confidence,
                'high_confidence_ratio': high_confidence / len(results),
                'validation_successful': True
            }
        else:
            return {'validation_successful': False, 'error': 'No successful classifications'}

def main():
    """Main training function."""
    print("🤖 NETRA Safe Training Manager")
    print("Crash-resistant training with system monitoring")
    print("=" * 50)
    
    # Initialize training manager with conservative settings
    trainer = SafeTrainingManager(chunk_size=500, max_memory_usage=75)
    
    try:
        # Resume or start training
        training_results = trainer.resume_training()
        
        # Run final validation if training completed
        if training_results.get('completed_chunks', 0) > 0:
            validation_results = trainer.run_final_validation()
            
            print("\n📋 TRAINING SUMMARY")
            print("=" * 30)
            print(f"✅ Chunks Processed: {training_results.get('completed_chunks', 0)}")
            print(f"📊 Samples Trained: {training_results.get('training_samples_processed', 0):,}")
            print(f"🏆 Best Accuracy: {training_results.get('best_accuracy', 0):.3f}")
            print(f"🎯 Final Confidence: {validation_results.get('average_confidence', 0):.3f}")
            print(f"💾 Last Checkpoint: {training_results.get('last_checkpoint', 'None')}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user. Progress has been saved.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💾 Check training_state.json for recovery information.")

if __name__ == "__main__":
    main()
