#!/usr/bin/env python3
"""
Massive Dataset Testing Framework
===============================

Designed for processing millions to billions of subdomains efficiently
with memory optimization, progress tracking, and performance monitoring.
"""

import os
import time
import json
import sqlite3
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Generator, Optional
from dataclasses import dataclass
from pathlib import Path
import psutil
import gc
from enhanced_classifier import EnhancedSubdomainClassifier, EnhancedClassificationResult

@dataclass
class ProcessingStats:
    """Statistics for massive dataset processing."""
    total_subdomains: int = 0
    processed_subdomains: int = 0
    processing_rate: float = 0.0  # subdomains per second
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    estimated_time_remaining: float = 0.0
    errors_encountered: int = 0
    start_time: float = 0.0

class MassiveDatasetProcessor:
    """Processor for handling massive subdomain datasets efficiently."""
    
    def __init__(self, 
                 batch_size: int = 1000,
                 max_workers: int = None,
                 memory_limit_gb: int = 8,
                 checkpoint_interval: int = 10000):
        """
        Initialize massive dataset processor.
        
        Args:
            batch_size: Number of subdomains to process in each batch
            max_workers: Maximum number of worker processes (None = auto)
            memory_limit_gb: Maximum memory usage in GB before triggering cleanup
            checkpoint_interval: Save progress every N subdomains
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Cap at 8 for balance
        self.memory_limit_gb = memory_limit_gb
        self.checkpoint_interval = checkpoint_interval
        
        # Processing state
        self.stats = ProcessingStats()
        self.checkpoint_file = "processing_checkpoint.json"
        self.results_db = "massive_results.db"
        
        # Initialize results database
        self._init_results_db()
    
    def _init_results_db(self):
        """Initialize SQLite database for storing results efficiently."""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        # Create optimized table for massive datasets
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subdomain_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subdomain TEXT NOT NULL,
                primary_category TEXT NOT NULL,
                primary_confidence REAL NOT NULL,
                uncertainty_level TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                multi_category_possible BOOLEAN NOT NULL,
                prediction_source TEXT NOT NULL,
                processing_time_ms REAL NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_subdomain ON subdomain_results (subdomain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON subdomain_results (primary_category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk ON subdomain_results (risk_level)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_uncertainty ON subdomain_results (uncertainty_level)')
        
        # Create summary statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                total_processed INTEGER DEFAULT 0,
                processing_rate REAL DEFAULT 0.0,
                accuracy_estimate REAL DEFAULT 0.0,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_subdomains_generator(self, file_path: str, skip_lines: int = 0) -> Generator[str, None, None]:
        """
        Memory-efficient generator for loading subdomains from large files.
        
        Args:
            file_path: Path to subdomain file
            skip_lines: Number of lines to skip (for resuming)
            
        Yields:
            Individual subdomains
        """
        print(f"[*] Loading subdomains from {file_path} (skipping first {skip_lines} lines)...")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip lines for resuming
            for _ in range(skip_lines):
                next(f, None)
            
            for line_num, line in enumerate(f, skip_lines + 1):
                subdomain = line.strip()
                if subdomain and '.' in subdomain and len(subdomain) < 253:  # Basic validation
                    yield subdomain
                
                # Periodic memory check
                if line_num % 100000 == 0:
                    self._check_memory_usage()
    
    def _check_memory_usage(self):
        """Check memory usage and trigger cleanup if needed."""
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
        
        if memory_usage > self.memory_limit_gb:
            print(f"[!] Memory usage {memory_usage:.2f}GB exceeded limit {self.memory_limit_gb}GB - triggering cleanup")
            gc.collect()  # Force garbage collection
    
    def process_batch_worker(self, subdomains_batch: List[str]) -> List[Dict]:
        """
        Worker function for processing a batch of subdomains.
        This runs in a separate process to avoid GIL limitations.
        """
        try:
            # Initialize classifier in worker process
            classifier = EnhancedSubdomainClassifier()
            batch_results = []
            
            for subdomain in subdomains_batch:
                start_time = time.time()
                
                try:
                    result = classifier.classify_with_uncertainty(subdomain)
                    processing_time = (time.time() - start_time) * 1000  # milliseconds
                    
                    # Convert to dictionary for serialization
                    result_dict = {
                        'subdomain': result.subdomain,
                        'primary_category': result.primary_category,
                        'primary_confidence': result.primary_confidence,
                        'uncertainty_level': result.uncertainty_level,
                        'risk_level': classifier._get_risk_level(result.primary_category),
                        'multi_category_possible': result.multi_category_possible,
                        'prediction_source': result.prediction_source,
                        'processing_time_ms': processing_time
                    }
                    
                    batch_results.append(result_dict)
                    
                except Exception as e:
                    print(f"[!] Error processing {subdomain}: {e}")
                    continue
            
            return batch_results
            
        except Exception as e:
            print(f"[!] Batch processing error: {e}")
            return []
    
    def save_batch_results(self, results: List[Dict]):
        """Save batch results to database efficiently."""
        if not results:
            return
        
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        # Use executemany for efficient bulk insert
        cursor.executemany('''
            INSERT INTO subdomain_results 
            (subdomain, primary_category, primary_confidence, uncertainty_level, 
             risk_level, multi_category_possible, prediction_source, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', [
            (r['subdomain'], r['primary_category'], r['primary_confidence'],
             r['uncertainty_level'], r['risk_level'], r['multi_category_possible'],
             r['prediction_source'], r['processing_time_ms'])
            for r in results
        ])
        
        conn.commit()
        conn.close()
    
    def save_checkpoint(self, processed_count: int, session_name: str):
        """Save processing checkpoint for recovery."""
        checkpoint_data = {
            'session_name': session_name,
            'processed_count': processed_count,
            'timestamp': time.time(),
            'stats': {
                'processing_rate': self.stats.processing_rate,
                'memory_usage_mb': self.stats.memory_usage_mb,
                'cpu_usage_percent': self.stats.cpu_usage_percent
            }
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load processing checkpoint for recovery."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Error loading checkpoint: {e}")
        return None
    
    def update_stats(self, processed_count: int, start_time: float):
        """Update processing statistics."""
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        self.stats.processed_subdomains = processed_count
        self.stats.processing_rate = processed_count / elapsed_time if elapsed_time > 0 else 0
        self.stats.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.stats.cpu_usage_percent = psutil.cpu_percent()
        
        # Estimate remaining time
        if self.stats.processing_rate > 0:
            remaining_subdomains = self.stats.total_subdomains - processed_count
            self.stats.estimated_time_remaining = remaining_subdomains / self.stats.processing_rate
        else:
            self.stats.estimated_time_remaining = 0
    
    def print_progress(self, processed_count: int):
        """Print processing progress with performance metrics."""
        progress_pct = (processed_count / self.stats.total_subdomains) * 100 if self.stats.total_subdomains > 0 else 0
        
        print(f"\n[*] Progress: {processed_count:,}/{self.stats.total_subdomains:,} ({progress_pct:.1f}%)")
        print(f"[*] Rate: {self.stats.processing_rate:.1f} subdomains/sec")
        print(f"[*] Memory: {self.stats.memory_usage_mb:.1f} MB")
        print(f"[*] CPU: {self.stats.cpu_usage_percent:.1f}%")
        
        if self.stats.estimated_time_remaining > 0:
            hours = int(self.stats.estimated_time_remaining // 3600)
            minutes = int((self.stats.estimated_time_remaining % 3600) // 60)
            print(f"[*] ETA: {hours:02d}:{minutes:02d}")
    
    def process_massive_dataset(self, 
                              file_path: str, 
                              session_name: str = None,
                              resume: bool = False) -> Dict:
        """
        Process massive dataset with parallel processing and checkpointing.
        
        Args:
            file_path: Path to subdomain file
            session_name: Name for this processing session
            resume: Whether to resume from checkpoint
            
        Returns:
            Processing summary statistics
        """
        if not session_name:
            session_name = f"massive_processing_{int(time.time())}"
        
        print(f"🚀 Massive Dataset Processing Started")
        print(f"Session: {session_name}")
        print(f"File: {file_path}")
        print(f"Workers: {self.max_workers}")
        print(f"Batch Size: {self.batch_size}")
        print("=" * 50)
        
        # Check for resume
        skip_lines = 0
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                skip_lines = checkpoint['processed_count']
                print(f"[*] Resuming from checkpoint: {skip_lines:,} subdomains already processed")
        
        # Count total lines for progress tracking
        print("[*] Counting total subdomains...")
        with open(file_path, 'r') as f:
            self.stats.total_subdomains = sum(1 for line in f if line.strip()) - skip_lines
        
        print(f"[*] Total subdomains to process: {self.stats.total_subdomains:,}")
        
        # Start processing
        start_time = time.time()
        self.stats.start_time = start_time
        processed_count = 0
        
        # Create session record
        self._create_processing_session(session_name)
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                
                # Process in batches
                subdomain_generator = self.load_subdomains_generator(file_path, skip_lines)
                batch = []
                futures = {}
                
                for subdomain in subdomain_generator:
                    batch.append(subdomain)
                    
                    # Process batch when full
                    if len(batch) >= self.batch_size:
                        future = executor.submit(self.process_batch_worker, batch.copy())
                        futures[future] = len(batch)
                        batch.clear()
                        
                        # Check for completed futures
                        if len(futures) >= self.max_workers * 2:  # Limit queue size
                            self._process_completed_futures(futures, processed_count, start_time, session_name)
                
                # Process final batch
                if batch:
                    future = executor.submit(self.process_batch_worker, batch)
                    futures[future] = len(batch)
                
                # Process all remaining futures
                while futures:
                    processed_count = self._process_completed_futures(futures, processed_count, start_time, session_name)
            
            # Final statistics
            total_time = time.time() - start_time
            self._complete_processing_session(session_name, processed_count, total_time)
            
            print(f"\n✅ Processing Complete!")
            print(f"Total Processed: {processed_count:,}")
            print(f"Total Time: {total_time/3600:.2f} hours")
            print(f"Average Rate: {processed_count/total_time:.1f} subdomains/sec")
            
            return {
                'session_name': session_name,
                'total_processed': processed_count,
                'total_time': total_time,
                'average_rate': processed_count / total_time,
                'database_file': self.results_db
            }
            
        except KeyboardInterrupt:
            print(f"\n[!] Processing interrupted by user")
            self.save_checkpoint(processed_count, session_name)
            print(f"[+] Checkpoint saved. Resume with --resume flag")
            return {'interrupted': True, 'processed': processed_count}
        
        except Exception as e:
            print(f"\n[!] Processing error: {e}")
            self.save_checkpoint(processed_count, session_name)
            return {'error': str(e), 'processed': processed_count}
    
    def _process_completed_futures(self, futures: Dict, processed_count: int, start_time: float, session_name: str) -> int:
        """Process completed futures and update statistics."""
        completed_futures = []
        
        for future in as_completed(futures.keys()):
            try:
                results = future.result(timeout=30)  # 30 second timeout
                batch_size = futures[future]
                
                # Save results to database
                self.save_batch_results(results)
                
                processed_count += batch_size
                completed_futures.append(future)
                
                # Update statistics and print progress
                self.update_stats(processed_count, start_time)
                
                if processed_count % (self.batch_size * 10) == 0:  # Every 10 batches
                    self.print_progress(processed_count)
                
                # Save checkpoint
                if processed_count % self.checkpoint_interval == 0:
                    self.save_checkpoint(processed_count, session_name)
                
            except Exception as e:
                print(f"[!] Future processing error: {e}")
                completed_futures.append(future)
        
        # Remove completed futures
        for future in completed_futures:
            futures.pop(future, None)
        
        return processed_count
    
    def _create_processing_session(self, session_name: str) -> int:
        """Create a processing session record."""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO processing_sessions (session_name, total_processed, started_at)
            VALUES (?, 0, CURRENT_TIMESTAMP)
        ''', (session_name,))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def _complete_processing_session(self, session_name: str, total_processed: int, total_time: float):
        """Complete a processing session record."""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        processing_rate = total_processed / total_time if total_time > 0 else 0
        
        cursor.execute('''
            UPDATE processing_sessions 
            SET total_processed = ?, processing_rate = ?, completed_at = CURRENT_TIMESTAMP
            WHERE session_name = ?
        ''', (total_processed, processing_rate, session_name))
        
        conn.commit()
        conn.close()
    
    def generate_summary_report(self, session_name: str = None) -> Dict:
        """Generate comprehensive summary report from processed data."""
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        # Session filter
        session_filter = f"WHERE session_name = '{session_name}'" if session_name else ""
        
        # Basic statistics
        cursor.execute("SELECT COUNT(*) FROM subdomain_results")
        total_processed = cursor.fetchone()[0]
        
        # Category distribution
        cursor.execute('''
            SELECT primary_category, COUNT(*) as count, AVG(primary_confidence) as avg_confidence
            FROM subdomain_results
            GROUP BY primary_category
            ORDER BY count DESC
        ''')
        category_stats = cursor.fetchall()
        
        # Risk level distribution
        cursor.execute('''
            SELECT risk_level, COUNT(*) as count
            FROM subdomain_results
            GROUP BY risk_level
            ORDER BY count DESC
        ''')
        risk_stats = cursor.fetchall()
        
        # Uncertainty distribution
        cursor.execute('''
            SELECT uncertainty_level, COUNT(*) as count
            FROM subdomain_results
            GROUP BY uncertainty_level
        ''')
        uncertainty_stats = cursor.fetchall()
        
        # Performance statistics
        cursor.execute('''
            SELECT 
                AVG(processing_time_ms) as avg_processing_time,
                MIN(processing_time_ms) as min_processing_time,
                MAX(processing_time_ms) as max_processing_time,
                AVG(primary_confidence) as avg_confidence
            FROM subdomain_results
        ''')
        perf_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_processed': total_processed,
            'category_distribution': category_stats,
            'risk_distribution': risk_stats,
            'uncertainty_distribution': uncertainty_stats,
            'performance_stats': {
                'avg_processing_time_ms': perf_stats[0],
                'min_processing_time_ms': perf_stats[1],
                'max_processing_time_ms': perf_stats[2],
                'avg_confidence': perf_stats[3]
            }
        }

def main():
    """Main function for massive dataset processing."""
    parser = argparse.ArgumentParser(description="Massive Subdomain Dataset Processor")
    parser.add_argument('--file', required=True, help='Path to subdomain file')
    parser.add_argument('--session', help='Processing session name')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--workers', type=int, help='Number of worker processes')
    parser.add_argument('--memory-limit', type=int, default=8, help='Memory limit in GB')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--report', action='store_true', help='Generate summary report')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = MassiveDatasetProcessor(
        batch_size=args.batch_size,
        max_workers=args.workers,
        memory_limit_gb=args.memory_limit
    )
    
    if args.report:
        # Generate report
        print("📊 Generating Summary Report...")
        report = processor.generate_summary_report(args.session)
        
        print(f"\nTotal Processed: {report['total_processed']:,}")
        print(f"Average Confidence: {report['performance_stats']['avg_confidence']:.3f}")
        print(f"Average Processing Time: {report['performance_stats']['avg_processing_time_ms']:.2f}ms")
        
        print(f"\nTop Categories:")
        for category, count, avg_conf in report['category_distribution'][:10]:
            print(f"  {category}: {count:,} ({avg_conf:.3f} avg confidence)")
        
        print(f"\nRisk Distribution:")
        for risk, count in report['risk_distribution']:
            print(f"  {risk}: {count:,}")
        
    else:
        # Process dataset
        result = processor.process_massive_dataset(
            file_path=args.file,
            session_name=args.session,
            resume=args.resume
        )
        
        if not result.get('interrupted') and not result.get('error'):
            print(f"\n🎉 Processing completed successfully!")
            print(f"Results stored in: {result['database_file']}")

if __name__ == "__main__":
    main()
