#!/usr/bin/env python3
"""
NETRA - AI-Powered Subdomain Analysis Tool
=========================================

An intelligent subdomain analysis tool that builds its own knowledge base
from real-world data, reducing dependency on external AI APIs.

Features:
- Self-learning knowledge base from millions of subdomains
- Advanced pattern recognition and feature extraction
- ML-based classification with confidence scoring
- Benchmarking and saturation curve analysis
- Incremental learning and model optimization

Author: Security Research Team
Version: 2.0.0
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import re
import math
import pickle
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import tempfile

# ML Libraries
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    from sklearn.model_selection import train_test_split
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Optional OpenAI for initial training data
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class SubdomainFeatures:
    """Represents extracted features from a subdomain."""
    domain: str
    length: int
    parts_count: int
    has_numbers: bool
    has_hyphens: bool
    has_underscores: bool
    starts_with_number: bool
    keyword_matches: List[str]
    tld: str
    entropy: float
    vowel_consonant_ratio: float
    common_prefix: Optional[str]
    common_suffix: Optional[str]


@dataclass
class ClassificationResult:
    """Result of subdomain classification."""
    subdomain: str
    predicted_category: str
    confidence: float
    alternative_categories: List[Tuple[str, float]]
    features_used: Dict[str, Any]
    prediction_source: str  # 'knowledge_base', 'ml_model', 'heuristic'


class KnowledgeBaseManager:
    """Manages the subdomain knowledge database."""
    
    def __init__(self, db_path: str = "subdomain_knowledge.db"):
        self.db_path = db_path
        self.init_database()
        
        # Category mapping
        self.categories = {
            'admin': 'Administrative / Management Interfaces',
            'api': 'APIs',
            'staging': 'Staging / Development / Testing',
            'auth': 'Authentication / Identity',
            'payment': 'Payment / Transactional',
            'cdn': 'CDN / Storage / Assets',
            'database': 'Database / Data Services',
            'internal': 'Internal Tools / Infrastructure',
            'content': 'Marketing / Content / CMS',
            'mobile': 'Mobile / Partner / Integration',
            'monitoring': 'Monitoring / Logging',
            'security': 'Security Services'
        }
        
        # Load keyword patterns
        self.keyword_patterns = self._load_keyword_patterns()
    
    def init_database(self):
        """Initialize the knowledge base database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Subdomains table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subdomains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subdomain_id INTEGER,
                feature_name TEXT NOT NULL,
                feature_value TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                FOREIGN KEY (subdomain_id) REFERENCES subdomains (id)
            )
        ''')
        
        # Patterns table for keyword matching
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                frequency INTEGER DEFAULT 1,
                accuracy REAL DEFAULT 0.0
            )
        ''')
        
        # Training sessions for tracking learning progress
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                subdomains_processed INTEGER DEFAULT 0,
                accuracy_score REAL DEFAULT 0.0,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                notes TEXT
            )
        ''')
        
        # Performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES training_sessions (id)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON subdomains (domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON subdomains (category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern ON patterns (pattern)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feature_name ON features (feature_name)')
        
        conn.commit()
        conn.close()
    
    def _load_keyword_patterns(self) -> Dict[str, List[str]]:
        """Load and return keyword patterns for each category."""
        return {
            'admin': ['admin', 'dashboard', 'console', 'portal', 'cpanel', 'manage', 'control', 'panel'],
            'api': ['api', 'graphql', 'rest', 'endpoint', 'service', 'microservice', 'webhook'],
            'staging': ['staging', 'dev', 'test', 'qa', 'sandbox', 'preprod', 'beta', 'alpha', 'demo'],
            'auth': ['auth', 'login', 'sso', 'accounts', 'idp', 'oauth', 'saml', 'identity'],
            'payment': ['payments', 'billing', 'checkout', 'invoice', 'pay', 'stripe', 'paypal'],
            'cdn': ['cdn', 'static', 'media', 'uploads', 'files', 'assets', 'images', 'js', 'css'],
            'database': ['db', 'sql', 'mysql', 'postgres', 'mongo', 'redis', 'elastic', 'data'],
            'internal': ['jira', 'jenkins', 'git', 'grafana', 'kibana', 'vpn', 'ci', 'build', 'deploy'],
            'content': ['blog', 'cms', 'wordpress', 'drupal', 'press', 'careers', 'about', 'help'],
            'mobile': ['mobile', 'app', 'android', 'ios', 'partner', 'integration', 'm'],
            'monitoring': ['status', 'monitor', 'metrics', 'logs', 'uptime', 'health', 'ping'],
            'security': ['vpn', 'firewall', 'waf', 'secure', 'ssl', 'cert', 'security']
        }
    
    def add_subdomain(self, domain: str, category: str, confidence: float = 1.0, source: str = 'manual'):
        """Add a subdomain to the knowledge base."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO subdomains (domain, category, confidence, source, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (domain, category, confidence, source))
            
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None
        finally:
            conn.close()
    
    def get_similar_subdomains(self, domain: str, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Get similar subdomains from the knowledge base."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple similarity based on shared keywords
        domain_parts = domain.lower().replace('.', ' ').replace('-', ' ').replace('_', ' ').split()
        
        similar = []
        for part in domain_parts:
            if len(part) > 2:  # Skip very short parts
                cursor.execute('''
                    SELECT domain, category, confidence 
                    FROM subdomains 
                    WHERE domain LIKE ? 
                    ORDER BY confidence DESC 
                    LIMIT ?
                ''', (f'%{part}%', limit))
                
                similar.extend(cursor.fetchall())
        
        conn.close()
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_similar = []
        for domain_sim, category, confidence in similar:
            if domain_sim not in seen and domain_sim != domain:
                seen.add(domain_sim)
                unique_similar.append((domain_sim, category, confidence))
        
        return sorted(unique_similar, key=lambda x: x[2], reverse=True)[:limit]
    
    def update_pattern_accuracy(self, pattern: str, correct: bool):
        """Update pattern accuracy based on classification results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT accuracy, frequency FROM patterns WHERE pattern = ?', (pattern,))
        result = cursor.fetchone()
        
        if result:
            current_accuracy, frequency = result
            # Update accuracy using incremental learning
            new_frequency = frequency + 1
            if correct:
                new_accuracy = (current_accuracy * frequency + 1) / new_frequency
            else:
                new_accuracy = (current_accuracy * frequency) / new_frequency
            
            cursor.execute('''
                UPDATE patterns 
                SET accuracy = ?, frequency = ?
                WHERE pattern = ?
            ''', (new_accuracy, new_frequency, pattern))
        
        conn.commit()
        conn.close()


class AdvancedFeatureExtractor:
    """Advanced feature extraction from subdomains."""
    
    def __init__(self):
        self.common_tlds = ['com', 'org', 'net', 'edu', 'gov', 'mil', 'io', 'co', 'uk', 'de', 'fr']
        self.tech_keywords = ['api', 'admin', 'dev', 'test', 'staging', 'prod', 'beta', 'alpha']
    
    def extract_features(self, domain: str) -> SubdomainFeatures:
        """Extract comprehensive features from a subdomain."""
        domain_lower = domain.lower()
        parts = domain_lower.split('.')
        subdomain_part = parts[0] if len(parts) > 1 else domain_lower
        
        # Basic features
        length = len(subdomain_part)
        parts_count = len(parts)
        has_numbers = bool(re.search(r'\d', subdomain_part))
        has_hyphens = '-' in subdomain_part
        has_underscores = '_' in subdomain_part
        starts_with_number = subdomain_part[0].isdigit() if subdomain_part else False
        
        # TLD extraction
        tld = parts[-1] if len(parts) > 1 else 'unknown'
        
        # Entropy calculation (measure of randomness)
        entropy = self._calculate_entropy(subdomain_part)
        
        # Vowel/consonant ratio
        vowel_consonant_ratio = self._calculate_vowel_consonant_ratio(subdomain_part)
        
        # Keyword matching
        keyword_matches = self._find_keyword_matches(subdomain_part)
        
        # Common prefix/suffix detection
        common_prefix = self._detect_common_prefix(subdomain_part)
        common_suffix = self._detect_common_suffix(subdomain_part)
        
        return SubdomainFeatures(
            domain=domain,
            length=length,
            parts_count=parts_count,
            has_numbers=has_numbers,
            has_hyphens=has_hyphens,
            has_underscores=has_underscores,
            starts_with_number=starts_with_number,
            keyword_matches=keyword_matches,
            tld=tld,
            entropy=entropy,
            vowel_consonant_ratio=vowel_consonant_ratio,
            common_prefix=common_prefix,
            common_suffix=common_suffix
        )
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        char_counts = Counter(text)
        text_length = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / text_length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _calculate_vowel_consonant_ratio(self, text: str) -> float:
        """Calculate vowel to consonant ratio."""
        if not text:
            return 0.0
        
        vowels = sum(1 for char in text.lower() if char in 'aeiou')
        consonants = sum(1 for char in text.lower() if char.isalpha() and char not in 'aeiou')
        
        if consonants == 0:
            return float('inf') if vowels > 0 else 0.0
        
        return vowels / consonants
    
    def _find_keyword_matches(self, text: str) -> List[str]:
        """Find matching keywords in subdomain."""
        matches = []
        text_lower = text.lower()
        
        # Check for exact matches and partial matches
        for keyword in self.tech_keywords:
            if keyword in text_lower:
                matches.append(keyword)
        
        return matches
    
    def _detect_common_prefix(self, text: str) -> Optional[str]:
        """Detect common prefixes."""
        common_prefixes = ['www', 'mail', 'ftp', 'blog', 'shop', 'store', 'news']
        
        for prefix in common_prefixes:
            if text.lower().startswith(prefix):
                return prefix
        
        return None
    
    def _detect_common_suffix(self, text: str) -> Optional[str]:
        """Detect common suffixes."""
        common_suffixes = ['api', 'cdn', 'app', 'web', 'mobile', 'admin']
        
        for suffix in common_suffixes:
            if text.lower().endswith(suffix):
                return suffix
        
        return None


class IntelligentClassifier:
    """ML-based classifier with knowledge base integration."""
    
    def __init__(self, knowledge_base: KnowledgeBaseManager):
        self.kb = knowledge_base
        self.feature_extractor = AdvancedFeatureExtractor()
        self.model = None
        self.vectorizer = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.model_path = "subdomain_classifier.joblib"
        self.vectorizer_path = "subdomain_vectorizer.joblib"
        
    def train_model(self, training_data: List[Tuple[str, str]], test_size: float = 0.2):
        """Train the ML model on subdomain data."""
        if not ML_AVAILABLE:
            print("[-] ML libraries not available. Install scikit-learn, numpy.")
            return False
        
        print(f"[*] Training model on {len(training_data)} samples...")
        
        # Prepare features
        domains = [item[0] for item in training_data]
        categories = [item[1] for item in training_data]
        
        # Extract features
        features = []
        text_features = []
        
        for domain in domains:
            feat = self.feature_extractor.extract_features(domain)
            
            # Numerical features
            feature_vector = [
                feat.length,
                feat.parts_count,
                int(feat.has_numbers),
                int(feat.has_hyphens),
                int(feat.has_underscores),
                int(feat.starts_with_number),
                feat.entropy,
                feat.vowel_consonant_ratio,
                len(feat.keyword_matches)
            ]
            
            features.append(feature_vector)
            text_features.append(' '.join([domain.replace('.', ' '), ' '.join(feat.keyword_matches)]))
        
        # Convert to numpy arrays
        X_numerical = np.array(features)
        
        # Text vectorization
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X_text = self.vectorizer.fit_transform(text_features)
        
        # Combine features
        X_combined = np.hstack([X_numerical, X_text.toarray()])
        
        # Encode labels
        unique_categories = list(set(categories))
        self.label_encoder = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.reverse_label_encoder = {idx: cat for cat, idx in self.label_encoder.items()}
        
        y = np.array([self.label_encoder[cat] for cat in categories])
        
        # Split data - adjust for small datasets
        unique_classes = len(unique_categories)
        min_samples_per_class = len(training_data) // unique_classes
        
        # Adjust test_size if needed for stratification
        if min_samples_per_class < 2 or len(training_data) * test_size < unique_classes:
            # Use simple split without stratification for small datasets
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=min(0.2, max(0.1, unique_classes / len(training_data))), 
                random_state=42
            )
        else:
            # Use stratified split for larger datasets
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=test_size, random_state=42, stratify=y
            )
        
        # Train ensemble model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        print(f"[+] Model trained successfully!")
        print(f"[+] Accuracy: {accuracy:.3f}")
        print(f"[+] Precision: {precision:.3f}")
        print(f"[+] Recall: {recall:.3f}")
        print(f"[+] F1-Score: {f1:.3f}")
        
        # Save model
        self.save_model()
        
        return True
    
    def classify_subdomain(self, domain: str) -> ClassificationResult:
        """Classify a subdomain using the trained model and knowledge base."""
        
        # 1. Check knowledge base first
        similar = self.kb.get_similar_subdomains(domain, limit=5)
        if similar:
            # Use most confident similar subdomain
            best_match = similar[0]
            if best_match[2] > 0.8:  # High confidence threshold
                return ClassificationResult(
                    subdomain=domain,
                    predicted_category=best_match[1],
                    confidence=best_match[2],
                    alternative_categories=[(s[1], s[2]) for s in similar[1:3]],
                    features_used={"similar_domain": best_match[0]},
                    prediction_source="knowledge_base"
                )
        
        # 2. Use ML model if available
        if self.model is not None and ML_AVAILABLE:
            try:
                features = self.feature_extractor.extract_features(domain)
                
                # Prepare features
                feature_vector = [
                    features.length,
                    features.parts_count,
                    int(features.has_numbers),
                    int(features.has_hyphens),
                    int(features.has_underscores),
                    int(features.starts_with_number),
                    features.entropy,
                    features.vowel_consonant_ratio,
                    len(features.keyword_matches)
                ]
                
                text_feature = ' '.join([domain.replace('.', ' '), ' '.join(features.keyword_matches)])
                X_text = self.vectorizer.transform([text_feature])
                X_combined = np.hstack([np.array([feature_vector]), X_text.toarray()])
                
                # Predict
                probabilities = self.model.predict_proba(X_combined)[0]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
                
                # Get alternatives
                sorted_indices = np.argsort(probabilities)[::-1]
                alternatives = [
                    (self.reverse_label_encoder[idx], probabilities[idx])
                    for idx in sorted_indices[1:4]
                ]
                
                return ClassificationResult(
                    subdomain=domain,
                    predicted_category=self.reverse_label_encoder[predicted_class],
                    confidence=confidence,
                    alternative_categories=alternatives,
                    features_used=features.__dict__,
                    prediction_source="ml_model"
                )
            except Exception as e:
                print(f"[-] ML prediction failed: {e}")
        
        # 3. Fallback to heuristic classification
        return self._heuristic_classification(domain)
    
    def _heuristic_classification(self, domain: str) -> ClassificationResult:
        """Fallback heuristic classification."""
        features = self.feature_extractor.extract_features(domain)
        domain_lower = domain.lower()
        
        # Simple keyword-based classification
        for category, keywords in self.kb.keyword_patterns.items():
            for keyword in keywords:
                if keyword in domain_lower:
                    return ClassificationResult(
                        subdomain=domain,
                        predicted_category=self.kb.categories[category],
                        confidence=0.6,  # Medium confidence for heuristics
                        alternative_categories=[],
                        features_used={"matched_keyword": keyword},
                        prediction_source="heuristic"
                    )
        
        # Default classification
        return ClassificationResult(
            subdomain=domain,
            predicted_category="Marketing / Content / CMS",
            confidence=0.3,
            alternative_categories=[],
            features_used=features.__dict__,
            prediction_source="heuristic_default"
        )
    
    def save_model(self):
        """Save the trained model and vectorizer."""
        if self.model and ML_AVAILABLE:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)
            joblib.dump((self.label_encoder, self.reverse_label_encoder), "label_encoders.joblib")
            print(f"[+] Model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """Load a previously trained model."""
        if not ML_AVAILABLE:
            return False
        
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.label_encoder, self.reverse_label_encoder = joblib.load("label_encoders.joblib")
                print(f"[+] Model loaded from {self.model_path}")
                return True
        except Exception as e:
            print(f"[-] Failed to load model: {e}")
        
        return False


class TrainingPipeline:
    """Manages the training pipeline for processing millions of subdomains."""
    
    def __init__(self, knowledge_base: KnowledgeBaseManager, classifier: IntelligentClassifier):
        self.kb = knowledge_base
        self.classifier = classifier
        self.batch_size = 1000
        self.checkpoint_interval = 10000
    
    def process_subdomain_file(self, file_path: str, session_name: str = None):
        """Process a large file of subdomains for training."""
        if not session_name:
            session_name = f"training_{int(time.time())}"
        
        print(f"[*] Starting training session: {session_name}")
        
        # Create training session
        session_id = self._create_training_session(session_name)
        
        processed_count = 0
        batch_data = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    subdomain = line.strip()
                    if subdomain and '.' in subdomain:
                        
                        # Get initial classification (using current model or heuristics)
                        result = self.classifier.classify_subdomain(subdomain)
                        
                        # Add to batch
                        batch_data.append((subdomain, result.predicted_category))
                        processed_count += 1
                        
                        # Process batch
                        if len(batch_data) >= self.batch_size:
                            self._process_batch(batch_data, session_id)
                            batch_data = []
                        
                        # Checkpoint
                        if processed_count % self.checkpoint_interval == 0:
                            self._create_checkpoint(session_id, processed_count)
                            print(f"[*] Processed {processed_count} subdomains...")
                
                # Process remaining batch
                if batch_data:
                    self._process_batch(batch_data, session_id)
                
        except Exception as e:
            print(f"[-] Error processing file: {e}")
            return False
        
        # Complete session
        self._complete_training_session(session_id, processed_count)
        print(f"[+] Training session completed: {processed_count} subdomains processed")
        
        return True
    
    def _create_training_session(self, session_name: str) -> int:
        """Create a new training session record."""
        conn = sqlite3.connect(self.kb.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_sessions (session_name, started_at)
            VALUES (?, CURRENT_TIMESTAMP)
        ''', (session_name,))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def _process_batch(self, batch_data: List[Tuple[str, str]], session_id: int):
        """Process a batch of subdomain data."""
        for subdomain, category in batch_data:
            # Add to knowledge base with lower confidence for auto-classified
            self.kb.add_subdomain(subdomain, category, confidence=0.7, source='auto_training')
    
    def _create_checkpoint(self, session_id: int, processed_count: int):
        """Create a training checkpoint."""
        conn = sqlite3.connect(self.kb.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE training_sessions 
            SET subdomains_processed = ?
            WHERE id = ?
        ''', (processed_count, session_id))
        
        conn.commit()
        conn.close()
    
    def _complete_training_session(self, session_id: int, processed_count: int):
        """Complete a training session."""
        conn = sqlite3.connect(self.kb.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE training_sessions 
            SET subdomains_processed = ?, completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (processed_count, session_id))
        
        conn.commit()
        conn.close()


class BenchmarkManager:
    """Manages benchmarking and performance evaluation."""
    
    def __init__(self, classifier: IntelligentClassifier, knowledge_base: KnowledgeBaseManager):
        self.classifier = classifier
        self.kb = knowledge_base
    
    def run_benchmark(self, test_file: str, ground_truth_file: str = None) -> Dict[str, float]:
        """Run benchmark against known test data."""
        print("[*] Running benchmark evaluation...")
        
        # Load test subdomains
        test_subdomains = []
        with open(test_file, 'r') as f:
            for line in f:
                subdomain = line.strip()
                if subdomain:
                    test_subdomains.append(subdomain)
        
        # Load ground truth if available
        ground_truth = {}
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ground_truth[parts[0]] = parts[1]
        
        # Run classifications
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        confidence_scores = []
        prediction_sources = Counter()
        
        for subdomain in test_subdomains[:1000]:  # Limit for demo
            result = self.classifier.classify_subdomain(subdomain)
            results.append(result)
            
            confidence_scores.append(result.confidence)
            prediction_sources[result.prediction_source] += 1
            
            # Check accuracy if ground truth available
            if subdomain in ground_truth:
                if result.predicted_category == ground_truth[subdomain]:
                    correct_predictions += 1
                total_predictions += 1
        
        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        metrics = {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'total_tested': len(results),
            'knowledge_base_usage': prediction_sources['knowledge_base'] / len(results),
            'ml_model_usage': prediction_sources['ml_model'] / len(results),
            'heuristic_usage': prediction_sources['heuristic'] / len(results)
        }
        
        print(f"[+] Benchmark Results:")
        print(f"    Accuracy: {accuracy:.3f}")
        print(f"    Average Confidence: {avg_confidence:.3f}")
        print(f"    Total Tested: {len(results)}")
        print(f"    Knowledge Base Usage: {metrics['knowledge_base_usage']:.3f}")
        print(f"    ML Model Usage: {metrics['ml_model_usage']:.3f}")
        print(f"    Heuristic Usage: {metrics['heuristic_usage']:.3f}")
        
        return metrics
    
    def analyze_learning_curve(self, training_sizes: List[int], test_file: str) -> List[Dict]:
        """Analyze learning curve to detect saturation."""
        curve_data = []
        
        for size in training_sizes:
            print(f"[*] Testing with {size} training samples...")
            
            # Get training data from knowledge base
            conn = sqlite3.connect(self.kb.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT domain, category FROM subdomains 
                ORDER BY RANDOM() LIMIT ?
            ''', (size,))
            training_data = cursor.fetchall()
            conn.close()
            
            if len(training_data) < size:
                print(f"[-] Not enough training data: {len(training_data)} < {size}")
                continue
            
            # Retrain model
            self.classifier.train_model(training_data)
            
            # Run benchmark
            metrics = self.run_benchmark(test_file)
            metrics['training_size'] = size
            curve_data.append(metrics)
        
        return curve_data


# Enhanced main class integrating all components
class SubdomainAIEnhanced:
    """Enhanced subdomain AI tool with self-learning capabilities."""
    
    def __init__(self):
        self.kb = KnowledgeBaseManager()
        self.classifier = IntelligentClassifier(self.kb)
        self.training_pipeline = TrainingPipeline(self.kb, self.classifier)
        self.benchmark_manager = BenchmarkManager(self.classifier, self.kb)
        
        # Try to load existing model
        self.classifier.load_model()
    
    def train_on_file(self, file_path: str, session_name: str = None):
        """Train the system on a large subdomain file."""
        return self.training_pipeline.process_subdomain_file(file_path, session_name)
    
    def benchmark(self, test_file: str, ground_truth_file: str = None):
        """Run benchmarks on the system."""
        return self.benchmark_manager.run_benchmark(test_file, ground_truth_file)
    
    def analyze_learning_curve(self, test_file: str):
        """Analyze learning curve saturation."""
        sizes = [1000, 5000, 10000, 50000, 100000, 500000]
        return self.benchmark_manager.analyze_learning_curve(sizes, test_file)
    
    def classify_subdomains(self, subdomains: List[str]) -> List[ClassificationResult]:
        """Classify a list of subdomains."""
        results = []
        for subdomain in subdomains:
            result = self.classifier.classify_subdomain(subdomain)
            results.append(result)
        return results


def main():
    """Enhanced main function."""
    parser = argparse.ArgumentParser(
        description="NETRA - AI-Powered Subdomain Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--train', metavar='FILE', help='Train on subdomain file')
    parser.add_argument('--analyze', metavar='FILE', help='Analyze subdomains from file')
    parser.add_argument('--benchmark', metavar='FILE', help='Run benchmark on test file')
    parser.add_argument('--learning-curve', metavar='FILE', help='Analyze learning curve saturation')
    parser.add_argument('--session-name', help='Name for training session')
    
    args = parser.parse_args()
    
    if not any([args.train, args.analyze, args.benchmark, args.learning_curve]):
        parser.print_help()
        sys.exit(1)
    
    print("🧠 Subdomain AI Enhanced - Self-Learning Tool")
    print("=" * 50)
    
    ai = SubdomainAIEnhanced()
    
    if args.train:
        print(f"[*] Training on: {args.train}")
        ai.train_on_file(args.train, args.session_name)
    
    if args.analyze:
        print(f"[*] Analyzing: {args.analyze}")
        # Load subdomains
        with open(args.analyze, 'r') as f:
            subdomains = [line.strip() for line in f if line.strip()][:100]  # Limit for demo
        
        results = ai.classify_subdomains(subdomains)
        
        # Output results
        for result in results:
            print(f"{result.subdomain} -> {result.predicted_category} "
                  f"(confidence: {result.confidence:.2f}, source: {result.prediction_source})")
    
    if args.benchmark:
        print(f"[*] Benchmarking with: {args.benchmark}")
        ai.benchmark(args.benchmark)
    
    if args.learning_curve:
        print(f"[*] Analyzing learning curve with: {args.learning_curve}")
        curve_data = ai.analyze_learning_curve(args.learning_curve)
        
        # Save results
        with open('learning_curve_results.json', 'w') as f:
            json.dump(curve_data, f, indent=2)
        print("[+] Learning curve results saved to learning_curve_results.json")


if __name__ == "__main__":
    main()
