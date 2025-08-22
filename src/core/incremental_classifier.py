#!/usr/bin/env python3
"""
Incremental Learning Classifier for NETRA
========================================

Proper incremental learning that accumulates knowledge across training sessions
instead of overwriting the model each time.
"""

import os
import json
import numpy as np
import joblib
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import sqlite3

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML libraries not available. Please install: pip install scikit-learn numpy")

from subdomain_ai_enhanced import AdvancedFeatureExtractor, ClassificationResult

class IncrementalLearningClassifier:
    """Classifier that properly accumulates knowledge across training sessions."""
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.feature_extractor = AdvancedFeatureExtractor()
        
        # Model components
        self.model = None
        self.vectorizer = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        
        # Incremental learning storage
        self.accumulated_data = {
            'domains': [],
            'categories': [],
            'features': [],
            'text_features': []
        }
        
        # File paths
        self.model_path = "incremental_model.joblib"
        self.vectorizer_path = "incremental_vectorizer.joblib"
        self.encoders_path = "incremental_encoders.joblib"
        self.data_path = "accumulated_training_data.joblib"
        self.stats_path = "incremental_training_stats.json"
        
        # Training statistics
        self.training_stats = {
            'total_samples_trained': 0,
            'training_sessions': 0,
            'categories_learned': set(),
            'accuracy_history': [],
            'last_updated': None,
            'model_version': '1.0'
        }
        
        # Load existing model and data if available
        self.load_accumulated_data()
        self.load_training_stats()
        self.load_model()
    
    def add_training_chunk(self, chunk_data: List[Tuple[str, str]]) -> Dict:
        """Add a chunk of training data to accumulated dataset."""
        print(f"📥 Adding {len(chunk_data)} samples to accumulated training data...")
        
        # Process new chunk
        new_features = []
        new_text_features = []
        
        for domain, category in chunk_data:
            if domain not in self.accumulated_data['domains']:  # Avoid duplicates
                # Extract features
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
                
                text_feature = ' '.join([domain.replace('.', ' '), ' '.join(feat.keyword_matches)])
                
                # Add to accumulated data
                self.accumulated_data['domains'].append(domain)
                self.accumulated_data['categories'].append(category)
                self.accumulated_data['features'].append(feature_vector)
                self.accumulated_data['text_features'].append(text_feature)
                
                new_features.append(feature_vector)
                new_text_features.append(text_feature)
                
                # Track categories
                self.training_stats['categories_learned'].add(category)
        
        # Update stats
        self.training_stats['total_samples_trained'] = len(self.accumulated_data['domains'])
        
        print(f"✅ Accumulated dataset now has {self.training_stats['total_samples_trained']:,} unique samples")
        print(f"📊 Categories learned: {len(self.training_stats['categories_learned'])}")
        
        # Save accumulated data
        self.save_accumulated_data()
        
        return {
            'new_samples_added': len(new_features),
            'total_accumulated': self.training_stats['total_samples_trained'],
            'unique_categories': len(self.training_stats['categories_learned'])
        }
    
    def train_incremental_model(self, validation_size: float = 0.15) -> Dict:
        """Train model on ALL accumulated data."""
        if not ML_AVAILABLE:
            return {'success': False, 'error': 'ML libraries not available'} 
        
        if not self.accumulated_data['domains']:
            return {'success': False, 'error': 'No training data available'}
        
        print(f"🧠 Training incremental model on {len(self.accumulated_data['domains']):,} accumulated samples...")
        
        # Prepare all accumulated data
        X_numerical = np.array(self.accumulated_data['features'])
        
        # Handle vectorizer - either update or create new
        if self.vectorizer is None:
            print("[*] Creating new TF-IDF vectorizer...")
            self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 3), min_df=2)
            X_text = self.vectorizer.fit_transform(self.accumulated_data['text_features'])
        else:
            print("[*] Using existing TF-IDF vectorizer...")
            try:
                # Try to transform with existing vectorizer
                X_text = self.vectorizer.transform(self.accumulated_data['text_features'])
            except Exception as e:
                print(f"[!] Vectorizer incompatible, creating new one: {e}")
                self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 3), min_df=2)
                X_text = self.vectorizer.fit_transform(self.accumulated_data['text_features'])
        
        # Combine features
        X_combined = np.hstack([X_numerical, X_text.toarray()])
        
        # Update label encoders with all categories
        all_categories = list(self.training_stats['categories_learned'])
        self.label_encoder = {cat: idx for idx, cat in enumerate(all_categories)}
        self.reverse_label_encoder = {idx: cat for cat, idx in self.label_encoder.items()}
        
        # Encode labels
        y = np.array([self.label_encoder[cat] for cat in self.accumulated_data['categories']])
        
        # Split for validation
        if len(set(y)) > 1 and len(y) > 10:  # Need multiple classes and enough samples
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_combined, y, test_size=validation_size, random_state=42, stratify=y
                )
            except ValueError:
                # Fallback to simple split if stratify fails
                X_train, X_val, y_train, y_val = train_test_split(
                    X_combined, y, test_size=validation_size, random_state=42
                )
        else:
            # Use all data for training if too small for validation split
            X_train, X_val, y_train, y_val = X_combined, X_combined, y, y
        
        # Create or update model - use more sophisticated model for large datasets
        n_samples = len(self.accumulated_data['domains'])
        if n_samples > 10000:
            # Use more complex model for larger datasets
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
        elif n_samples > 1000:
            # Medium complexity for medium datasets
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            # Simpler model for smaller datasets
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        # Train model
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Validate model
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_val, average='weighted')
        
        # Update training stats
        self.training_stats['training_sessions'] += 1
        self.training_stats['accuracy_history'].append({
            'session': self.training_stats['training_sessions'],
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'samples_used': n_samples,
            'categories_count': len(all_categories),
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        })
        self.training_stats['last_updated'] = datetime.now().isoformat()
        
        print(f"✅ Incremental training completed!")
        print(f"📊 Training Accuracy: {train_accuracy:.3f}")
        print(f"📊 Validation Accuracy: {val_accuracy:.3f}")
        print(f"📊 Precision: {precision:.3f}")
        print(f"📊 Recall: {recall:.3f}")
        print(f"📊 F1-Score: {f1:.3f}")
        print(f"⏱️ Training Time: {training_time:.1f}s")
        
        # Save everything
        self.save_model()
        self.save_training_stats()
        
        return {
            'success': True,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'total_samples': n_samples,
            'categories_count': len(all_categories)
        }
    
    def classify_subdomain(self, domain: str) -> ClassificationResult:
        """Classify subdomain using incremental model."""
        # 1. Check knowledge base first
        similar = self.kb.get_similar_subdomains(domain, limit=5)
        if similar:
            best_match = similar[0]
            if best_match[2] > 0.8:
                return ClassificationResult(
                    subdomain=domain,
                    predicted_category=best_match[1],
                    confidence=best_match[2],
                    alternative_categories=[(s[1], s[2]) for s in similar[1:3]],
                    features_used={"similar_domain": best_match[0]},
                    prediction_source="knowledge_base"
                )
        
        # 2. Use incremental ML model
        if self.model is not None and self.vectorizer is not None and ML_AVAILABLE:
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
                
                # Predict with confidence
                probabilities = self.model.predict_proba(X_combined)[0]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
                
                # Get alternatives
                sorted_indices = np.argsort(probabilities)[::-1]
                alternatives = [
                    (self.reverse_label_encoder[idx], probabilities[idx])
                    for idx in sorted_indices[1:4] if idx in self.reverse_label_encoder
                ]
                
                return ClassificationResult(
                    subdomain=domain,
                    predicted_category=self.reverse_label_encoder[predicted_class],
                    confidence=confidence,
                    alternative_categories=alternatives,
                    features_used=features.__dict__,
                    prediction_source="incremental_ml_model"
                )
                
            except Exception as e:
                print(f"[-] Incremental ML prediction failed: {e}")
        
        # 3. Fallback to heuristic
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
                        confidence=0.6,
                        alternative_categories=[],
                        features_used={"matched_keyword": keyword},
                        prediction_source="heuristic"
                    )
        
        return ClassificationResult(
            subdomain=domain,
            predicted_category="Marketing / Content / CMS",
            confidence=0.3,
            alternative_categories=[],
            features_used=features.__dict__,
            prediction_source="heuristic_default"
        )
    
    def save_accumulated_data(self):
        """Save accumulated training data."""
        # Convert set to list for JSON serialization
        data_to_save = dict(self.accumulated_data)
        
        joblib.dump(data_to_save, self.data_path)
    
    def load_accumulated_data(self):
        """Load previously accumulated training data."""
        if os.path.exists(self.data_path):
            try:
                self.accumulated_data = joblib.load(self.data_path)
                print(f"[+] Loaded {len(self.accumulated_data['domains']):,} accumulated samples")
                
                # Rebuild categories set
                for category in self.accumulated_data['categories']:
                    self.training_stats['categories_learned'].add(category)
                    
                return True
            except Exception as e:
                print(f"[-] Could not load accumulated data: {e}")
                return False
        return False
    
    def save_model(self):
        """Save the incremental model and components."""
        if self.model and ML_AVAILABLE:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)
            joblib.dump((self.label_encoder, self.reverse_label_encoder), self.encoders_path)
            print(f"[+] Incremental model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """Load previously trained incremental model."""
        if not ML_AVAILABLE:
            return False
        
        try:
            if all(os.path.exists(path) for path in [self.model_path, self.vectorizer_path, self.encoders_path]):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.label_encoder, self.reverse_label_encoder = joblib.load(self.encoders_path)
                print(f"[+] Incremental model loaded from {self.model_path}")
                return True
        except Exception as e:
            print(f"[-] Could not load incremental model: {e}")
        
        return False
    
    def save_training_stats(self):
        """Save training statistics."""
        # Convert sets to lists for JSON serialization
        stats_to_save = dict(self.training_stats)
        stats_to_save['categories_learned'] = list(self.training_stats['categories_learned'])
        
        with open(self.stats_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
    
    def load_training_stats(self):
        """Load training statistics."""
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path, 'r') as f:
                    loaded_stats = json.load(f)
                
                # Convert lists back to sets
                loaded_stats['categories_learned'] = set(loaded_stats['categories_learned'])
                self.training_stats.update(loaded_stats)
                
                return True
            except Exception as e:
                print(f"[-] Could not load training stats: {e}")
        return False
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary."""
        return {
            'total_samples_trained': self.training_stats['total_samples_trained'],
            'training_sessions': self.training_stats['training_sessions'],
            'categories_count': len(self.training_stats['categories_learned']),
            'categories_learned': list(self.training_stats['categories_learned']),
            'accuracy_progression': [
                {'session': session['session'], 'accuracy': session['val_accuracy']}
                for session in self.training_stats['accuracy_history']
            ],
            'last_updated': self.training_stats['last_updated'],
            'model_available': self.model is not None
        }
