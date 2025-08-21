# 🔍 NETRA - AI-Powered Subdomain Analysis Tool

An intelligent subdomain analysis tool that builds its own knowledge base from real-world data, reducing dependency on external AI APIs while providing advanced ML-powered analysis with confidence scoring.

## 🚀 Features

### 🧠 Self-Learning Intelligence
- **Knowledge base construction** from millions of real subdomains
- **Pattern recognition** and feature extraction algorithms
- **ML-based classification** with confidence scoring
- **Incremental learning** and model optimization
- **Honest uncertainty reporting** when confidence is low

### 📊 Advanced Analysis
- **Multi-dimensional feature extraction** (length, patterns, TLD analysis)
- **Confidence-based predictions** with uncertainty quantification
- **Benchmarking and saturation curve analysis**
- **Model performance metrics** and validation
- **Comprehensive testing framework** with 10K+ samples

### Categories
The tool classifies subdomains into these security-relevant categories:

- 🔧 **Administrative / Management Interfaces**
- 🔌 **APIs**
- 🚧 **Staging / Development / Testing**
- 🔐 **Authentication / Identity**
- 💳 **Payment / Transactional**
- 📦 **CDN / Storage / Assets**
- 🗄️ **Database / Data Services**
- ⚙️ **Internal Tools / Infrastructure**
- 📝 **Marketing / Content / CMS**
- 📱 **Mobile / Partner / Integration**
- 📊 **Monitoring / Logging**
- 🛡️ **Security Services**

## 📋 Prerequisites

### Python Dependencies
```bash
# Basic dependencies
pip install -r requirements.txt

# Complete ML dependencies (recommended)
pip install -r requirements_complete.txt

# Enhanced ML features
pip install -r requirements_enhanced.txt
```

### OpenAI API Key (optional)
Optional for enhanced analysis (NETRA works great without it):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/trinity999/Netra.git
cd Netra
```

2. Install dependencies:
```bash
pip install -r requirements_complete.txt
```

3. Initialize knowledge base (optional):
```bash
python seed_knowledge_base.py
```

## 💻 Usage

### 🚀 Main NETRA Tool

#### Basic Analysis
```bash
python netra.py sample_subdomains.txt
```

#### With Confidence Threshold
```bash
python netra.py subdomains.txt --threshold 0.7
```

#### Detailed Verbose Output
```bash
python netra.py subdomains.txt --verbose
```

### 🧠 Knowledge Base Builder

#### Build Knowledge Base
```bash
python learning_analyzer.py
```

#### Seed Initial Knowledge
```bash
python seed_knowledge_base.py
```

### 🧑‍🔬 Enhanced Classifier

#### Train and Evaluate ML Models
```bash
python enhanced_classifier.py
```

### 🧪 Comprehensive Testing

#### Run Full Test Suite
```bash
python massive_testing_framework.py
```

#### Demo with Examples
```bash
python test_enhanced_demo.py
```

### Command Reference

```
NETRA - AI-Powered Subdomain Analysis Tool

Main Tool (netra.py):
  python netra.py [file] [options]
  --threshold FLOAT    Confidence threshold (0.0-1.0)
  --verbose           Detailed output with confidence scores
  --help              Show help message

Knowledge Builder (learning_analyzer.py):
  python learning_analyzer.py
  
Enhanced Classifier (enhanced_classifier.py):
  python enhanced_classifier.py
  
Testing Framework (massive_testing_framework.py):
  python massive_testing_framework.py
```

## 📊 Output Files

### Collection Output
- **`all_subs.txt`** (or custom filename): List of unique subdomains

### Analysis Output
- **`analysis.json`**: Structured JSON with categories and risks
- **`analysis_report.txt`**: Human-readable security report

### Example JSON Output
```json
[
  {
    "subdomain": "admin.example.com",
    "categories": ["Administrative / Management Interfaces"],
    "possible_risks": [
      "Weak authentication",
      "Exposed management interface",
      "Privilege escalation"
    ]
  },
  {
    "subdomain": "api-staging.example.com",
    "categories": ["APIs", "Staging / Development / Testing"],
    "possible_risks": [
      "Unauthenticated endpoints",
      "Outdated dev version",
      "Debug info leaks"
    ]
  }
]
```

## 🧪 Testing

Test NETRA with the comprehensive testing suite:

```bash
# Quick demo with examples
python test_enhanced_demo.py

# Basic NETRA analysis
python netra.py sample_subdomains.txt

# Full testing framework (10K+ samples)
python massive_testing_framework.py

# ML classifier evaluation
python enhanced_classifier.py
```

## 🏢 Architecture

### 🧩 NETRA Core Components
- **`netra.py`**: Main analysis engine with ML integration
- **`enhanced_classifier.py`**: Advanced ML classification system
- **`learning_analyzer.py`**: Knowledge base builder and analyzer
- **`massive_testing_framework.py`**: Comprehensive testing suite

### 📦 Data Flow
1. **Input Processing**: Subdomain list normalization
2. **Feature Extraction**: Multi-dimensional analysis
3. **ML Classification**: Confidence-based prediction
4. **Knowledge Integration**: Learning from patterns
5. **Output Generation**: Structured results with uncertainty

### ⚙️ Error Handling
- **Confidence thresholding**: Honest uncertainty reporting
- **Graceful degradation**: Fallback analysis methods
- **Input validation**: Robust subdomain processing
- **Model recovery**: Handles ML model failures

## 🔧 Extending NETRA

### Adding New Features

1. **Enhance Feature Extraction**:
```python
# In enhanced_classifier.py
def extract_advanced_features(self, subdomain):
    # Add new feature extraction logic
    features['new_metric'] = calculate_new_metric(subdomain)
    return features
```

2. **Add Classification Categories**:
```python
# In netra.py
categories = [
    "Administrative", "API", "Development",
    "Your New Category"  # Add here
]
```

### Improving ML Models

1. **Add New Algorithms**:
```python
# In enhanced_classifier.py
from sklearn.ensemble import GradientBoostingClassifier
models['gradient_boost'] = GradientBoostingClassifier()
```

2. **Enhance Knowledge Base**:
```python
# In learning_analyzer.py
def build_domain_knowledge(self, new_data_source):
    # Process additional data sources
    pass
```

## ⚠️ Security Considerations

- **Rate limiting**: Built-in delays for API calls
- **API key protection**: Never logs or exposes keys
- **Timeout controls**: Prevents resource exhaustion
- **Input validation**: Sanitizes domain inputs
- **Safe tool execution**: Proper subprocess handling

## 🐛 Troubleshooting

### Common Issues

**Dependencies missing:**
- Install complete requirements: `pip install -r requirements_complete.txt`
- Check Python version compatibility (3.7+)

**Low confidence predictions:**
- Normal behavior - NETRA reports uncertainty honestly
- Build larger knowledge base with `learning_analyzer.py`
- Lower confidence threshold: `--threshold 0.5`

**ML model errors:**
- Ensure scikit-learn is properly installed
- Check if models need retraining
- Fallback to heuristic analysis available

**File not found errors:**
- Verify input file path and format
- Use provided `sample_subdomains.txt` for testing
- Ensure write permissions for output files

## 📝 License

This tool is provided for educational and authorized security testing purposes only.

## 🤝 Contributing

Contributions are welcome! Please:

1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Submit pull requests

## 📞 Support

For issues, questions, or contributions:
- Check existing issues and documentation
- Provide detailed error messages and system info
- Include steps to reproduce problems

---

**⚡ Happy hunting!** Use responsibly and only on systems you own or have explicit permission to test.
