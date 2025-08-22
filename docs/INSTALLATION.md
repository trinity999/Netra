# Installation Guide

## Prerequisites

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended for training
- **Storage**: 2GB free space for models and data
- **CPU**: Multi-core processor recommended for training

### Python Dependencies
The core dependencies are listed in `requirements.txt`:
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `psutil>=5.8.0` - System monitoring
- `joblib>=1.1.0` - Model persistence

## Installation Methods

### Method 1: Clone from GitHub (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/netra.git
cd netra

# Create virtual environment (recommended)
python -m venv netra-env
source netra-env/bin/activate  # On Windows: netra-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python tests/verify_incremental_model.py
```

### Method 2: Download ZIP
1. Download the ZIP file from GitHub
2. Extract to your desired directory
3. Follow the same steps as Method 1 starting from creating virtual environment

## Initial Setup

### 1. Verify Python Version
```bash
python --version
# Should show Python 3.8 or higher
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Test Basic Functionality
```bash
# Test the core system
python tests/test_enhanced_demo.py

# Verify trained model
python tests/verify_incremental_model.py
```

## Configuration

### Environment Variables (Optional)
Create a `.env` file in the root directory for custom configuration:
```bash
# Optional: Custom paths
NETRA_MODEL_PATH=/path/to/custom/models/
NETRA_DATA_PATH=/path/to/custom/data/

# Optional: Performance tuning
NETRA_MAX_MEMORY=80
NETRA_BATCH_SIZE=1000
```

### Model Files
The pre-trained models should be in the `models/` directory:
- `incremental_model.joblib` - Main AI model
- `incremental_vectorizer.joblib` - Text vectorizer
- `incremental_encoders.joblib` - Label encoders
- `accumulated_training_data.joblib` - Training dataset

## Quick Start

### Basic Usage
```python
from src.core.subdomain_ai_enhanced import SubdomainAIEnhanced

# Initialize NETRA
ai = SubdomainAIEnhanced()

# Classify a subdomain
result = ai.classify_subdomain("api.example.com")
print(f"Category: {result.predicted_category}")
print(f"Confidence: {result.confidence:.3f}")
```

### Command Line Usage
```bash
# Legacy interface
python src/legacy/netra.py sample_subdomains.txt

# Enhanced analysis
python tests/test_enhanced_demo.py
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're in the project root directory
cd /path/to/netra

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use absolute imports
python -m src.core.subdomain_ai_enhanced
```

#### 2. Missing Dependencies
**Error**: `ImportError: No module named 'sklearn'`

**Solution**:
```bash
pip install -r requirements.txt

# If still failing, try upgrading pip
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Model Files Not Found
**Error**: `FileNotFoundError: incremental_model.joblib not found`

**Solution**:
- Ensure models are in `models/` directory
- If missing, you may need to retrain:
```bash
python src/training/enhanced_incremental_trainer.py
```

#### 4. Memory Issues During Training
**Error**: `MemoryError` or system freezing

**Solution**:
- Ensure you have at least 8GB RAM available
- Close other applications during training
- Reduce chunk size in training configuration
- Use system monitoring during training:
```python
from src.training.enhanced_incremental_trainer import EnhancedIncrementalTrainer
trainer = EnhancedIncrementalTrainer(chunk_size=500, max_memory_usage=70)
```

#### 5. Permission Errors
**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
# On Linux/macOS
chmod +x src/training/enhanced_incremental_trainer.py

# On Windows, run as Administrator or check file permissions
```

### Diagnostic Commands
```bash
# Check Python environment
python -c "import sys; print(sys.version)"
python -c "import sklearn; print(sklearn.__version__)"

# Check system resources
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB available')"

# Verify model files
ls -la models/

# Test core imports
python -c "from src.core.subdomain_ai_enhanced import SubdomainAIEnhanced; print('✅ Core imports working')"
```

## Development Setup

### For Contributors
```bash
# Clone with development dependencies
git clone https://github.com/yourusername/netra.git
cd netra

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

### Project Structure
```
netra/
├── src/           # Source code
├── models/        # Trained AI models
├── data/          # Training data and results
├── tests/         # Test files
├── docs/          # Documentation
├── requirements.txt
└── README.md
```

## Performance Optimization

### For Training
- Ensure 16GB+ RAM for full training
- Use SSD storage for better I/O performance
- Close other applications during training
- Monitor system resources

### For Classification
- Pre-load models at application startup
- Use batch processing for multiple subdomains
- Enable result caching for repeated queries

## Support

If you encounter issues not covered here:

1. **Check Documentation**: Review README.md and other docs/
2. **Verify Setup**: Run diagnostic commands above
3. **Check System Resources**: Ensure adequate RAM and storage
4. **Update Dependencies**: `pip install --upgrade -r requirements.txt`
5. **Create Issue**: If problem persists, create a GitHub issue with:
   - Your system information
   - Error messages
   - Steps to reproduce

## Next Steps

After successful installation:
1. **Explore Examples**: Run `python tests/test_enhanced_demo.py`
2. **Read Documentation**: Check `README.md` for usage examples
3. **Try Training**: If you have custom data, try the incremental trainer
4. **Integration**: Integrate NETRA into your security workflow

---

**🎯 You're ready to use NETRA for advanced subdomain intelligence analysis!**
