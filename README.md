# 🔍 NETRA - AI-Powered Subdomain Analysis Tool

An intelligent subdomain analysis tool that builds its own knowledge base from real-world data, reducing dependency on external AI APIs and providing honest uncertainty reporting.

## 🚀 Features

### Collection Mode
- **Multi-tool subdomain enumeration** using:
  - Subfinder
  - Amass (passive mode)
  - Assetfinder
- **Automatic deduplication** and output consolidation
- **Graceful error handling** for missing tools
- **Modular architecture** for easy extension

### Analysis Mode
- **AI-powered categorization** using OpenAI's GPT models
- **Security risk assessment** for each subdomain
- **Structured JSON output** for automation
- **Human-readable reports** for manual review
- **Fallback heuristic analysis** when AI is unavailable

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

### Required Tools (for collection)
Install these subdomain enumeration tools:

```bash
# Install Subfinder
go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest

# Install Amass
go install -v github.com/OWASP/Amass/v3/...@master

# Install Assetfinder
go install github.com/tomnomnom/assetfinder@latest
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

### OpenAI API Key (optional)
Set your OpenAI API key for AI-powered analysis:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 🛠️ Installation

1. Clone or download the tool:
```bash
git clone <repository-url>
cd subdomain-ai
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Make the script executable (Linux/Mac):
```bash
chmod +x subdomain_ai.py
```

## 💻 Usage

### Basic Usage

#### Collection Only
```bash
python subdomain_ai.py --collect example.com
```

#### Analysis Only
```bash
python subdomain_ai.py --analyze subdomains.txt
```

#### Both Collection and Analysis
```bash
python subdomain_ai.py --both example.com
```

### Advanced Options

#### Custom Output File
```bash
python subdomain_ai.py --collect example.com --output my_subdomains.txt
```

#### Skip AI Analysis (Use Heuristics)
```bash
python subdomain_ai.py --analyze subdomains.txt --no-ai
```

#### Specify API Key
```bash
python subdomain_ai.py --analyze subdomains.txt --api-key "your-key"
```

### Command Reference

```
🔍 Subdomain AI - Security Research Tool

options:
  -h, --help           show help message
  --collect DOMAIN     Collect subdomains for the specified domain
  --analyze FILE       Analyze subdomains from the specified file
  --both DOMAIN        Collect and then analyze subdomains for the domain
  --output FILE        Output file for collected subdomains (default: all_subs.txt)
  --no-ai             Skip AI analysis and use heuristic-based analysis
  --api-key KEY        OpenAI API key (or set OPENAI_API_KEY env var)
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

Test the tool with the included sample:

```bash
# Test analysis functionality
python subdomain_ai.py --analyze sample_subdomains.txt

# Test with heuristic analysis (no AI)
python subdomain_ai.py --analyze sample_subdomains.txt --no-ai
```

## 🏗️ Architecture

### Modular Design
- **`SubdomainCollector`**: Handles enumeration tools
- **`SubdomainAnalyzer`**: Manages AI analysis and reporting
- **Clean separation** of collection and analysis logic
- **Easy extension** for new tools and analysis methods

### Error Handling
- **Graceful tool failures**: Continues with available tools
- **Timeout protection**: Prevents hanging on slow tools
- **Missing dependency detection**: Warns about unavailable tools
- **API error recovery**: Falls back to heuristic analysis

## 🔧 Extending the Tool

### Adding New Collection Tools

1. Add tool function to `SubdomainCollector`:
```python
def _run_newtool(self, domain: str) -> Set[str]:
    # Implementation here
    pass
```

2. Register in `__init__`:
```python
self.tools['newtool'] = self._run_newtool
```

### Adding New Analysis Categories

1. Update categories list in `SubdomainAnalyzer.__init__`:
```python
self.categories.append("New Category")
```

2. Add heuristic rules in `_mock_analysis` method

## ⚠️ Security Considerations

- **Rate limiting**: Built-in delays for API calls
- **API key protection**: Never logs or exposes keys
- **Timeout controls**: Prevents resource exhaustion
- **Input validation**: Sanitizes domain inputs
- **Safe tool execution**: Proper subprocess handling

## 🐛 Troubleshooting

### Common Issues

**Tools not found:**
- Ensure tools are in PATH
- Check installation with `subfinder -version`

**API errors:**
- Verify OPENAI_API_KEY is set
- Check API quota and billing
- Review network connectivity

**Empty results:**
- Domain may have no public subdomains
- Tools may be blocked by rate limiting
- Check domain spelling and validity

**Permission errors:**
- Ensure write permissions in current directory
- Check file system space availability

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
