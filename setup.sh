#!/bin/bash
# Quick setup script for SRL + SDK

echo "SRL + Synthetic Data Kit Setup"
echo "==================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

echo "Python found: $(python3 --version)"

# Create virtualenv
echo ""
echo "Creating virtual environment..."
python3 -m venv srl_env
source srl_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start vLLM backend:"
echo "   vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000"
echo ""
echo "2. Run the full pipeline:"
echo "   bash run_pipeline.sh"
echo ""
echo "3. Or run individual commands (see QUICKSTART_SDK.md)"
echo ""
echo "To deactivate environment: deactivate"
