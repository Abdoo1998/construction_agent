#!/bin/bash

echo "PDF RAG System Setup Script for Unix-like systems (macOS/Linux)"
echo "=============================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    echo "Please install Python 3 using your package manager:"
    echo "  - macOS: brew install python3"
    echo "  - Ubuntu/Debian: sudo apt install python3 python3-venv"
    echo "  - CentOS/RHEL: sudo yum install python3 python3-venv"
    exit 1
fi

# Check if virtual environment exists, create if it doesn't
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment and install requirements
echo "Installing requirements..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements."
    exit 1
fi

echo ""
echo "Setup completed successfully!"
echo ""
echo "To run the RAG API server:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "To deactivate the virtual environment when done:"
echo "  deactivate"
echo ""

# Make the script executable
chmod +x setup.sh 