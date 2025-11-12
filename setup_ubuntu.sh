#!/bin/bash
# Setup script for Ubuntu/Debian - Indian Names (Maharashtra Voter Name Extractor)

set -e  # Exit on error

echo "========================================="
echo "Indian Names - Ubuntu/Debian Setup Script"
echo "========================================="
echo ""

# Check if running with sudo or as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script requires sudo privileges to install system packages."
    echo "Please run with: sudo ./setup_ubuntu.sh"
    exit 1
fi

echo "✓ Running with sudo privileges"
echo ""

# Update package lists
echo "Updating package lists..."
apt update

# Install Tesseract OCR
echo ""
echo "Installing Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract is already installed ($(tesseract --version | head -n 1))"
else
    apt install -y tesseract-ocr
    echo "✓ Tesseract installed successfully"
fi

# Install Tesseract language data for Marathi
echo ""
echo "Installing Tesseract language data for Marathi..."
apt install -y tesseract-ocr-mar
echo "✓ Marathi language data installed"

# Verify Marathi language support
echo ""
echo "Verifying Marathi (mar) language support..."
if tesseract --list-langs 2>/dev/null | grep -q "mar"; then
    echo "✓ Marathi language data is available"
else
    echo "⚠ Warning: Marathi language data may not be installed correctly"
    echo "  You may need to manually install it: sudo apt install tesseract-ocr-mar"
fi

# Check Python version
echo ""
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✓ Python $PYTHON_VERSION is installed"

    # Check if Python version is 3.8+
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        echo "✓ Python version meets requirements (3.8+)"
    else
        echo "⚠ Warning: Python 3.8+ is required. Current version: $PYTHON_VERSION"
        echo "  Attempting to install Python 3.11..."
        apt install -y python3.11 python3.11-venv python3.11-dev
    fi
else
    echo "Installing Python 3..."
    apt install -y python3 python3-venv python3-dev
fi

# Check if pip is available, install if needed
echo ""
echo "Checking pip..."
if command -v pip3 &> /dev/null; then
    echo "✓ pip3 is installed"
else
    echo "Installing pip3..."
    apt install -y python3-pip
    echo "✓ pip3 installed"
fi

# Install python3-venv if not available
echo ""
echo "Ensuring python3-venv is installed..."
apt install -y python3-venv
echo "✓ python3-venv is installed"

# Get the original user (the one who ran sudo)
ORIGINAL_USER=${SUDO_USER:-$USER}
ORIGINAL_HOME=$(eval echo ~$ORIGINAL_USER)

# Navigate to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create virtual environment as the original user
echo ""
echo "Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists"
else
    sudo -u $ORIGINAL_USER python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment and install Python dependencies
echo ""
echo "Installing Python dependencies from requirements.txt..."
sudo -u $ORIGINAL_USER bash << EOF
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Python dependencies installed"
EOF

# Fix permissions
chown -R $ORIGINAL_USER:$ORIGINAL_USER venv

echo ""
echo "========================================="
echo "✓ Setup completed successfully!"
echo "========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application:"
echo "  python extract_voter_names.py --help"
echo ""
