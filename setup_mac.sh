#!/bin/bash
# Setup script for macOS - Indian Names (Maharashtra Voter Name Extractor)

set -e  # Exit on error

echo "========================================="
echo "Indian Names - macOS Setup Script"
echo "========================================="
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed."
    echo "Please install Homebrew first: https://brew.sh"
    echo "Run: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "✓ Homebrew is installed"
echo ""

# Update Homebrew
echo "Updating Homebrew..."
brew update

# Install Tesseract OCR
echo ""
echo "Installing Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract is already installed ($(tesseract --version | head -n 1))"
else
    brew install tesseract
    echo "✓ Tesseract installed successfully"
fi

# Install Tesseract language data for Marathi
echo ""
echo "Installing Tesseract language data for Marathi..."
brew install tesseract-lang
echo "✓ Tesseract language data installed"

# Verify Marathi language support
echo ""
echo "Verifying Marathi (mar) language support..."
if tesseract --list-langs 2>/dev/null | grep -q "mar"; then
    echo "✓ Marathi language data is available"
else
    echo "⚠ Warning: Marathi language data may not be installed correctly"
    echo "  You may need to manually download it from: https://github.com/tesseract-ocr/tessdata"
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
        echo "  Install newer Python: brew install python@3.11"
    fi
else
    echo "⚠ Warning: Python 3 is not installed"
    echo "  Install Python: brew install python@3.11"
fi

# Check if pip is available
echo ""
echo "Checking pip..."
if command -v pip3 &> /dev/null; then
    echo "✓ pip3 is installed"
else
    echo "⚠ Warning: pip3 is not installed"
    echo "  Install pip: python3 -m ensurepip --upgrade"
fi

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment and install Python dependencies
echo ""
echo "Installing Python dependencies from requirements.txt..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Python dependencies installed"

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
