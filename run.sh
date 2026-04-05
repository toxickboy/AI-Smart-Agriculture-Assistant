#!/bin/bash
echo "============================================"
echo "  AI Smart Agriculture Assistant"
echo "============================================"
echo

# Change to script directory
cd "$(dirname "$0")"

# Install if needed
python3 -c "import streamlit" 2>/dev/null || pip3 install -r requirements.txt

echo "Starting app at http://localhost:8501"
echo "Press Ctrl+C to stop."
echo

python3 -m streamlit run app.py
