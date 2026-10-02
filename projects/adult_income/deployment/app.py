"""Compatibility launcher for the original Streamlit path."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apps.streamlit_app import main
main()
