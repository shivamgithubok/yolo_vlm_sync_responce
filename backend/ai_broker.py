import sys
from pathlib import Path
from typing import Optional, Dict, Any
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
import backend.queen_ai as queen_ai

def get_wildlife_info(detected_class: str, base64_image: Optional[str] = None, history: Optional[str] = None, mime_type: str = "image/jpeg") -> Any:
    """
    Broker function that routes identification requests to QWEEN CLOUD (VL).
    """
    print(f"👑 [BROKER] Routing to QWEEN CLOUD (OpenRouter)")
    return queen_ai.get_wildlife_info(detected_class, base64_image, history, mime_type)

def set_vlm_mode(mode: str):
    """Update the VLM mode in runtime. (Only qween_cloud supported now)"""
    if mode.lower() == "qween_cloud":
        Config.VLM_MODE = "qween_cloud"
        print(f"🔄 [BROKER] VLM Mode switched to: {Config.VLM_MODE.upper()}")
        return True
    return False

def get_vlm_mode():
    """Get the current VLM mode."""
    return getattr(Config, "VLM_MODE", "cloud")
