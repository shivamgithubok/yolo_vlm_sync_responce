"""
Wildlife identification system using Vision Language Models.

Improvements:
- Better error handling (raises exceptions instead of returning Unknown)
- Configurable model names
- Timeout support
- Return values from save functions
"""

import json
from typing import Optional, Dict, Any
from openai import OpenAI
from pydantic import BaseModel, Field
import sys
from pathlib import Path
import sqlite3

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

# Initialize client with OpenRouter's base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=Config.OPENROUTER_API_KEY,
)

DB_PATH = Path(__file__).parent / "animals.db"

# Configuration constants
DEFAULT_TIMEOUT = 30.0
SEARCH_MODEL = getattr(Config, "SEARCH_MODEL", "google/gemini-3-flash-preview:online")
VLM_MODEL = getattr(Config, "VLM_MODEL", "google/gemini-3-flash-preview")


class Wildlife(BaseModel):
    """Wildlife information model."""
    id: Optional[str] = Field(default=None, description="Unique identifier")
    detected_class: str = Field(description="YOLO detection class name (e.g., buffalo, elephant)")
    is_animal: bool = Field(description="Whether the detected object is an animal")
    is_person: bool = Field(default=False, description="Whether the detected object is a person")
    commonName: Optional[str] = None
    scientificName: Optional[str] = None
    description: Optional[str] = None
    habitat: Optional[str] = None
    behavior: Optional[str] = None
    safetyInfo: Optional[str] = None
    conservationStatus: Optional[str] = Field(default=None, description="LC, NT, VU, EN, or CR")
    isDangerous: Optional[bool] = Field(default=None)
    imageUrl: Optional[str] = None

    # Additional info from animals.db/web search
    diet: Optional[str] = None
    lifespan: Optional[str] = None
    height_cm: Optional[str] = None
    weight_kg: Optional[str] = None
    color: Optional[str] = None
    predators: Optional[str] = None
    average_speed_kmh: Optional[str] = None
    countries_found: Optional[str] = None
    family: Optional[str] = None
    gestation_period_days: Optional[str] = None
    top_speed_kmh: Optional[str] = None
    social_structure: Optional[str] = None
    offspring_per_birth: Optional[str] = None


def search_local_db(name: str, by_scientific: bool = False) -> Optional[Dict[str, Any]]:
    """
    Search for animal in local animals.db.
    
    Args:
        name: Animal name to search for
        by_scientific: If True, search by scientific name, else by common name
        
    Returns:
        Dictionary with animal info if found, None otherwise
    """
    if not DB_PATH.exists():
        print(f"⚠️ animals.db not found at {DB_PATH}")
        return None
    
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Use parameterized query for safety
        column = "scientific_name" if by_scientific else "animal"
        cursor.execute(f"SELECT * FROM animals WHERE {column} LIKE ? COLLATE NOCASE", (f"%{name}%",))
        row = cursor.fetchone()
        
        if row:
            data = row
            return {
                "id": str(data["id"]),
                "commonName": data["animal"],
                "scientificName": data["scientific_name"],
                "habitat": data["habitat"],
                "diet": data["diet"],
                "lifespan": data["lifespan_years"],
                "conservationStatus": data["conservation_status"],
                "height_cm": data["height_cm"],
                "weight_kg": data["weight_kg"],
                "color": data["color"],
                "predators": data["predators"],
                "average_speed_kmh": data["average_speed_kmh"],
                "countries_found": data["countries_found"],
                "family": data["family"],
                "gestation_period_days": data["gestation_period_days"],
                "top_speed_kmh": data["top_speed_kmh"],
                "social_structure": data["social_structure"],
                "offspring_per_birth": data["offspring_per_birth"],
                "is_animal": True
            }

        return None
    except Exception as e:
        print(f"❌ Error searching local DB: {e}")
        return None
    finally:
        conn.close()


def search_web_for_animal(animal_name: str, timeout: float = DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    """
    Search for animal info using OpenRouter search-preview model.
    
    Args:
        animal_name: Name of the animal to search for
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with animal info if found, None otherwise
        
    Raises:
        Exception: If the API call fails
    """
    print(f"🌐 Searching web for: {animal_name}")
    try:
        response = client.chat.completions.create(
            model=SEARCH_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a wildlife expert. Provide detailed information about the animal in JSON format."
                },
                {
                    "role": "user",
                    "content": f"Search for and provide detailed info about '{animal_name}' including: common name, scientific name, description, habitat, behavior, safety info, conservation status (LC, NT, VU, EN, or CR), height (cm), weight (kg), color, lifespan (years), diet, predators, average speed (km/h), countries found, family, gestation period (days), top speed (km/h), social structure, and offspring per birth."
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "animal_info",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "commonName": {"type": "string"},
                            "scientificName": {"type": "string"},
                            "description": {"type": "string"},
                            "habitat": {"type": "string"},
                            "behavior": {"type": "string"},
                            "safetyInfo": {"type": "string"},
                            "conservationStatus": {"type": "string"},
                            "isDangerous": {"type": "boolean"},
                            "diet": {"type": "string"},
                            "lifespan": {"type": "string"},
                            "height_cm": {"type": "string"},
                            "weight_kg": {"type": "string"},
                            "color": {"type": "string"},
                            "predators": {"type": "string"},
                            "average_speed_kmh": {"type": "string"},
                            "countries_found": {"type": "string"},
                            "family": {"type": "string"},
                            "gestation_period_days": {"type": "string"},
                            "top_speed_kmh": {"type": "string"},
                            "social_structure": {"type": "string"},
                            "offspring_per_birth": {"type": "string"}
                        },
                        "required": [
                            "commonName", "scientificName", "description", "habitat", "behavior", 
                            "safetyInfo", "conservationStatus", "isDangerous", "diet", "lifespan", 
                            "height_cm", "weight_kg", "color", "predators", "average_speed_kmh", 
                            "countries_found", "family", "gestation_period_days", "top_speed_kmh", 
                            "social_structure", "offspring_per_birth"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            timeout=timeout
        )
        
        data = json.loads(response.choices[0].message.content)
        data["is_animal"] = True
        print(f"✓ Found web info for: {data.get('commonName')}")
        return data
        
    except Exception as e:
        print(f"❌ Error searching web: {e}")
        raise  # Re-raise for retry logic


def save_to_local_db(data: Dict[str, Any]) -> bool:
    """
    Save animal data to local animals.db.
    
    Args:
        data: Animal information dictionary
        
    Returns:
        True if saved successfully, False otherwise
    """
    if not DB_PATH.exists():
        print(f"⚠️ animals.db not found at {DB_PATH}")
        return False
    
    print(f"💾 Saving {data.get('commonName')} to local DB")
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Get common and scientific names
        animal = data.get("commonName")
        sci_name = data.get("scientificName")
        
        # Check if already exists
        cursor.execute("SELECT 1 FROM animals WHERE animal = ? OR scientific_name = ?", (animal, sci_name))
        if cursor.fetchone():
            print(f"ℹ️ {animal} already exists in DB")
            conn.close()
            return True  # Already exists is success
        
        # Prepare insertion based on table schema
        columns = (
            "animal, height_cm, weight_kg, color, lifespan_years, diet, habitat, "
            "predators, average_speed_kmh, countries_found, conservation_status, "
            "family, gestation_period_days, top_speed_kmh, social_structure, "
            "offspring_per_birth, scientific_name"
        )
        
        placeholders = ", ".join(["?"] * 17)
        
        values = (
            animal,
            data.get("height_cm"),
            data.get("weight_kg"),
            data.get("color"),
            data.get("lifespan"),
            data.get("diet"),
            data.get("habitat"),
            data.get("predators"),
            data.get("average_speed_kmh"),
            data.get("countries_found"),
            data.get("conservationStatus"),
            data.get("family"),
            data.get("gestation_period_days"),
            data.get("top_speed_kmh"),
            data.get("social_structure"),
            data.get("offspring_per_birth"),
            sci_name
        )
        
        cursor.execute(f"INSERT INTO animals ({columns}) VALUES ({placeholders})", values)
        conn.commit()
        conn.close()
        
        print(f"✓ Saved {animal} to local DB")
        return True
        
    except Exception as e:
        print(f"❌ Error inserting into DB: {e}")
        return False


def identify_wildlife(
    detected_class: str, 
    base64_image: Optional[str] = None, 
    history: Optional[str] = None, 
    mime_type: str = "image/jpeg",
    timeout: float = DEFAULT_TIMEOUT
) -> Dict[str, Any]:
    """
    Identify wildlife using Vision Language Model.
    
    Args:
        detected_class: YOLO detection class name (e.g., "bird", "elephant")
        base64_image: Optional base64-encoded image
        history: Optional context about recent sightings
        mime_type: MIME type of the image
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with wildlife information
        
    Raises:
        Exception: If VLM API call fails (for retry logic)
    """
    print(f"\n🔍 Identifying: {detected_class}")
    if history:
        print(f"📜 Context: Recently seen {history}")
    
    # Build messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a wildlife expert. Analyze the image and classify the animal. "
                "Provide accurate identification including common name, scientific name, and description. "
                "If it's a person or not an animal, set is_animal=false and is_person=true if human."
            )
        }
    ]
    
    # Build user message content
    user_content = []
    
    # Add image if provided
    if base64_image:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        })
    
    # Add text prompt
    prompt = f"Detected class: {detected_class}."
    if history:
        prompt += f" Recent sightings: {history}."
    prompt += " Identify this animal with accurate details."
    
    user_content.append({
        "type": "text",
        "text": prompt
    })
    
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    # Call VLM with structured output
    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=messages,
            timeout=timeout,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "wildlife_identification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_animal": {
                                "type": "boolean",
                                "description": "Whether the detected object is an animal"
                            },
                            "is_person": {
                                "type": "boolean",
                                "description": "Whether the detected object is a person"
                            },
                            "commonName": {
                                "type": "string",
                                "description": "Common name of the animal"
                            },
                            "scientificName": {
                                "type": "string",
                                "description": "Scientific name (binomial nomenclature)"
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the animal"
                            }
                        },
                        "required": [
                            "is_animal",
                            "is_person",
                            "commonName",
                            "scientificName",
                            "description"
                        ],
                        "additionalProperties": False
                    }
                }
            }
        )
    except Exception as e:
        print(f"❌ VLM API call failed: {e}")
        raise  # Re-raise for retry logic

    # Validate response
    if not response or not hasattr(response, 'choices') or not response.choices:
        raise Exception("VLM API returned empty response")

    content = response.choices[0].message.content
    if not content:
        raise Exception("VLM message content is empty")
    
    # Clean markdown if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip("` \n")

    try:
        wildlife_data = json.loads(content)
        if not isinstance(wildlife_data, dict):
            raise ValueError("VLM response is not a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Failed to parse JSON: {e}")
        print(f"DEBUG: VLM Content: {content}")
        raise Exception(f"Invalid JSON response from VLM: {e}")
    
    # Map snake_case to camelCase if necessary (fallback for legacy models)
    if "common_name" in wildlife_data:
        wildlife_data.setdefault("commonName", wildlife_data.pop("common_name"))
    if "scientific_name" in wildlife_data:
        wildlife_data.setdefault("scientificName", wildlife_data.pop("scientific_name"))
    
    # --- SEARCH LOGIC ---
    common_name = wildlife_data.get("commonName")
    if not common_name and detected_class and detected_class.lower() != "person":
        common_name = detected_class
        print(f"ℹ️ VLM did not provide name, falling back to class: {common_name}")
    
    sci_name = wildlife_data.get("scientificName")
    
    # Search for additional information
    search_results = None
    
    # Try local DB first (by common name)
    if common_name:
        print(f"🔍 Searching local DB for: {common_name}")
        search_results = search_local_db(common_name)
    
    # If not found by common name, try web search FIRST to get specific info
    # This ensures new breeds (e.g. "Beagle") are added even if "Dog" (Canis lupus) exists
    if not search_results and (common_name or sci_name):
        search_name = common_name or sci_name
        try:
            search_results = search_web_for_animal(search_name, timeout=timeout)
            if search_results:
                # Save to local DB for future use
                save_to_local_db(search_results)
        except Exception as e:
            print(f"⚠️ Web search failed, checking local DB for scientific name fallback: {e}")
            
            # Fallback: Try local DB by scientific name
            if sci_name:
                print(f"🔍 Searching local DB for: {sci_name}")
                search_results = search_local_db(sci_name, by_scientific=True)
    
    # Merge search results with VLM data
    if search_results:
        # Prioritize search-based technical info but keep VLM descriptions if richer
        for k, v in search_results.items():
            if v and (k not in wildlife_data or not wildlife_data[k] or wildlife_data[k] == "Unknown"):
                wildlife_data[k] = v
    
    # Add detected_class back to the data
    wildlife_data["detected_class"] = detected_class
    
    print(f"✓ Identified: {wildlife_data.get('commonName')} ({wildlife_data.get('scientificName')})")
    return wildlife_data


def get_wildlife_info(
    detected_class: str, 
    base64_image: Optional[str] = None, 
    history: Optional[str] = None, 
    mime_type: str = "image/jpeg"
) -> Wildlife:
    """
    Get wildlife information and return as Wildlife model instance.
    
    Args:
        detected_class: YOLO detection class name
        base64_image: Optional base64-encoded image string
        history: Optional text describing previous sightings for context
        mime_type: MIME type of the image (default: "image/jpeg")
        
    Returns:
        Wildlife model instance with all information populated
        
    Raises:
        Exception: If identification fails (for retry logic)
    """
    data = identify_wildlife(detected_class, base64_image, history, mime_type)
    return Wildlife(**data)
