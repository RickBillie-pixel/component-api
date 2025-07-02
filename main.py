"""
Component API - Detects architectural components from extracted vector data
Identifies doors, windows, and other building components
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import math
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("component_api")

app = FastAPI(
    title="Component Detection API",
    description="Detects architectural components from extracted vector data",
    version="1.0.0",
)

class PageData(BaseModel):
    page_number: int
    drawings: List[Dict[str, Any]]
    texts: List[Dict[str, Any]]

class Wall(BaseModel):
    p1: Dict[str, float]
    p2: Dict[str, float]
    wall_thickness: float
    wall_length: float
    wall_type: str
    confidence: float
    reason: str

class ComponentDetectionRequest(BaseModel):
    pages: List[PageData]
    walls: List[List[Wall]]
    scale_m_per_pixel: float = 1.0

@app.post("/detect-components/")
async def detect_components(request: ComponentDetectionRequest):
    """
    Detect architectural components from extracted vector data
    
    Args:
        request: JSON with pages, walls, and scale information
        
    Returns:
        JSON with detected components for each page
    """
    try:
        logger.info(f"Detecting components for {len(request.pages)} pages with scale {request.scale_m_per_pixel}")
        
        results = []
        
        for i, page_data in enumerate(request.pages):
            logger.info(f"Analyzing components on page {page_data.page_number}")
            
            # Convert walls to list of dictionaries
            walls_dict = [wall.dict() for wall in request.walls[i]]
            
            components = _detect_components(page_data, walls_dict, request.scale_m_per_pixel)
            
            results.append({
                "page_number": page_data.page_number,
                "components": components
            })
        
        logger.info(f"Successfully detected components for {len(results)} pages")
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting components: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _detect_components(page_data: PageData, walls: List[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    """
    Detect architectural components using rule-based approach
    
    Args:
        page_data: Page data containing drawings
        walls: List of wall dictionaries
        scale: Scale factor in meters per pixel
        
    Returns:
        List of detected components with properties
    """
    components = []
    
    # Extract arcs and lines for door detection
    arcs = []
    lines = []
    
    for drawing in page_data.drawings:
        for item in drawing["items"]:
            if item["type"] == "curve":
                arcs.append(item)
            elif item["type"] == "line":
                lines.append(item)
    
    # Detect doors (arc + line combination)
    for arc in arcs:
        for line in lines:
            # Check if arc and line are connected
            if (_points_close(arc["p1"], line["p1"]) or 
                _points_close(arc["p1"], line["p2"])):
                
                width = math.hypot(
                    line["p2"]["x"] - line["p1"]["x"], 
                    line["p2"]["y"] - line["p1"]["y"]
                ) * scale
                
                components.append({
                    "type": "door",
                    "position": line["p1"],
                    "width_m": round(width, 2),
                    "confidence": 1.0,
                    "reason": "Arc+line door geometry"
                })
    
    # Detect sliding doors (double parallel lines)
    for i, l1 in enumerate(walls):
        for j, l2 in enumerate(walls):
            if i >= j:
                continue
            
            # Check if lines are parallel and close together
            if (_is_parallel_lines(l1, l2) and 
                _points_close(l1["p1"], l2["p1"]) and 
                _points_close(l1["p2"], l2["p2"])):
                
                width = math.hypot(
                    l1["p2"]["x"] - l1["p1"]["x"], 
                    l1["p2"]["y"] - l1["p1"]["y"]
                ) * scale
                
                components.append({
                    "type": "sliding_door",
                    "position": l1["p1"],
                    "width_m": round(width, 2),
                    "confidence": 0.9,
                    "reason": "Double parallel lines"
                })
    
    # Detect windows (thin rectangles)
    for drawing in page_data.drawings:
        for item in drawing["items"]:
            if item["type"] == "rect":
                rect = item["rect"]
                width = rect["width"] * scale
                height = rect["height"] * scale
                
                # Window dimensions should be reasonable
                if 0.02 < width < 0.5 and 0.02 < height < 0.5:
                    components.append({
                        "type": "window",
                        "position": {
                            "x": (rect["x0"] + rect["x1"]) / 2, 
                            "y": (rect["y0"] + rect["y1"]) / 2
                        },
                        "width_m": round(width, 2),
                        "height_m": round(height, 2),
                        "confidence": 0.9,
                        "reason": "Thin rectangle in wall"
                    })
    
    if not components:
        logger.warning(f"No components detected on page {page_data.page_number}")
        return [{
            "type": "unknown", 
            "reason": "No components detected", 
            "confidence": 0.0
        }]
    
    logger.info(f"Detected {len(components)} components on page {page_data.page_number}")
    return components

def _points_close(p1: Dict[str, float], p2: Dict[str, float], tolerance: float = 5.0) -> bool:
    """Check if two points are close together"""
    dx = abs(p2["x"] - p1["x"])
    dy = abs(p2["y"] - p1["y"])
    return dx < tolerance and dy < tolerance

def _is_parallel_lines(l1: Dict[str, Any], l2: Dict[str, Any], tolerance: float = 0.01) -> bool:
    """Check if two lines have similar thickness (parallel walls)"""
    return abs(l1["wall_thickness"] - l2["wall_thickness"]) < tolerance

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "component-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004) 