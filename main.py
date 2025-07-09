"""
Component API - Detects architectural components from extracted vector data
Implements knowledge base rules (Rule 5.4, 5.5, 5.7, 5.9) for component detection
Identifies doors, windows, stairs, and other building components
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import math
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("component_api")

# Knowledge Base Constants (Rules 5.4, 5.5)
DOOR_WIDTH_MIN = 0.7  # meters - minimum door width (Rule 5.4)
DOOR_WIDTH_MAX = 1.2  # meters - maximum door width (Rule 5.4)
WINDOW_WIDTH_MIN = 0.6  # meters - minimum window width (Rule 5.4)
WINDOW_WIDTH_MAX = 3.0  # meters - maximum window width (Rule 5.4)
STAIRS_WIDTH_MIN = 0.8  # meters - minimum stairs width (Rule 5.5)
STAIRS_WIDTH_MAX = 1.2  # meters - maximum stairs width (Rule 5.5)
STAIRS_RISER_MAX = 0.188  # meters - maximum riser height (Rule 5.5)
STAIRS_TREAD_MIN = 0.22  # meters - minimum tread depth (Rule 5.5)
STAIRS_RAILING_MIN = 0.85  # meters - minimum railing height (Rule 5.5)
POINT_TOLERANCE = 0.05  # meters - tolerance for point matching

# Component patterns for text detection
COMPONENT_PATTERNS = {
    "door": [
        r"deur|door|entry|entrance|doorway",
        r"DE\d+|D\d+"  # Door codes
    ],
    "window": [
        r"raam|window|venster|glasopening",
        r"RW\d+|W\d+"  # Window codes
    ],
    "sliding_door": [
        r"schuifdeur|sliding door|sliding",
        r"SD\d+|SL\d+"  # Sliding door codes
    ],
    "stairs": [
        r"trap|stairs|staircase|stairway",
        r"TR\d+"  # Stairs codes
    ],
    "plinth": [
        r"plint|plinth|baseboard",
        r"PL\d+"  # Plinth codes
    ],
    "balcony": [
        r"balkon|balcony|terrace|terras",
        r"BA\d+"  # Balcony codes
    ],
    "opening": [
        r"opening|doorgang|sparing",
        r"OP\d+"  # Opening codes
    ]
}

app = FastAPI(
    title="Component Detection API",
    description="Detects architectural components from extracted vector data",
    version="1.0.0",
)

class Wall(BaseModel):
    type: str
    label_code: str
    label_nl: str
    label_en: str
    label_type: str
    thickness_meters: float
    properties: Dict[str, Any]
    classification: Dict[str, Any]
    line1_index: int
    line2_index: int
    orientation: str
    wall_type: str
    confidence: float
    reason: str

class TextItem(BaseModel):
    text: str
    position: Dict[str, float]
    font_size: float
    font_name: str
    color: List[float] = [0, 0, 0]
    bbox: Dict[str, float]

class DrawingItem(BaseModel):
    type: str
    p1: Optional[Dict[str, float]] = None
    p2: Optional[Dict[str, float]] = None
    p3: Optional[Dict[str, float]] = None
    rect: Optional[Dict[str, float]] = None
    length: Optional[float] = None
    color: List[float] = [0, 0, 0]
    width: Optional[float] = 1.0
    area: Optional[float] = None
    fill: List[Any] = []

class Drawings(BaseModel):
    lines: List[DrawingItem]
    rectangles: List[DrawingItem]
    curves: List[DrawingItem]

class PageData(BaseModel):
    page_number: int
    page_size: Dict[str, float]
    drawings: Drawings
    texts: List[TextItem]
    is_vector: bool = True
    processing_time_ms: Optional[int] = None

class ComponentDetectionRequest(BaseModel):
    pages: List[PageData]
    walls: List[List[Wall]]
    scale_m_per_pixel: float = 1.0

class ComponentDetectionResponse(BaseModel):
    pages: List[Dict[str, Any]]

# Utility functions
def distance(p1: dict, p2: dict) -> float:
    """Calculate distance between two points"""
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def is_point_near_line(point: Dict[str, float], line_p1: Dict[str, float], line_p2: Dict[str, float], tolerance: float) -> bool:
    """Check if a point is near a line segment"""
    # Calculate distance from point to line
    line_length = distance(line_p1, line_p2)
    if line_length == 0:
        return distance(point, line_p1) <= tolerance
    
    t = ((point['x'] - line_p1['x']) * (line_p2['x'] - line_p1['x']) + 
         (point['y'] - line_p1['y']) * (line_p2['y'] - line_p1['y'])) / (line_length ** 2)
    
    if t < 0:
        return distance(point, line_p1) <= tolerance
    elif t > 1:
        return distance(point, line_p2) <= tolerance
    
    projection = {
        'x': line_p1['x'] + t * (line_p2['x'] - line_p1['x']),
        'y': line_p1['y'] + t * (line_p2['y'] - line_p1['y'])
    }
    
    return distance(point, projection) <= tolerance

def is_point_on_wall(point: Dict[str, float], wall: Wall, tolerance: float) -> bool:
    """Check if a point is on or near a wall"""
    if "polygon" in wall.properties:
        # Check if point is near any edge of the wall polygon
        polygon = wall.properties["polygon"]
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i+1) % len(polygon)]
            if is_point_near_line(point, p1, p2, tolerance):
                return True
        return False
    else:
        # Fallback to checking against wall endpoints
        p1 = wall.properties.get("p1", {})
        p2 = wall.properties.get("p2", {})
        return is_point_near_line(point, p1, p2, tolerance)

def find_component_type_from_text(text: str) -> str:
    """Determine component type from text using patterns"""
    text_lower = text.lower()
    
    for comp_type, patterns in COMPONENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return comp_type
    
    return None

def find_component_code_from_text(text: str) -> str:
    """Extract component code from text (e.g., 'DE01', 'RW03')"""
    code_patterns = [
        r'(DE\d+)',  # Door codes
        r'(RW\d+)',  # Window codes
        r'(DK\d+)',  # Tilt-turn window codes
        r'(TR\d+)',  # Stairs codes
        r'(BA\d+)',  # Balcony codes
        r'(PL\d+)',  # Plinth codes
        r'(OP\d+)'   # Opening codes
    ]
    
    for pattern in code_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None

@app.post("/detect-components/", response_model=ComponentDetectionResponse)
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
            
            # Get walls for current page
            page_walls = request.walls[i] if i < len(request.walls) else []
            
            components = _detect_components(page_data, page_walls, request.scale_m_per_pixel)
            
            results.append({
                "page_number": page_data.page_number,
                "components": components
            })
        
        logger.info(f"Successfully detected components for {len(results)} pages")
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting components: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _detect_components(page_data: PageData, walls: List[Wall], scale: float) -> List[Dict[str, Any]]:
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
    rectangles = []
    
    # Collect all drawing elements
    for line in page_data.drawings.lines:
        lines.append(line.dict())
    
    for rect in page_data.drawings.rectangles:
        rectangles.append(rect.dict())
    
    for curve in page_data.drawings.curves:
        if curve.type == "curve":
            arcs.append(curve.dict())
    
    # Convert texts to dictionaries
    texts = [text.dict() for text in page_data.texts]
    
    # Convert walls to dictionaries
    walls_dict = [wall.dict() for wall in walls]
    
    # Step 1: Detect doors (arc + line combination) - Rule 5.4
    logger.info("Detecting doors...")
    for arc in arcs:
        for line in lines:
            # Check if arc and line are connected
            if (_points_close(arc["p1"], line["p1"], POINT_TOLERANCE / scale) or 
                _points_close(arc["p1"], line["p2"], POINT_TOLERANCE / scale) or
                _points_close(arc["p3"], line["p1"], POINT_TOLERANCE / scale) or
                _points_close(arc["p3"], line["p2"], POINT_TOLERANCE / scale)):
                
                # Calculate door width
                width = distance(line["p1"], line["p2"]) * scale
                
                # Check if width is within door range
                if DOOR_WIDTH_MIN <= width <= DOOR_WIDTH_MAX:
                    # Determine door position (midpoint of line)
                    position = {
                        "x": (line["p1"]["x"] + line["p2"]["x"]) / 2,
                        "y": (line["p1"]["y"] + line["p2"]["y"]) / 2
                    }
                    
                    # Check if door is on a wall
                    on_wall = False
                    wall_data = None
                    for wall in walls_dict:
                        if is_point_on_wall(position, Wall(**wall), POINT_TOLERANCE / scale):
                            on_wall = True
                            wall_data = wall
                            break
                    
                    if on_wall:
                        # Find associated text/label
                        label = None
                        component_code = None
                        for text in texts:
                            text_pos = text["position"]
                            # Check if text is near door
                            if distance(text_pos, position) < 100 * scale:
                                label = text["text"]
                                comp_type = find_component_type_from_text(label)
                                if comp_type and comp_type == "door":
                                    component_code = find_component_code_from_text(label)
                                    break
                        
                        # If no specific door code found, use default
                        if not component_code:
                            component_code = "DE01"  # Default door code
                        
                        # Determine swing direction based on arc
                        swing_direction = "unknown"
                        if arc["p1"]["x"] < arc["p3"]["x"]:
                            swing_direction = "right"
                        else:
                            swing_direction = "left"
                        
                        components.append({
                            "type": "door",
                            "label_code": component_code,
                            "label_type": "component",
                            "label_nl": "Deur_enkel",
                            "label_en": "Door_single",
                            "position": position,
                            "width_m": round(width, 2),
                            "swing_direction": swing_direction,
                            "wall_reference": wall_data["label_code"] if wall_data else None,
                            "confidence": 1.0,
                            "reason": "Arc+line door geometry",
                            "properties": {
                                "arc": arc,
                                "line": line
                            }
                        })
    
    # Step 2: Detect sliding doors (double parallel lines) - Rule 5.4
    logger.info("Detecting sliding doors...")
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i >= j:
                continue
            
            # Check if lines are parallel and close together
            if (_is_parallel_lines(line1, line2) and 
                _points_close(line1["p1"], line2["p1"], POINT_TOLERANCE / scale) and 
                _points_close(line1["p2"], line2["p2"], POINT_TOLERANCE / scale)):
                
                # Calculate door width
                width = distance(line1["p1"], line1["p2"]) * scale
                
                # Check if width is within door range
                if DOOR_WIDTH_MIN <= width <= DOOR_WIDTH_MAX * 1.5:  # Sliding doors can be wider
                    # Determine door position (midpoint between the lines)
                    position = {
                        "x": (line1["p1"]["x"] + line1["p2"]["x"] + line2["p1"]["x"] + line2["p2"]["x"]) / 4,
                        "y": (line1["p1"]["y"] + line1["p2"]["y"] + line2["p1"]["y"] + line2["p2"]["y"]) / 4
                    }
                    
                    # Check if on wall
                    on_wall = False
                    wall_data = None
                    for wall in walls_dict:
                        if is_point_on_wall(position, Wall(**wall), POINT_TOLERANCE / scale):
                            on_wall = True
                            wall_data = wall
                            break
                    
                    if on_wall:
                        # Find associated text/label
                        label = None
                        component_code = None
                        for text in texts:
                            text_pos = text["position"]
                            # Check if text is near door
                            if distance(text_pos, position) < 100 * scale:
                                label = text["text"]
                                comp_type = find_component_type_from_text(label)
                                if comp_type and comp_type == "sliding_door":
                                    component_code = find_component_code_from_text(label)
                                    break
                        
                        # If no specific door code found, use default
                        if not component_code:
                            component_code = "DE03"  # Default sliding door code
                        
                        components.append({
                            "type": "sliding_door",
                            "label_code": component_code,
                            "label_type": "component",
                            "label_nl": "Schuifdeur",
                            "label_en": "Door_sliding",
                            "position": position,
                            "width_m": round(width, 2),
                            "wall_reference": wall_data["label_code"] if wall_data else None,
                            "confidence": 0.9,
                            "reason": "Double parallel lines for sliding door",
                            "properties": {
                                "line1": line1,
                                "line2": line2
                            }
                        })
    
    # Step 3: Detect windows (rectangles in walls) - Rule 5.4
    logger.info("Detecting windows...")
    for rect in rectangles:
        if "rect" not in rect:
            continue
            
        r = rect["rect"]
        width = r["width"] * scale
        height = r["height"] * scale
        
        # Window dimensions should be reasonable
        if WINDOW_WIDTH_MIN <= width <= WINDOW_WIDTH_MAX and height >= 0.5:
            position = {
                "x": (r["x0"] + r["x1"]) / 2, 
                "y": (r["y0"] + r["y1"]) / 2
            }
            
            # Check if window is on a wall
            on_wall = False
            wall_data = None
            for wall in walls_dict:
                if is_point_on_wall(position, Wall(**wall), POINT_TOLERANCE / scale):
                    on_wall = True
                    wall_data = wall
                    break
            
            if on_wall:
                # Find associated text/label
                label = None
                component_code = None
                for text in texts:
                    text_pos = text["position"]
                    # Check if text is near window
                    if distance(text_pos, position) < 100 * scale:
                        label = text["text"]
                        comp_type = find_component_type_from_text(label)
                        if comp_type and comp_type == "window":
                            component_code = find_component_code_from_text(label)
                            break
                
                # If no specific window code found, use default
                if not component_code:
                    component_code = "RW01"  # Default window code
                
                components.append({
                    "type": "window",
                    "label_code": component_code,
                    "label_type": "component",
                    "label_nl": "Raam_enkel",
                    "label_en": "Window_single",
                    "position": position,
                    "width_m": round(width, 2),
                    "height_m": round(height, 2),
                    "wall_reference": wall_data["label_code"] if wall_data else None,
                    "confidence": 0.9,
                    "reason": "Rectangle in wall indicating window",
                    "properties": {
                        "rect": rect
                    }
                })
    
    # Step 4: Detect stairs (parallel lines with steps) - Rule 5.5
    logger.info("Detecting stairs...")
    # Group parallel lines that could form stairs
    stair_candidates = []
    processed_lines = set()
    
    for i, line1 in enumerate(lines):
        if i in processed_lines:
            continue
            
        parallel_group = [line1]
        for j, line2 in enumerate(lines):
            if j in processed_lines or i == j:
                continue
                
            if _is_parallel_lines(line1, line2):
                parallel_group.append(line2)
                processed_lines.add(j)
        
        if len(parallel_group) >= 3:  # At least 3 parallel lines could form stairs
            stair_candidates.append(parallel_group)
    
    # Process stair candidates
    for group in stair_candidates:
        # Sort lines by position
        if len(group[0]["p1"]) > 0 and "y" in group[0]["p1"]:
            # Sort horizontally or vertically based on orientation
            if abs(group[0]["p1"]["x"] - group[0]["p2"]["x"]) > abs(group[0]["p1"]["y"] - group[0]["p2"]["y"]):
                # Horizontal stairs
                group.sort(key=lambda line: line["p1"]["y"])
            else:
                # Vertical stairs
                group.sort(key=lambda line: line["p1"]["x"])
        
        # Check if spacing between lines is regular (steps)
        if len(group) >= 3:
            # Calculate average step size
            step_sizes = []
            for i in range(1, len(group)):
                if "y" in group[i]["p1"] and "y" in group[i-1]["p1"]:
                    if abs(group[0]["p1"]["x"] - group[0]["p2"]["x"]) > abs(group[0]["p1"]["y"] - group[0]["p2"]["y"]):
                        # Horizontal stairs
                        step_sizes.append(abs(group[i]["p1"]["y"] - group[i-1]["p1"]["y"]))
                    else:
                        # Vertical stairs
                        step_sizes.append(abs(group[i]["p1"]["x"] - group[i-1]["p1"]["x"]))
            
            if step_sizes:
                avg_step = sum(step_sizes) / len(step_sizes)
                step_m = avg_step * scale
                
                # Check if step size is reasonable
                if 0.15 <= step_m <= STAIRS_RISER_MAX + STAIRS_TREAD_MIN:
                    # Determine stair position (center of the stair)
                    x_coords = []
                    y_coords = []
                    for line in group:
                        x_coords.extend([line["p1"]["x"], line["p2"]["x"]])
                        y_coords.extend([line["p1"]["y"], line["p2"]["y"]])
                    
                    position = {
                        "x": sum(x_coords) / len(x_coords),
                        "y": sum(y_coords) / len(y_coords)
                    }
                    
                    # Calculate stair width
                    stair_width = distance(group[0]["p1"], group[0]["p2"]) * scale
                    
                    # Find associated text/label
                    label = None
                    component_code = None
                    for text in texts:
                        text_pos = text["position"]
                        # Check if text is near stairs
                        if distance(text_pos, position) < 150 * scale:
                            label = text["text"]
                            comp_type = find_component_type_from_text(label)
                            if comp_type and comp_type == "stairs":
                                component_code = find_component_code_from_text(label)
                                break
                    
                    # If no specific stairs code found, use default
                    if not component_code:
                        component_code = "TR01"  # Default stairs code
                    
                    if STAIRS_WIDTH_MIN <= stair_width <= STAIRS_WIDTH_MAX:
                        components.append({
                            "type": "stairs",
                            "label_code": component_code,
                            "label_type": "component",
                            "label_nl": "Trap",
                            "label_en": "Stairs",
                            "position": position,
                            "width_m": round(stair_width, 2),
                            "step_count": len(group) - 1,
                            "step_size_m": round(step_m, 2),
                            "confidence": 0.8,
                            "reason": f"Parallel lines forming {len(group)-1} steps",
                            "properties": {
                                "steps": group
                            }
                        })
    
    # Step 5: Detect other components from text references
    logger.info("Detecting components from text references...")
    for text in texts:
        comp_type = find_component_type_from_text(text["text"])
        if comp_type and comp_type not in ["door", "sliding_door", "window", "stairs"]:
            # We've already handled these types above
            component_code = find_component_code_from_text(text["text"])
            
            if not component_code:
                # Assign default code based on type
                if comp_type == "balcony":
                    component_code = "BA01"
                elif comp_type == "plinth":
                    component_code = "PL01"
                elif comp_type == "opening":
                    component_code = "OP01"
                else:
                    continue  # Skip if no code and unknown type
            
            # Check if position is on/near any walls
            position = text["position"]
            on_wall = False
            wall_data = None
            for wall in walls_dict:
                if is_point_on_wall(position, Wall(**wall), 3 * POINT_TOLERANCE / scale):  # Use larger tolerance for text
                    on_wall = True
                    wall_data = wall
                    break
            
            # Create component with appropriate labels
            label_nl = ""
            label_en = ""
            if comp_type == "balcony":
                label_nl = "Balkon"
                label_en = "Balcony"
            elif comp_type == "plinth":
                label_nl = "Plint"
                label_en = "Plinth"
            elif comp_type == "opening":
                label_nl = "Opening"
                label_en = "Opening"
            
            if label_nl and label_en:
                components.append({
                    "type": comp_type,
                    "label_code": component_code,
                    "label_type": "component",
                    "label_nl": label_nl,
                    "label_en": label_en,
                    "position": position,
                    "wall_reference": wall_data["label_code"] if wall_data else None,
                    "confidence": 0.7,
                    "reason": f"Component detected from text reference: {text['text']}",
                    "properties": {
                        "text": text
                    }
                })
    
    if not components:
        logger.warning(f"No components detected on page {page_data.page_number}")
        return [{
            "type": "unknown", 
            "reason": "No components detected", 
            "confidence": 0.0
        }]
    
    logger.info(f"Detected {len(components)} components")
    return components

def _points_close(p1: dict, p2: dict, tolerance: float) -> bool:
    """Check if two points are close together"""
    if not p1 or not p2:
        return False
    return distance(p1, p2) < tolerance

def _is_parallel_lines(line1: dict, line2: dict, tolerance: float = 0.1) -> bool:
    """Check if two lines are parallel"""
    if not line1 or not line2 or "p1" not in line1 or "p2" not in line1 or "p1" not in line2 or "p2" not in line2:
        return False
        
    # Calculate direction vectors
    dx1 = line1["p2"]["x"] - line1["p1"]["x"]
    dy1 = line1["p2"]["y"] - line1["p1"]["y"]
    dx2 = line2["p2"]["x"] - line2["p1"]["x"]
    dy2 = line2["p2"]["y"] - line2["p1"]["y"]
    
    # Normalize vectors
    len1 = math.sqrt(dx1**2 + dy1**2)
    len2 = math.sqrt(dx2**2 + dy2**2)
    
    if len1 == 0 or len2 == 0:
        return False
    
    # Check if vectors are parallel (dot product close to 1 or -1)
    dot_product = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
    return abs(abs(dot_product) - 1) < tolerance

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Component Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect-components/": "Detect architectural components",
            "/health/": "Health check"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "component-api",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)