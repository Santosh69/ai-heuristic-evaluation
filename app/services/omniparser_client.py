from app.utils.transformers_patches import patch_transformers_flash_attn_check
patch_transformers_flash_attn_check()

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io
import json
from ultralytics import YOLO
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import re  # For cleaning Florence output



TYPE_MAPPING = {
    "icon":  "icon",
    "clickable_button": "button",
    "icon_button": "button",
    "submit_button": "button",
    "button": "button",
    "text_input": "input",
    "input": "input",
    "search_bar": "input",
    "heading": "heading",
    "header": "heading",
    "title": "heading",
    "h1": "heading",
    "h2": "heading",
    "h3": "heading"
}

from app.services.exceptions import InvalidInputError, OmniParserError
from app.core.config import ALLOWED_IMAGE_TYPES, MAX_IMAGE_SIZE_BYTES

logger = logging.getLogger(__name__)

class UIElement:
    """UI Element matching real Omniparser output format.
    
    Fields align with Omniparser:
    - type: element type (button, text, input, etc.)
    - bbox: [x1, y1, x2, y2] bounding box coordinates
    - interactivity: boolean indicating if element is interactive
    - content: text content of the element
    """
    
    def __init__(
        self,
        element_type: str,
        bbox: List[float],
        content: str = "",
        interactivity: bool = False
    ):
        """Initialize UIElement with Omniparser-compatible format.
        
        Args:
            element_type: Type of UI element (button, text, input, etc.)
            bbox: Bounding box as [x1, y1, x2, y2]
            content: Text content of the element
            interactivity: Whether the element is interactive
        """
        self.element_type = element_type
        self.bbox = bbox
        self.content = content
        self.interactivity = interactivity
        
        # Computed properties for convenience
        self._width = bbox[2] - bbox[0] if len(bbox) == 4 else 0
        self._height = bbox[3] - bbox[1] if len(bbox) == 4 else 0
    
    @property
    def text(self) -> str:
        """Alias for content to maintain backward compatibility."""
        return self.content
    
    @property
    def width(self) -> float:
        """Computed width from bbox."""
        return self._width
    
    @property
    def height(self) -> float:
        """Computed height from bbox."""
        return self._height
    
    @property
    def bounds(self) -> Dict[str, float]:
        """Legacy bounds format for backward compatibility."""
        return {
            "x": self.bbox[0],
            "y": self.bbox[1],
            "width": self.width,
            "height": self.height
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.element_type,
            "bbox": self.bbox,
            "content": self.content,
            "interactivity": self.interactivity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UIElement':
        """Create UIElement from Omniparser output dict."""
        return cls(
            element_type=data.get("type", data.get("element_type", "unknown")),
            bbox=data.get("bbox", [0, 0, 0, 0]),
            content=data.get("content", data.get("text", "")),
            interactivity=data.get("interactivity", data.get("interactive", False))
        )


def infer_heading_level(element: UIElement, all_text_elements: List[UIElement]) -> Optional[int]:
    """Infer heading level (1-6) based on bounding box height.
    
    Larger height indicates higher-level heading (h1 > h2 > h3, etc.).
    Uses normalization across all text elements to ensure consistency.
    
    Args:
        element: The element to classify
        all_text_elements: All text elements for normalization
    
    Returns:
        Heading level 1-6, or None if not a heading
    """
    if element.element_type not in ["text", "heading"]:
        return None
    
    # Get heights of all text elements
    heights = [e.height for e in all_text_elements if e.height > 0]
    if not heights or element.height <= 0:
        return None
    
    # Calculate percentile thresholds
    sorted_heights = sorted(heights, reverse=True)
    max_height = sorted_heights[0]
    min_height = sorted_heights[-1]
    
    # If height is below median, it's likely body text, not a heading
    median_height = sorted_heights[len(sorted_heights) // 2]
    if element.height < median_height:
        return None
    
    # Map height to heading levels using percentiles
    # Top 5% = h1, next 10% = h2, next 15% = h3, etc.
    percentile_rank = sorted_heights.index(element.height) / len(sorted_heights) if element.height in sorted_heights else 1.0
    
    if percentile_rank <= 0.05:  # Top 5%
        return 1
    elif percentile_rank <= 0.15:  # Top 15%
        return 2
    elif percentile_rank <= 0.30:  # Top 30%
        return 3
    elif percentile_rank <= 0.50:  # Top 50%
        return 4
    else:
        return None  # Body text


def calculate_height_variance(elements: List[UIElement]) -> float:
    """Calculate variance in element heights.
    
    Useful for detecting inconsistent sizing.
    
    Args:
        elements: List of UI elements
        
    Returns:
        Coefficient of variation (std dev / mean) as percentage
    """
    if len(elements) < 2:
        return 0.0
    
    heights = [e.height for e in elements if e.height > 0]
    if not heights:
        return 0.0
    
    mean_height = sum(heights) / len(heights)
    variance = sum((h - mean_height) ** 2 for h in heights) / len(heights)
    std_dev = variance ** 0.5
    
    # Return coefficient of variation as percentage
    return (std_dev / mean_height * 100) if mean_height > 0 else 0.0

class UIElementDetectionResult:
    def __init__(
        self,
        elements: List[UIElement],
        layout_hierarchy: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.elements = elements
        self.layout_hierarchy = layout_hierarchy
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "elements": [e.to_dict() for e in self.elements],
            "layout_hierarchy": self.layout_hierarchy,
            "metadata": self.metadata
        }

class OmniParserClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_loaded = False
        self.yolo_model = None       # YOLO model for element detection
        self.caption_model = None  # Florence-2 model for captioning
        self.processor = None      # Processor for Florence-2

    async def initialize(self):
        self.logger.info("Initializing OmniParser client...")
        # Loading YOLO 
        self.logger.info("Loading YOLO model...")
        self.yolo_model = YOLO("weights/icon_detect/model.pt")
        self.logger.info("✅ YOLO model loaded")

        device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        self.device = device
        self.logger.info(f"Using device: {device}")
        # Loading Florence-2 
        self.caption_model = AutoModelForCausalLM.from_pretrained(
            "weights/icon_caption_florence", 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(device)
        
        self.logger.info("✅ Florence model loaded")
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
        self.logger.info("✅ Florence processor loaded")
        self.caption_model.eval()
        self.model_loaded = True
        self.logger.info("OmniParser client initialized successfully")

    def _map_element_type(self, raw_type: str) -> str:
        return TYPE_MAPPING.get(raw_type.lower(), "unknown")

    async def detect_elements(
        self,
        image_data: bytes,
        image_url: Optional[str] = None,
        content_type: str = "image/jpeg"
    ) -> UIElementDetectionResult:
        self.logger.info("Starting UI element detection...")

        if not self.model_loaded:
            await self.initialize()

        try:
            # Validate image
            self.validate_image(image_data, content_type)

            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            self.logger.info(f"Processing image: {width}x{height}")

            # YOLO detection
            results = self.yolo_model(image)
            elements = []
            
            for result in results:
                for box in result.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get type 
                    cls_id = int(box.cls[0])
                    raw_type  = self.yolo_model.names[cls_id]
                    self.logger.info(f"YOLO detected: {raw_type} (class {cls_id})")
                    mapped_type = self._map_element_type(raw_type)

                    # FLORENCE CAPTIONING - Makes content DYNAMIC
                    element_content = ""
                    if self.caption_model is not None and self.processor is not None:
                        try:
                            # Crop the detected element
                            crop_x1 = max(0, int(x1))
                            crop_y1 = max(0, int(y1))
                            crop_x2 = min(width, int(x2))
                            crop_y2 = min(height, int(y2))
                            
                            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                                continue

                            element_crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                            self.logger.info(f"Crop format: {element_crop.mode}, size: {element_crop.size}")
                            
                            if element_crop.mode != "RGB":
                                element_crop = element_crop.convert("RGB")

                            # Skip very small elements
                            if element_crop.width >= 10 and element_crop.height >= 10:
                                # Use Florence to generate caption
                                prompt = "<CAPTION>"
                                inputs = self.processor(
                                    text=prompt, 
                                    images=element_crop, 
                                    return_tensors="pt"
                                ).to(self.device)
                                
                                # Generate caption
                                with torch.no_grad():
                                    generated_ids = self.caption_model.generate(
                                        input_ids=inputs["input_ids"],
                                        pixel_values=inputs["pixel_values"],
                                        max_new_tokens=50,
                                        num_beams=3
                                    )
                                
                                # Decode caption
                                generated_text = self.processor.batch_decode(
                                    generated_ids, 
                                    skip_special_tokens=False
                                )[0]
                                self.logger.info(f"Florence raw output: {repr(generated_text)}")
                                # Extract caption (remove tags)
                                element_content = (
                                    generated_text
                                    .replace("<s>", "")
                                    .replace("</s>", "")
                                    .replace("<CAPTION>", "")
                                    .replace("</CAPTION>", "")
                                    .replace("<pad>", "")
                                    .strip()
                                )
                                if element_content:
                                    self.logger.info(f"Caption: {element_content[:50]}")
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to caption {mapped_type}: {e}")
                            element_content = ""

                    # Create Element with DYNAMIC content from Florence
                    elements.append(UIElement(
                        element_type=mapped_type,
                        bbox=[x1, y1, x2, y2],
                        content=element_content,  
                        interactivity=mapped_type in ["button", "input", "link"]
                    ))

            layout_hierarchy = {}
            result = UIElementDetectionResult(
                elements=elements,
                layout_hierarchy=layout_hierarchy,
                metadata={
                    "width": width,
                    "height": height,
                    "total_elements": len(elements),
                    "florence_enabled": self.caption_model is not None
                }
            )

            self.logger.info(
                f"Detection complete: {len(elements)} elements, "
                f"Florence: {'ON' if self.caption_model else 'OFF'}"
            )
            return result

        except InvalidInputError:
            # Re-raise validation errors
            raise
        except Exception as e:
            self.logger.error(f"Error in detection: {str(e)}", exc_info=True)
            raise OmniParserError(
                message="Failed to detect UI elements",
                details={"error": str(e)}
            )

    def group_related_elements(self, elements: List[UIElement]) -> Dict[str, List[UIElement]]:
        grouped = {
            "buttons": [],
            "inputs": [],
            "navigation": [],
            "content": [],
            "links": []
        }

        for element in elements:
            element_type = element.element_type.lower()
            if element_type in ["button", "submit", "reset"]:
                grouped["buttons"].append(element)
            elif element_type in ["input", "textarea", "select"]:
                grouped["inputs"].append(element)
            elif element_type in ["nav", "menu", "header", "footer"]:
                grouped["navigation"].append(element)
            elif element_type in ["a", "link"]:
                grouped["links"].append(element)
            else:
                grouped["content"].append(element)

        return grouped

    def validate_image(self, image_data: bytes, content_type: str):
        """Validate image data and content type."""
        if not image_data:
            raise InvalidInputError("No image data provided")
        
        if len(image_data) > MAX_IMAGE_SIZE_BYTES:
            raise InvalidInputError(
                f"Image too large: {len(image_data)} bytes (max: {MAX_IMAGE_SIZE_BYTES})"
            )
        
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise InvalidInputError(
                f"Unsupported image type: {content_type}. Allowed: {ALLOWED_IMAGE_TYPES}"
            )
