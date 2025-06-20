#!/usr/bin/env python3
"""
StoryboardGen - A tool for generating storyboards from scripts.

This tool uses the Gemini API to create visual storyboards from text scripts,
helping streamline the pre-production process for video and animation projects.
"""

import os
import json
import logging
import re
import base64
import uuid
import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import tempfile

# Third-party imports
import requests
from PIL import Image, ImageDraw, ImageFont
import firebase_admin
from firebase_admin import credentials, firestore, storage
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('storyboard_gen')

@dataclass
class StoryboardFrame:
    """Represents a single frame in a storyboard."""
    frame_id: str
    scene_id: str
    description: str
    image_path: str
    timestamp: Optional[str] = None
    shot_type: Optional[str] = None
    camera_movement: Optional[str] = None
    characters: Optional[List[str]] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert the storyboard frame to a dictionary."""
        return {
            'frame_id': self.frame_id,
            'scene_id': self.scene_id,
            'description': self.description,
            'image_path': self.image_path,
            'timestamp': self.timestamp,
            'shot_type': self.shot_type,
            'camera_movement': self.camera_movement,
            'characters': self.characters,
            'notes': self.notes,
            'created_at': datetime.datetime.now().isoformat()
        }

class StoryboardGen:
    """
    Tool for generating storyboards from scripts.
    
    Uses the Gemini API to create visual storyboards from text scripts,
    helping streamline the pre-production process.
    """
    
    # Shot type definitions for prompting
    SHOT_TYPES = {
        "ECU": "Extreme Close-Up - Shows a small detail like eyes or hands",
        "CU": "Close-Up - Shows a character's face",
        "MCU": "Medium Close-Up - Shows character from chest up",
        "MS": "Medium Shot - Shows character from waist up",
        "MLS": "Medium Long Shot - Shows character from knees up",
        "LS": "Long Shot - Shows the full character",
        "ELS": "Extreme Long Shot - Shows character in environment",
        "WS": "Wide Shot - Shows the entire setting",
        "OTS": "Over-The-Shoulder - Shot from behind a character looking at another subject",
        "POV": "Point of View - Shot from character's perspective",
        "AERIAL": "Aerial Shot - Shot from high above the scene",
        "LOW": "Low Angle - Shot from below looking up",
        "HIGH": "High Angle - Shot from above looking down",
        "DUTCH": "Dutch Angle - Camera is tilted for disorienting effect"
    }
    
    # Camera movement definitions for prompting
    CAMERA_MOVEMENTS = {
        "STATIC": "Static - Camera doesn't move",
        "PAN": "Pan - Camera rotates horizontally",
        "TILT": "Tilt - Camera rotates vertically",
        "DOLLY": "Dolly - Camera moves forward or backward",
        "TRUCK": "Truck - Camera moves horizontally parallel to subject",
        "CRANE": "Crane - Camera moves up or down",
        "HANDHELD": "Handheld - Camera has slight shake for realism",
        "STEADICAM": "Steadicam - Smooth following movement",
        "ZOOM": "Zoom - Camera lens zooms in or out"
    }
    
    def __init__(self, project_id: str, api_key: str = None, use_firebase: bool = True):
        """
        Initialize the StoryboardGen.
        
        Args:
            project_id: Identifier for the project being worked on
            api_key: Gemini API key (optional if using environment variable)
            use_firebase: Whether to use Firebase for storage (default: True)
        """
        self.project_id = project_id
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Provide as parameter or set GEMINI_API_KEY environment variable.")
            
        self.use_firebase = use_firebase
        self.frames = []
        self.script_text = ""
        self.script_scenes = []
        
        # Set up working directory
        self.work_dir = os.path.join(tempfile.gettempdir(), f"storyboard_{project_id}")
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Initialize APIs
        try:
            self._init_gemini_api()
            if use_firebase:
                self._init_firebase()
            logger.info(f"StoryboardGen initialized for project {project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize APIs: {e}")
            raise
            
    def _init_gemini_api(self):
        """Initialize the Gemini API client."""
        genai.configure(api_key=self.api_key)
        
        # Set up the Gemini model for text processing
        self.text_model = genai.GenerativeModel('gemini-pro')
        
        # Set up the Gemini model for image generation
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
        
        logger.debug("Gemini API initialized")
        
    def _init_firebase(self):
        """Initialize Firebase connection."""
        try:
            # Check if already initialized
            firebase_admin.get_app()
        except ValueError:
            # Initialize with default credentials
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {
                'storageBucket': f"{os.environ.get('FIREBASE_PROJECT_ID')}.appspot.com"
            })
        
        self.db = firestore.client()
        self.project_ref = self.db.collection('projects').document(self.project_id)
        self.storyboard_ref = self.project_ref.collection('storyboards')
        self.bucket = storage.bucket()
        logger.debug("Firebase initialized")
        
    def load_script(self, script_path: str = None, script_text: str = None):
        """
        Load a script from a file or text string.
        
        Args:
            script_path: Path to the script file (optional)
            script_text: Script text string (optional)
            
        Note: Provide either script_path or script_text, not both.
        """
        if script_path and script_text:
            raise ValueError("Provide either script_path or script_text, not both")
            
        if script_path:
            with open(script_path, 'r') as f:
                self.script_text = f.read()
        elif script_text:
            self.script_text = script_text
        else:
            raise ValueError("Either script_path or script_text must be provided")
            
        logger.info("Script loaded successfully")
        
        # Parse script into scenes
        self._parse_script()
        
    def _parse_script(self):
        """Parse the script text into scenes."""
        # Simple scene parser - looks for scene headings
        # In a real implementation, this would use a proper screenplay parser
        
        scene_pattern = re.compile(r'(INT\.|EXT\.)\s+(.+?)(?=\n\n|\Z)', re.DOTALL)
        scenes = scene_pattern.findall(self.script_text)
        
        self.script_scenes = []
        
        if not scenes:
            # If no formal scene headings, try to split by double newlines
            chunks = self.script_text.split("\n\n")
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    self.script_scenes.append({
                        'scene_id': f"scene_{i+1}",
                        'heading': f"Scene {i+1}",
                        'content': chunk.strip()
                    })
        else:
            # Process formal screenplay format
            current_position = 0
            for i, (location_type, scene_content) in enumerate(scenes):
                scene_heading = f"{location_type} {scene_content.split('(')[0].strip()}"
                match_obj = scene_pattern.search(self.script_text, current_position)
                if match_obj:
                    start = match_obj.start()
                    end = match_obj.end()
                    
                    # Find the end of this scene (start of next scene or end of script)
                    if i < len(scenes) - 1:
                        next_match = scene_pattern.search(self.script_text, end)
                        if next_match:
                            scene_end = next_match.start()
                        else:
                            scene_end = len(self.script_text)
                    else:
                        scene_end = len(self.script_text)
                    
                    scene_content = self.script_text[start:scene_end].strip()
                    current_position = end
                    
                    self.script_scenes.append({
                        'scene_id': f"scene_{i+1}",
                        'heading': scene_heading,
                        'content': scene_content
                    })
        
        logger.info(f"Script parsed into {len(self.script_scenes)} scenes")
        
    def generate_storyboard(self, output_dir: str = None):
        """
        Generate storyboard frames from the loaded script.
        
        Args:
            output_dir: Directory to save generated images (optional)
            
        Returns:
            List of StoryboardFrame objects
        """
        if not self.script_scenes:
            raise ValueError("No script loaded or script parsing failed")
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self.work_dir
            
        self.frames = []
        
        for scene in self.script_scenes:
            logger.info(f"Generating storyboard for {scene['heading']}")
            
            # Step 1: Analyze scene to determine shot breakdown
            shots = self._analyze_scene_for_shots(scene)
            
            # Step 2: Generate storyboard frames for each shot
            scene_frames = self._generate_frames_for_shots(scene, shots, output_dir)
            
            # Add to frames collection
            self.frames.extend(scene_frames)
            
            # Save to Firebase if enabled
            if self.use_firebase:
                self._save_frames_to_firebase(scene_frames)
                
        logger.info(f"Storyboard generation complete with {len(self.frames)} frames")
        return self.frames
    
    def _analyze_scene_for_shots(self, scene: Dict) -> List[Dict]:
        """
        Analyze a scene to determine shot breakdown.
        
        Args:
            scene: Scene dictionary with heading and content
            
        Returns:
            List of shot dictionaries
        """
        # Create a prompt for Gemini to analyze the scene and suggest shots
        prompt = f"""
        You are a professional storyboard artist and film director. Analyze this scene and break it down into 3-5 key shots.
        
        SCENE:
        {scene['heading']}
        
        {scene['content']}
        
        For each shot, provide:
        1. A brief description of what happens in the shot
        2. The shot type (choose from: {', '.join(self.SHOT_TYPES.keys())})
        3. Camera movement (choose from: {', '.join(self.CAMERA_MOVEMENTS.keys())})
        4. Any characters in the shot
        5. Any important visual elements to include
        
        Format your response as a JSON array of shot objects with the following properties:
        [
            {{
                "description": "Description of the shot",
                "shot_type": "One of the shot types listed above",
                "camera_movement": "One of the camera movements listed above",
                "characters": ["Character1", "Character2"],
                "visual_elements": ["Element1", "Element2"]
            }}
        ]
        
        ONLY respond with the JSON array, nothing else.
        """
        
        try:
            response = self.text_model.generate_content(prompt)
            result = response.text
            
            # Extract JSON from response
            try:
                # Clean the response in case it's not properly formatted
                result = result.replace('```json', '').replace('```', '').strip()
                shots = json.loads(result)
                logger.debug(f"Successfully parsed {len(shots)} shots for {scene['heading']}")
                return shots
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Gemini response: {e}")
                logger.debug(f"Raw response: {result}")
                # Fallback to a simple default shot
                return [{
                    "description": f"Establishing shot for {scene['heading']}",
                    "shot_type": "WS",
                    "camera_movement": "STATIC",
                    "characters": [],
                    "visual_elements": []
                }]
                
        except Exception as e:
            logger.error(f"Error generating shot breakdown: {e}")
            # Fallback to a simple default shot
            return [{
                "description": f"Establishing shot for {scene['heading']}",
                "shot_type": "WS",
                "camera_movement": "STATIC",
                "characters": [],
                "visual_elements": []
            }]
    
    def _generate_frames_for_shots(self, scene: Dict, shots: List[Dict], output_dir: str) -> List[StoryboardFrame]:
        """
        Generate storyboard frames for each shot in a scene.
        
        Args:
            scene: Scene dictionary with heading and content
            shots: List of shot dictionaries
            output_dir: Directory to save generated images
            
        Returns:
            List of StoryboardFrame objects
        """
        frames = []
        
        for i, shot in enumerate(shots):
            frame_id = f"{scene['scene_id']}_shot_{i+1}"
            image_path = os.path.join(output_dir, f"{frame_id}.png")
            
            # Generate image for this shot
            try:
                self._generate_image_for_shot(scene, shot, image_path)
                
                # Create frame object
                frame = StoryboardFrame(
                    frame_id=frame_id,
                    scene_id=scene['scene_id'],
                    description=shot['description'],
                    image_path=image_path,
                    shot_type=shot['shot_type'],
                    camera_movement=shot['camera_movement'],
                    characters=shot.get('characters', []),
                    notes=f"Generated for {scene['heading']}"
                )
                
                frames.append(frame)
                logger.debug(f"Generated frame {frame_id}")
                
            except Exception as e:
                logger.error(f"Failed to generate image for shot {i+1} in {scene['heading']}: {e}")
                # Continue to next shot
                
        return frames
    
    def _generate_image_for_shot(self, scene: Dict, shot: Dict, image_path: str):
        """
        Generate an image for a storyboard shot using Gemini API.
        
        Args:
            scene: Scene dictionary with heading and content
            shot: Shot dictionary with description and metadata
            image_path: Path to save the generated image
        """
        # In a production system, this would use the Gemini image generation capabilities
        # For now, this is a simplified placeholder implementation
        
        # Create a detailed prompt for the image generation
        shot_type_desc = self.SHOT_TYPES.get(shot['shot_type'], "Medium Shot")
        camera_movement_desc = self.CAMERA_MOVEMENTS.get(shot['camera_movement'], "Static")
        
        prompt = f"""
        Create a detailed storyboard frame image for a film scene with the following specifications:
        
        Scene: {scene['heading']}
        Shot Description: {shot['description']}
        Shot Type: {shot_type_desc}
        Camera Movement: {camera_movement_desc}
        
        Characters: {', '.join(shot.get('characters', []))}
        Visual Elements: {', '.join(shot.get('visual_elements', []))}
        
        Make it look like a hand-drawn storyboard sketch in black and white. Include appropriate composition, depth, and lighting.
        """
        
        # PLACEHOLDER: In a real implementation, this would call the Gemini image generation API
        # For demonstration, we'll create a simple placeholder image with text
        
        # Create a blank image
        width, height = 640, 480
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw a frame border
        draw.rectangle([10, 10, width-10, height-10], outline='black', width=2)
        
        # Add scene information
        try:
            # Try to use a font, fall back to default if not available
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((20, 20), scene['heading'], fill='black', font=font)
        draw.text((20, 40), f"Shot: {shot['shot_type']} - {camera_movement_desc}", fill='black', font=font)
        
        # Draw lines for the description (word wrap)
        description = shot['description']
        y_position = 70
        words = description.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width = draw.textlength(test_line, font=font)
            
            if test_width < width - 40:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
                
        if current_line:
            lines.append(current_line)
            
        for line in lines:
            draw.text((20, y_position), line, fill='black', font=font)
            y_position += 20
            
        # Draw a simple composition guide based on shot type
        center_x, center_y = width // 2, height // 2
        
        if shot['shot_type'] in ['CU', 'ECU', 'MCU']:
            # Close-up shots - draw a face outline
            draw.ellipse([center_x-60, center_y-80, center_x+60, center_y+80], outline='black')
            draw.rectangle([center_x-40, center_y-20, center_x+40, center_y+40], outline='black')
        elif shot['shot_type'] in ['MS', 'MLS']:
            # Medium shots - draw a person outline
            draw.ellipse([center_x-30, center_y-120, center_x+30, center_y-60], outline='black')
            draw.rectangle([center_x-50, center_y-60, center_x+50, center_y+80], outline='black')
        elif shot['shot_type'] in ['LS', 'ELS', 'WS']:
            # Wide shots - draw a landscape
            draw.line([20, center_y+50, width-20, center_y+50], fill='black', width=2)
            draw.polygon([center_x-100, center_y+50, center_x, center_y-50, center_x+100, center_y+50], outline='black')
        elif shot['shot_type'] == 'OTS':
            # Over the shoulder - draw shoulder silhouette
            draw.polygon([20, center_y-100, 120, center_y-40, 150, center_y+150, 20, center_y+150], fill='black')
            draw.ellipse([center_x, center_y-60, center_x+100, center_y+40], outline='black')
            
        # Add characters if specified
        for i, character in enumerate(shot.get('characters', [])):
            y_pos = height - 40 - (i * 20)
            draw.text((20, y_pos), f"Character: {character}", fill='black', font=font)
            
        # Save the image
        image.save(image_path)
        logger.debug(f"Generated placeholder image at {image_path}")
        
    def _save_frames_to_firebase(self, frames: List[StoryboardFrame]):
        """
        Save storyboard frames to Firebase.
        
        Args:
            frames: List of StoryboardFrame objects
        """
        if not self.use_firebase:
            return
            
        for frame in frames:
            # Upload image to Firebase Storage
            blob_path = f"storyboards/{self.project_id}/{frame.frame_id}.png"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(frame.image_path)
            
            # Get public URL
            blob.make_public()
            public_url = blob.public_url
            
            # Update frame data with storage URL
            frame_data = frame.to_dict()
            frame_data['storage_path'] = blob_path
            frame_data['public_url'] = public_url
            
            # Save to Firestore
            self.storyboard_ref.document(frame.frame_id).set(frame_data)
            
        logger.info(f"Saved {len(frames)} frames to Firebase")
        
    def generate_storyboard_document(self, output_path: str, format: str = 'pdf'):
        """
        Generate a complete storyboard document with all frames.
        
        Args:
            output_path: Path to save the storyboard document
            format: Document format ('pdf' or 'html')
        """
        if format not in ['pdf', 'html']:
            raise ValueError("Format must be 'pdf' or 'html'")
            
        if not self.frames:
            raise ValueError("No storyboard frames generated yet")
            
        logger.info(f"Generating {format} storyboard document at {output_path}")
        
        # Placeholder - in a real implementation this would generate a proper document
        # using a PDF library or HTML template
        
        if format == 'html':
            with open(output_path, 'w') as f:
                f.write("<!DOCTYPE html>\n<html>\n<head>\n")
                f.write(f"<title>Storyboard: {self.project_id}</title>\n")
                f.write("<style>\n")
                f.write("body { font-family: Arial, sans-serif; }\n")
                f.write(".frame { border: 1px solid #ccc; margin: 20px; padding: 10px; }\n")
                f.write(".frame img { max-width: 100%; }\n")
                f.write("</style>\n</head>\n<body>\n")
                f.write(f"<h1>Storyboard: {self.project_id}</h1>\n")
                
                for frame in self.frames:
                    f.write(f"<div class='frame'>\n")
                    f.write(f"<h2>{frame.scene_id} - {frame.frame_id}</h2>\n")
                    f.write(f"<img src='{os.path.relpath(frame.image_path, os.path.dirname(output_path))}' alt='{frame.description}'>\n")
                    f.write(f"<p><strong>Description:</strong> {frame.description}</p>\n")
                    f.write(f"<p><strong>Shot Type:</strong> {frame.shot_type}</p>\n")
                    f.write(f"<p><strong>Camera Movement:</strong> {frame.camera_movement}</p>\n")
                    if frame.characters:
                        f.write(f"<p><strong>Characters:</strong> {', '.join(frame.characters)}</p>\n")
                    if frame.notes:
                        f.write(f"<p><strong>Notes:</strong> {frame.notes}</p>\n")
                    f.write("</div>\n")
                    
                f.write("</body>\n</html>")
                
        elif format == 'pdf':
            # Placeholder - in a real implementation this would use a PDF library
            with open(output_path, 'w') as f:
                f.write(f"Storyboard: {self.project_id}\n\n")
                
                for frame in self.frames:
                    f.write(f"Frame: {frame.frame_id}\n")
                    f.write(f"Scene: {frame.scene_id}\n")
                    f.write(f"Description: {frame.description}\n")
                    f.write(f"Shot Type: {frame.shot_type}\n")
                    f.write(f"Camera Movement: {frame.camera_movement}\n")
                    if frame.characters:
                        f.write(f"Characters: {', '.join(frame.characters)}\n")
                    if frame.notes:
                        f.write(f"Notes: {frame.notes}\n")
                    f.write(f"Image: {frame.image_path}\n\n")
                    
        logger.info(f"Storyboard document generated at {output_path}")
        
    def export_to_timeline_assembler(self, output_path: str = None):
        """
        Export storyboard data in a format compatible with TimelineAssembler.
        
        Args:
            output_path: Path to save the export file (optional)
            
        Returns:
            Export data dictionary
        """
        if not self.frames:
            raise ValueError("No storyboard frames generated yet")
            
        export_data = {
            "project_id": self.project_id,
            "export_time": datetime.datetime.now().isoformat(),
            "frames": [frame.to_dict() for frame in self.frames],
            "metadata": {
                "source": "StoryboardGen",
                "version": "1.0",
                "scene_count": len(set(frame.scene_id for frame in self.frames)),
                "frame_count": len(self.frames)
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Exported storyboard data to {output_path}")
            
        return export_data

if __name__ == "__main__":
    # Example usage
    api_key = os.environ.get("GEMINI_API_KEY", "your_api_key_here")
    
    generator = StoryboardGen(project_id="example_project", api_key=api_key, use_firebase=False)
    
    # Load a sample script
    sample_script = """
    INT. COFFEE SHOP - DAY
    
    JOHN sits alone at a small table, staring at his laptop. The coffee shop is busy with morning customers.
    
    SARAH enters and spots John. She walks over to his table.
    
    SARAH
    Hey, I got your message. What's so urgent?
    
    John looks up, closes his laptop.
    
    JOHN
    I found something. About the Henderson case.
    
    Sarah sits down across from him, leaning forward.
    
    SARAH
    Show me.
    
    EXT. PARKING LOT - MOMENTS LATER
    
    John and Sarah exit the coffee shop, walking quickly towards Sarah's car.
    
    JOHN
    We need to get to the office. Now.
    
    A black SUV pulls into the lot, stopping nearby. Sarah notices it.
    
    SARAH
    (whispering)
    Don't look now. I think we're being followed.
    """
    
    generator.load_script(script_text=sample_script)
    
    # Generate storyboard
    output_dir = os.path.join(os.getcwd(), "example_storyboard")
    frames = generator.generate_storyboard(output_dir=output_dir)
    
    # Generate storyboard document
    generator.generate_storyboard_document(os.path.join(output_dir, "storyboard.html"), format="html")
    
    # Export data for TimelineAssembler
    generator.export_to_timeline_assembler(os.path.join(output_dir, "storyboard_export.json"))