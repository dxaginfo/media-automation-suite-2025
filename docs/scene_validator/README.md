# SceneValidator

SceneValidator is a tool for validating scene composition and continuity across media projects. It ensures consistency in visual elements, character positioning, lighting, and other scene components.

## Overview

This tool uses a combination of computer vision and Gemini API to analyze scenes and identify inconsistencies that could disrupt viewer experience. It's particularly useful for video production, animation, and game development workflows.

## Features

- **Scene Composition Analysis**: Validates the composition of scenes against established guidelines
- **Continuity Checking**: Identifies continuity errors between sequential scenes
- **Character Positioning**: Tracks character positions across scenes
- **Lighting Consistency**: Ensures lighting remains consistent where appropriate
- **Object Tracking**: Monitors the presence and position of key objects
- **Time-of-Day Validation**: Verifies time-of-day consistency
- **Automated Reporting**: Generates detailed reports of identified issues
- **Visual Annotations**: Provides visual annotations of detected problems

## Technology Stack

- **Python**: Core implementation language
- **Gemini API**: Powers the AI analysis of scene content
- **Google Cloud Vision API**: Used for image analysis
- **TensorFlow**: Provides machine learning capabilities for specialized detection
- **Firebase**: Stores validation results and history

## Integration Points

SceneValidator integrates with:
- **StoryboardGen**: Validates storyboard consistency
- **ContinuityTracker**: Shares continuity data for comprehensive tracking
- **EnvironmentTagger**: Utilizes tagged environment elements for validation

## Documentation Index

- [Specifications](./specifications.md)
- [Setup Guide](./setup.md)
- [Usage Examples](./usage.md)
- [API Reference](./api.md)
- [Troubleshooting](./troubleshooting.md)
- [Integration Guide](./integration.md)

## Example Usage

```python
from scene_validator import SceneValidator

# Initialize the validator
validator = SceneValidator(project_id="my_project")

# Add scenes to validate
validator.add_scene("scene1.jpg", scene_id="scene_1", timestamp="00:01:30")
validator.add_scene("scene2.jpg", scene_id="scene_2", timestamp="00:01:45")

# Run validation
results = validator.validate()

# Generate report
validator.generate_report("validation_report.pdf")

# Access specific validation results
continuity_issues = results.get_continuity_issues()
composition_issues = results.get_composition_issues()

# Apply suggested fixes
validator.apply_suggested_fixes(scene_id="scene_2", fix_ids=["fix_001", "fix_003"])
```

## Sample Output

The validation report includes:

```
Scene Validation Report
Project: my_project
Date: 2025-06-20
Total Scenes: 2
Issues Found: 3

Issues:
1. Continuity Error - Character position changed (scene_1 → scene_2)
   Severity: Medium
   Description: Character "John" moved from left side to right side without transition
   Suggestion: Add transition scene or adjust character position

2. Lighting Inconsistency - Lighting direction changed (scene_1 → scene_2)
   Severity: Low
   Description: Light source direction shifted from north-east to north-west
   Suggestion: Adjust lighting in scene_2 to match scene_1

3. Object Missing - Prop disappeared (scene_1 → scene_2)
   Severity: High
   Description: Coffee cup on table present in scene_1 is missing in scene_2
   Suggestion: Add coffee cup to scene_2 or show it being removed
```