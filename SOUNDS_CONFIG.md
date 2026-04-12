# Custom Sounds Configuration Guide

## Overview
EYESONIC AI now uses custom audio files from the `sounds/` folder instead of text-to-speech. This allows for more personalized and realistic audio alerts.

## How It Works

1. **Sound Mapping**: The system maps detected objects and people names to audio files
2. **Priority System**: Known faces in danger zone play first, then other faces, then obstacles
3. **File Fallback**: If a specific sound file doesn't exist, it plays the default sound

## Adding Custom Sounds

### Step 1: Prepare Audio Files
- Create audio files for each object/person you want
- Supported formats: `.m4a`, `.mp3`, `.wav`, `.ogg`
- Place files in the `sounds/` folder
- Example: `sounds/ayush.m4a`, `sounds/chair.m4a`

### Step 2: Update Sound Mapping
Edit `app.py` and update the `SOUND_FILES` dictionary:

```python
SOUND_FILES = {
    # Object sounds
    "person": "person.m4a",
    "chair": "chair.m4a",
    "table": "table.m4a",
    "bottle": "bottle.m4a",
    "car": "car.m4a",
    "bicycle": "bicycle.m4a",
    "dog": "dog.m4a",
    "cat": "cat.m4a",
    "bus": "bus.m4a",
    "truck": "truck.m4a",
    
    # Known people faces
    "ayush": "ayush.m4a",
    "kushneet": "kushneet.m4a",
    "sachin": "sachin.m4a",
    "sauravanand": "sauravanand.m4a",
    
    # Default for unknown sounds
    "default": "srk_Trim1.m4a"
}
```

### Step 3: Add Files to Static
Create a symlink or copy the sounds folder to static:

**Windows (PowerShell as Admin):**
```powershell
New-Item -ItemType SymbolicLink -Path "static/sounds" -Target "..\sounds" -Force
```

**Or copy the folder:**
```bash
cp -r sounds static/
```

## Object Sound Names

When you add sounds, use these exact names as file prefixes (convert to lowercase):

### Common Objects
- `person`
- `chair`
- `table`
- `bottle`
- `cup`
- `bed`
- `couch`
- `tv`
- `car`
- `bus`
- `bicycle`
- `dog`
- `cat`
- `backpack`
- `handbag`
- `suitcase`
- `train`
- `airplane`
- `truck`
- `motorcycle`
- `traffic light`
- `stop sign`
- `remote`

### Known Faces
- `ayush`
- `kushneet`
- `sachin`
- `sauravanand`

## Example Setup

1. Create audio files:
   ```
   sounds/
   ├── srk_Trim1.m4a          (default)
   ├── ayush.m4a
   ├── kushneet.m4a
   ├── person.m4a
   ├── chair.m4a
   ├── car.m4a
   └── dog.m4a
   ```

2. Update `app.py` SOUND_FILES dictionary with these entries

3. Restart the Flask app

4. When the system detects "ayush", it will play `ayush.m4a`
5. When it detects a "chair", it will play `chair.m4a`

## Default Behavior

- If a file is not mapped or doesn't exist, the system plays `srk_Trim1.m4a` (default)
- The default sound can be changed by updating the "default" entry in SOUND_FILES

## Testing

1. Start the app: `python app.py`
2. Open the web interface
3. Click **START SYSTEM**
4. Point at objects or people
5. You should hear custom sounds instead of text-to-speech

## Audio File Requirements

- **Format**: MP3, M4A, WAV, OGG (browser-compatible)
- **Duration**: 1-3 seconds recommended (shorter is better for real-time alerts)
- **Volume**: -6dB to -3dB recommended (don't use max volume)
- **Sample Rate**: 44.1kHz or higher

## Tips

- Use distinct sounds for different objects (makes it easier to understand)
- Keep sounds short to avoid lag
- Use clear, high-quality recordings
- For people, use their name spoken clearly
- For objects, use the object name or a descriptive sound

## Troubleshooting

### Sounds Don't Play
- Check browser console (F12) for errors
- Verify audio files are in `sounds/` folder
- Ensure files are mapped in SOUND_FILES
- Check browser audio permissions

### Wrong Sound Playing
- Verify the object name matches the SOUND_FILES key
- Check spelling (case-insensitive but must be exact)
- Ensure file exists in sounds folder

### Sound Delays
- Use shorter audio files (< 2 seconds)
- Ensure system resources are available
- Check network latency to server
