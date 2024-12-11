# BeatSyncer

A Python project that creates beat-synchronized video montages with text overlays. This tool can split videos into clips and synchronize them with music beats.

## Features

- Beat detection and synchronization with audio
- Video clip splitting based on scene detection
- Text overlay support with motivational quotes
- Customizable video output settings
- Supports various audio and video formats

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Splitting Videos into Clips

Use `split_clips.py` to split a video into smaller clips based on scene detection:

```python
python split_clips.py
```

Configure the following parameters in the script:
- `video_path`: Path to your input video
- `threshold`: Scene detection sensitivity (default: 27.0)
- `min_clip_length`: Minimum clip duration in seconds (default: 0.5)

### Creating Beat-Synchronized Videos

Use `app.py` to create a beat-synchronized video montage:

```python
python app.py
```

Configure the following parameters in the script:
- `audio_path`: Path to your audio or video file
- `output_path`: Desired output video path
- `duration_minutes`: Optional total duration limit
- `text_overlays`: List of text overlays to display
- Video settings like size and dark overlay intensity

## Configuration Options

### BeatSyncer Parameters
- Video size: (720, 1280) default
- Dark overlay: 0.5 default
- Beat detection sensitivity: 'medium' default

### Supported File Formats

Audio:
- .mp3, .wav, .m4a, .aac, .ogg, .flac

Video:
- .mp4, .avi, .mov, .mkv, .wmv, .flv

## License

This project is licensed under the MIT License - see the LICENSE file for details.
