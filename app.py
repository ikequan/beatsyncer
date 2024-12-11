from beatsyncer import BeatSyncer

audio_path = "<path to audio or video file>"
output_path = 'output_video.mp4'
         
duration_minutes = None  # Desired total duration in minutes

# Define text overlays:
text_overlays = [
    "Set, focus, achieve.",
    "Dream it, do it.",
    "Goals turn dreams real.",
    "Plan, act, succeed.",
    "Chase goals, not approval."
]

# Initialize the BeatSyncer
beat_syncer = BeatSyncer(audio_path)
beat_syncer.load_audio()
beat_syncer.detect_beat_drops(sensitivity='medium')
beat_syncer.sync_with_media_directory("clips", output_path, 
                                    duration_minutes=duration_minutes,
                                    video_size=(720,1280), 
                                    dark_overlay=0.5,
                                    text_overlays=text_overlays
                                    )

