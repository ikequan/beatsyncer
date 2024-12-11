from scene_splitter import VideoSceneSplitter

def main():
    # Path to your input video
    video_path = "<path to video file>"
    
    # Initialize the splitter
    splitter = VideoSceneSplitter(video_path)
    
    # Split the video into scenes
    # Parameters you can adjust:
    # - threshold: Controls scene detection sensitivity (default: 27.0)
    #   Lower values (e.g., 20.0) = more scenes detected
    #   Higher values (e.g., 35.0) = fewer scenes detected
    # - min_clip_length: Minimum duration for a clip in seconds (default: 0.5)
    clips = splitter.split_video(
        threshold=25.0,  # Slightly more sensitive than default
        min_clip_length=0.5  # Minimum half-second clips
    )
    
    # Print the generated clip paths
    print("\nGenerated clips:")
    for clip in clips:
        print(f"- {clip}")

if __name__ == "__main__":
    main()