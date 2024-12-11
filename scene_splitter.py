import os
from typing import List, Tuple
from moviepy.editor import VideoFileClip
from scenedetect import detect, ContentDetector

class VideoSceneSplitter:
    def __init__(self, video_path: str):
        """
        Initialize the VideoSceneSplitter with a video file path.
        
        Args:
            video_path (str): Path to the input video file
        """
        self.video_path = video_path
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clips')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

    def detect_scenes(self, threshold: float = 27.0) -> List[Tuple[float, float]]:
        """
        Detect scenes in the video using content-aware scene detection.
        
        Args:
            threshold (float): Threshold for scene detection sensitivity (default: 27.0)
                             Lower values will detect more scenes
        
        Returns:
            List[Tuple[float, float]]: List of (start_time, end_time) tuples for each scene
        """
        print("Detecting scenes...")
        
        # Detect scenes using the newer API
        scene_list = detect(self.video_path, ContentDetector(threshold=threshold))
        
        # Convert scene list to time ranges
        scenes = []
        for scene in scene_list:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scenes.append((start_time, end_time))

        print(f"Detected {len(scenes)} scenes")
        return scenes

    def split_video(self, threshold: float = 27.0, min_clip_length: float = 0.5) -> List[str]:
        """
        Split the video into separate clips based on scene detection.
        
        Args:
            threshold (float): Threshold for scene detection sensitivity
            min_clip_length (float): Minimum length for a clip in seconds
        
        Returns:
            List[str]: List of paths to the generated clip files
        """
        print(f"Processing video: {self.video_path}")
        scenes = self.detect_scenes(threshold)
        
        # Filter out scenes that are too short
        scenes = [(start, end) for start, end in scenes if end - start >= min_clip_length]
        print(f"Found {len(scenes)} scenes after filtering (min length: {min_clip_length}s)")

        # Create output filename template
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        clip_files = []

        # Load the video
        print("Loading video...")
        video = VideoFileClip(self.video_path)

        # Split video into scenes
        print("Splitting video into scenes...")
        for i, (start_time, end_time) in enumerate(scenes, 1):
            try:
                # Extract the subclip
                scene_clip = video.subclip(start_time, end_time)
                
                # Generate output path
                output_path = os.path.join(self.output_dir, f"{base_name}_scene_{i:03d}.mp4")
                
                # Write the clip
                print(f"Writing scene {i}/{len(scenes)}: {start_time:.1f}s - {end_time:.1f}s")
                scene_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True
                )
                
                clip_files.append(output_path)
                
            except Exception as e:
                print(f"Error processing scene {i}: {str(e)}")
            
        # Clean up
        video.close()

        print(f"\nSuccessfully created {len(clip_files)} clips")
        print(f"Clips saved in: {self.output_dir}")
        
        return clip_files
