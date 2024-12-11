import os
import random
import librosa
import numpy as np
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
    vfx,
    ColorClip,
    TextClip
)


class BeatSyncer:
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}

    def __init__(self, audio_filename):
        # Construct the full path to the audio file relative to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(script_dir, audio_filename)
        
        # Convert to audio if needed
        if self._is_video_file(input_path):
            print(f"Converting video file '{input_path}' to audio...")
            audio_path = self._convert_to_audio(input_path)
            self.audio_path = audio_path
        elif self._is_audio_file(input_path):
            self.audio_path = input_path
        else:
            raise ValueError(f"Unsupported file format. File must be one of: {self.AUDIO_EXTENSIONS.union(self.VIDEO_EXTENSIONS)}")
        
        self.y = None  # Audio samples
        self.sr = None  # Sample rate
        self.beat_times = None
        self.beat_drop_times = None

    def _is_audio_file(self, filepath):
        """Check if the file is an audio file based on its extension."""
        return os.path.splitext(filepath)[1].lower() in self.AUDIO_EXTENSIONS

    def _is_video_file(self, filepath):
        """Check if the file is a video file based on its extension."""
        return os.path.splitext(filepath)[1].lower() in self.VIDEO_EXTENSIONS

    def _convert_to_audio(self, video_path):
        """Convert a video file to audio and return the path to the audio file."""
        try:
            # Create tmp directory in project root if it doesn't exist
            script_dir = os.path.dirname(os.path.abspath(__file__))
            tmp_dir = os.path.join(script_dir, 'tmp')
            os.makedirs(tmp_dir, exist_ok=True)
            
            # Create audio filename in tmp directory
            base_name = os.path.basename(video_path)
            audio_name = os.path.splitext(base_name)[0] + '.mp3'
            audio_path = os.path.join(tmp_dir, audio_name)
            
            # Convert video to audio
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)
            video.close()
            
            return audio_path
        except Exception as e:
            raise ValueError(f"Error converting video to audio: {str(e)}")

    def load_audio(self, duration_minutes=None):
        """
        Load the audio file using librosa, limiting to duration_minutes if provided.
        """
        try:
            self.sr = 44100  # Set a standard sample rate
            if duration_minutes is not None:
                duration_seconds = duration_minutes * 60
            else:
                duration_seconds = None  # Load full audio
            self.y, _ = librosa.load(self.audio_path, sr=self.sr, duration=duration_seconds)
            if self.y is None or len(self.y) == 0:
                raise ValueError("Audio data is empty.")
            duration = len(self.y) / self.sr
            print(f"Loaded audio file '{self.audio_path}' with duration {duration:.2f} seconds.")
        except Exception as e:
            print(f"An error occurred while loading the audio file: {e}")
            raise

    def detect_beat_drops(self, sensitivity='medium'):
        """
        Improved beat detection using onset_strength_multi and adjusted parameters.
        
        Args:
            sensitivity (str): Detection sensitivity level ('low', 'medium', or 'high').
                             - low: Less sensitive, detects only major beat drops
                             - medium: Balanced sensitivity (default)
                             - high: More sensitive, detects subtle beat changes
        """
        print("Detecting beat drops...")
        if self.y is None or len(self.y) == 0:
            raise ValueError("Audio data is empty. Cannot perform beat detection.")

        # Map sensitivity levels to parameters
        sensitivity_params = {
            'low': {
                'hop_length': 512,  # Larger hop length for less granular detection
                'std_multiplier': 1.0,  # Higher threshold for beat detection
                'min_interval': 0.05  # Longer minimum interval between beats
            },
            'medium': {
                'hop_length': 256,  # Default hop length
                'std_multiplier': 0.5,  # Default threshold
                'min_interval': 0.02  # Default minimum interval
            },
            'high': {
                'hop_length': 128,  # Smaller hop length for more granular detection
                'std_multiplier': 0.25,  # Lower threshold for beat detection
                'min_interval': 0.01  # Shorter minimum interval between beats
            }
        }

        # Validate and get sensitivity parameters
        sensitivity = sensitivity.lower()
        if sensitivity not in sensitivity_params:
            raise ValueError(f"Invalid sensitivity level. Must be one of: {list(sensitivity_params.keys())}")
        
        params = sensitivity_params[sensitivity]
        
        # Parameters for onset detection
        hop_length = params['hop_length']
        
        params.update({
            "detrend": True,
            "tightness": 100
        })

        # Compute the onset envelope using onset_strength_multi with adjusted parameters
        onset_env_multi = librosa.onset.onset_strength_multi(
            y=self.y,
            sr=self.sr,
            hop_length=params["hop_length"],
            aggregate=np.median,
            lag=1,
            max_size=1,
            detrend=params["detrend"],
            center=True,
        )

        # Sum across channels if multi-channel audio
        onset_env = onset_env_multi.mean(axis=0)

        # Beat tracking with adjusted parameters
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=params["hop_length"],
            tightness=params["tightness"],
            units='time'
        )

        # Convert tempo to scalar if it's an array
        tempo = float(np.mean(tempo)) if isinstance(tempo, np.ndarray) else float(tempo)

        # Adjust tempo if it's unusually high (e.g., over 180 BPM)
        if tempo >= 180:
            tempo /= 2
            print(f"Adjusted tempo: {tempo:.2f} BPM")
        else:
            print(f"Detected tempo: {tempo:.2f} BPM")

        self.beat_times = beats

        if len(self.beat_times) == 0:
            raise ValueError("No beats detected in the audio.")

        # Compute the difference in onset envelope with sensitivity-based threshold
        onset_diff = np.diff(onset_env)
        threshold = np.mean(onset_diff) + params['std_multiplier'] * np.std(onset_diff)
        beat_drop_indices = np.where(onset_diff > threshold)[0] + 1

        # Convert beat frames to times
        onset_times = librosa.frames_to_time(
            np.arange(len(onset_env)), sr=self.sr, hop_length=hop_length
        )

        # Map beat drop indices to times
        beat_drop_times = onset_times[beat_drop_indices]

        # Remove duplicate beat times
        self.beat_drop_times = np.unique(beat_drop_times)

        # Remove beat drops that are too close together based on sensitivity
        min_interval = params['min_interval']
        filtered_beat_drop_times = [self.beat_drop_times[0]]
        for bd_time in self.beat_drop_times[1:]:
            if bd_time - filtered_beat_drop_times[-1] >= min_interval:
                filtered_beat_drop_times.append(bd_time)
        self.beat_drop_times = np.array(filtered_beat_drop_times)

        print(f"Detected {len(self.beat_drop_times)} beat drops with {sensitivity} sensitivity.")

    def _get_media_files(self, media_dir):
        """
        Get list of media files from the directory.
        Returns a shuffled list of absolute paths to media files.
        """
        supported_extensions = ('.mp4', '.avi', '.mov', '.png', '.jpg', '.jpeg', '.gif', '.mkv')
        media_files = []
        
        # Walk through directory
        for root, _, files in os.walk(media_dir):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    full_path = os.path.join(root, file)
                    media_files.append(full_path)
        
        if not media_files:
            raise ValueError(f"No supported media files found in directory: {media_dir}")
        
        # Shuffle the media files list
        random.shuffle(media_files)
        print(f"Found and shuffled {len(media_files)} media files")
            
        return media_files

    def _calculate_text_duration(self, text):
        """
        Calculate the duration a text should be displayed based on its word count.
        
        Args:
            text (str): The text to be displayed
            
        Returns:
            float: Duration in seconds
        """
        # Count words (split by whitespace)
        word_count = len(text.split())
        
        # Base duration calculation:
        # - Minimum duration: 2 seconds
        # - Add 0.5 seconds per word
        # - Maximum duration: 8 seconds
        duration = min(max(2, word_count * 0.5), 8)
        
        return duration

    def _wrap_text(self, text, max_chars_per_line=30):
        """
        Wrap text to ensure it fits nicely on screen.
        
        Args:
            text (str): Text to wrap
            max_chars_per_line (int): Maximum characters per line
            
        Returns:
            str: Wrapped text with newlines and increased line spacing
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_length + word_length + len(current_line) <= max_chars_per_line:
                current_line.append(word)
                current_length += word_length
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(' '.join(current_line))
            
        # Add extra line spacing by using double newlines
        return '\n'.join(lines)

    def sync_with_media_directory(self, media_dir, output_path, duration_minutes=None, video_size=None, dark_overlay=0.3, 
                                text_overlays=None):
        """
        Create a video by syncing media files with beat drops.
        Args:
            media_dir: Directory containing images and videos
            output_path: Path where the output video will be saved
            duration_minutes: Desired total duration in minutes
            video_size: Tuple of (width, height) for output video
            dark_overlay: Opacity of dark overlay (0.0 to 1.0, where 1.0 is completely black)
            text_overlays: List of text strings or tuples (text, duration_seconds). 
                         If only text is provided, duration will be calculated based on word count.
        """
        print("Creating video from media files...")
        media_files = self._get_media_files(media_dir)
        
        if not media_files:
            raise ValueError(f"No media files found in directory: {media_dir}")

        # Prepare clips with dark theme
        clips = []
        
        # Set desired_total_duration
        if duration_minutes is not None:
            desired_total_duration = duration_minutes * 60  # Convert minutes to seconds
        else:
            desired_total_duration = len(self.y) / self.sr  # Use the length of the audio

        # Limit the beat_drop_times to the desired duration
        beat_drop_times = self.beat_drop_times[self.beat_drop_times <= desired_total_duration]

        if len(beat_drop_times) == 0:
            raise ValueError("No beat drops detected within the desired duration.")

        # Include the start (0) and end times to cover the entire duration
        beat_times_full = np.concatenate(([0], beat_drop_times, [desired_total_duration]))

        num_media = len(media_files)
        media_index = 0

        synced_clips = []

        # Define the desired clip length for initial blank space
        initial_clip_duration = 2  # 2 seconds for each clip in the initial blank space
        base_scale = 1.2  # Resize factor to make clips larger than the video size
        zoom_factor = 0.009  # Zoom factor for zoom effects

        # Loop over intervals between beat times to cover entire duration
        for i in range(len(beat_times_full) - 1):
            start_time = beat_times_full[i]
            end_time = beat_times_full[i + 1]
            interval_duration = end_time - start_time

            if i == 0 and interval_duration > initial_clip_duration:
                # Handle the initial blank space with multiple clips
                num_initial_clips = int(np.ceil(interval_duration / initial_clip_duration))
                for j in range(num_initial_clips):
                    clip_start_time = start_time + j * initial_clip_duration
                    clip_end_time = min(clip_start_time + initial_clip_duration, end_time)
                    clip_duration = clip_end_time - clip_start_time

                    media_file = media_files[media_index % num_media]
                    media_index += 1

                    if media_file.lower().endswith(('.mp4', '.avi', '.mov')):
                        # It's a video file
                        clip = VideoFileClip(media_file).subclip(0, clip_duration).resize(base_scale)
                        
                    else:
                        # It's an image file
                        clip = ImageClip(media_file).with_duration(clip_duration).resize(base_scale)


                    # Crop the clip to the desired video size
                    clip = clip.crop(width=video_size[0], height=video_size[1], x_center=clip.w / 2,
                                     y_center=clip.h / 2)

                    # Set the start time of the clip
                    clip = clip.with_start(clip_start_time)

                    # Apply random zoom in or zoom out effect with slower zoom
                    zoom_type = random.choice(['zoom_in', 'zoom_out', None])
                    if zoom_type == 'zoom_in':
                        # Slow down zoom in effect
                        clip = clip.fx(vfx.resize, lambda t: 1.0 + zoom_factor * (t / clip.duration))
                    elif zoom_type == 'zoom_out':
                        # Slow down zoom out effect
                        clip = clip.fx(vfx.resize, lambda t: 1.0 + zoom_factor * (1 - t / clip.duration))

                    # Add random transition to the clip
                    if synced_clips:
                        transition_type = random.choice(['crossfadein', 'fadein', None])
                        if transition_type == 'crossfadein':
                            # clip = clip.crossfadein(0.1)
                            print()
                        elif transition_type == 'fadein':
                            # clip = clip.fadein(0.1)
                            print()

                    synced_clips.append(clip)
            else:
                # Handle regular intervals between beat drops
                media_file = media_files[media_index % num_media]
                media_index += 1

                if media_file.lower().endswith(('.mp4', '.avi', '.mov')):
                    # It's a video file
                    clip = VideoFileClip(media_file).subclip(0, interval_duration).resize(base_scale)
                    
                else:
                    # It's an image file
                    clip = ImageClip(media_file).with_duration(interval_duration).resize(base_scale)
                   

                # Crop the clip to the desired video size
                clip = clip.crop(width=video_size[0], height=video_size[1], x_center=clip.w / 2, y_center=clip.h / 2)

                # Set the start time of the clip
                clip = clip.with_start(start_time)

                # Apply random zoom in or zoom out effect with slower zoom
                zoom_type = random.choice(['zoom_in', 'zoom_out', None])
                if zoom_type == 'zoom_in':
                    # Slow down zoom in effect
                    clip = clip.fx(vfx.resize, lambda t: 1 + zoom_factor * (t / clip.duration))
                elif zoom_type == 'zoom_out':
                    # Slow down zoom out effect
                    clip = clip.fx(vfx.resize, lambda t: 1 + zoom_factor * (1 - t / clip.duration))

                # If this is the last clip, apply fadeout
                if i == len(beat_times_full) - 2:
                    fade_duration = min(2, clip.duration / 2)
                    clip = clip.fadeout(fade_duration)

                synced_clips.append(clip)

        # Create the final video
        final_clip = CompositeVideoClip(synced_clips, size=video_size)

        # Create a list of all clips for final composition
        all_clips = [final_clip]

        # Add dark overlay if enabled
        if dark_overlay > 0:
            # Create a black ColorClip with the same size as the video
            black_clip = ColorClip(size=video_size, color=(0, 0, 0))
            black_clip = black_clip.with_duration(final_clip.duration)
            black_clip = black_clip.with_opacity(dark_overlay)
            all_clips.append(black_clip)

        # Add text overlays if provided
        if text_overlays:
            current_time = 0
            for text_item in text_overlays:
                # Handle both string and tuple inputs
                if isinstance(text_item, tuple):
                    text, duration = text_item
                else:
                    text = text_item
                    duration = self._calculate_text_duration(text)
                
                # Wrap text if needed
                wrapped_text = self._wrap_text(text)
                print(f"Adding text: '{wrapped_text}' with duration: {duration:.1f}s")
                
                # Create text clip with white color and nice font
                txt_clip = TextClip(wrapped_text, font_size=40, color='white', 
                                  font='Arial-Bold', bg_color="transparent",
                                  method='label', kerning=-2, interline=-1)  # 'label' method handles multiline text better
                
                # Center the text
                txt_clip = txt_clip.with_position(('center', 'center'))
                
                # Set the duration and start time
                txt_clip = txt_clip.with_duration(duration)
                txt_clip = txt_clip.with_start(current_time)
                
                # Add fade in/out effects
                txt_clip = txt_clip.crossfadein(0.5)
                txt_clip = txt_clip.crossfadeout(0.5)
                
                all_clips.append(txt_clip)
                current_time += duration

        # Create final composition with all layers
        final_clip = CompositeVideoClip(all_clips, size=video_size)

        # Set the audio of the final clip
        audio_clip = AudioFileClip(self.audio_path).subclip(0, desired_total_duration)
        # Apply audio fadeout at the end
        fade_duration = min(2, audio_clip.duration / 2)
        audio_clip = audio_clip.audio_fadeout(fade_duration)
        final_clip = final_clip.with_audio(audio_clip)
        final_clip = final_clip.with_duration(desired_total_duration)

        # Write the output video file
        final_clip.write_videofile(
            output_path, fps=30, threads=32, audio_codec="aac", preset='ultrafast'
        )
        print(f"Output video saved to '{output_path}'")
