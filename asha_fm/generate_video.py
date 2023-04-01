import moviepy.editor as mp

class GenerateVideo:
    def __init__(self, interp_imgs, frame_rate, audio_path, max_duration, video_save_path):
        self.interp_imgs = interp_imgs
        self.frame_rate = frame_rate
        self.audio_path = audio_path
        self.max_duration = max_duration
        self.video_save_path = video_save_path
    
    def load_audio(self):
        # Load the audio file and set its duration
        audio_clip = mp.AudioFileClip(self.audio_path).set_duration(self.max_duration)
        return audio_clip
    
    def create_video_with_audio(self):
        # Load the audio clip
        audio_clip = self.load_audio()
        
        # Create ImageClips from the interpolated images
        frames = [mp.ImageClip(interp_img).set_duration(1/self.frame_rate) for interp_img in self.interp_imgs]
        
        # Concatenate the ImageClips into a single video clip
        video_clip = mp.concatenate_videoclips(frames, method='chain')
        
        # Set the audio for the video clip
        final_clip = video_clip.set_audio(audio_clip)
        
        # Write the final video clip to file
        final_clip.write_videofile(self.video_save_path, fps=self.frame_rate)
