import numpy as np 
import os 
import cv2  
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class Video: 

    def __init__(self, results_dir) -> None:
        
        self.results_dir = results_dir
        self.video_path = None 
        self.video_name = None
        self.frames_path = None # frames_path should be the same as the output directory. 
        self.cap = None 
        self.playing = False 
        self.paused = False
        self.update_delay = 30 
        self.on_confirm = None 

    
    def set_output_dir(self): 
        self.output_dir = os.path.join(self.results_dir, self.video_name)
        self.frames_path = self.output_dir # este paso creo que no hace falta. 

    def video_changed(self, video_path: str): 
        """
        Changes the video in the video viewer when the button is pressed. 
        video_path: str containing the new video selected in the viewer. 
        """
        # Update the video path of the Video object to the new selected video. 
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0] 
         
        self.set_output_dir() # now the video has an output directory where the frames will be stored. 

        # Stop current video if playing and open the new video file
        self.playing = False
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video.")
            return
        
        # Start the video preview
        self.playing = True
        self.paused = False

    def video_confirmed(self): 
        
        self.playing = False
        self.cap.release()
        # Notify main application that video is confirmed
        if self.on_confirm:
            self.on_confirm(self.video_path, self.output_dir)

    def get_video_path(self):
        """
        Returns the path where video is stored
        """
        return self.video_path

    def get_frames_path(self): 
        """
        Returns the path where frames are stored
        """
        return self.frames_path
