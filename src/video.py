import numpy as np 
import os 
import cv2  
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from src.functions import find_frames

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
        self.on_confirmed_frames = None
        self.frame_size = (0,0)
        # For the predictions, the coordinates will be stored in a 
        # dictionary that contains [frame, coordinate]
        self.coordinates = {} 

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
        
        # Get the dimensions of the video in case they change 
        self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    def video_confirmed(self): 
        
        self.playing = False
        self.cap.release()
        # Notify main application that video is confirmed
        if self.on_confirm:
            # self.on_confirm(self.video_path, self.frames_path)
            self.on_confirm(self)

    def on_frames_confirmed(self): 
        """
        This function is called when the frames of video are confirmed. 
        It calls the function in the main app for proceding with the 
        prediction of the images. 
        """
        if self.on_confirmed_frames: 
            self.on_confirmed_frames(self)

    def process_frame(self, frame): 

        frame = cv2.resize(frame, (640,360))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Convert from numpy array to PIL image for TKinter 
        image = ImageTk.PhotoImage(image=Image.fromarray(frame))

        return image 
    
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

    def get_coordinates(self): 

        return self.coordinates 

    def get_frame_names(self): 
        """
        Finds all the frames in the frames path of the video. 
        """
        frame_names = find_frames(self.frames_path)
        
        return frame_names
