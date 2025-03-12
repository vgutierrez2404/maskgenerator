import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import os

from src.video import Video

class FrameViewer: 
    def __init__(self, root, video: Video, on_confirm=None):
        self.root = root
        self.root.title("Frame Viewer - Select Point")
        
        # Directory containing the frames
        self.video = video
        self.frames_dir = self.video.selected_frames_path if self.video.selected_frames_path is not None else self.video.frames_path
        self.frame_index = 0  # Start with the first frame
        self.coordinates = {}  # Dictionary of points selected in each of the frames.
        
        self.frame_panel = tk.Label(self.root)
        self.frame_panel.pack()
        self.current_frame = None
        self.on_confirm = on_confirm
        
        # Buttons associated with the frame viewer
        btn_frame = tk.Frame(self.root)
        btn_frame.pack()
        
        self.prev_button = tk.Button(btn_frame, text="Previous Frame", command=self.previous_frame)
        self.prev_button.grid(row=0, column=0)
        
        self.next_button = tk.Button(btn_frame, text="Next Frame", command=self.next_frame)
        self.next_button.grid(row=0, column=1)

        self.confirm_button = tk.Button(btn_frame, text="Confirm frames", command=self.confirm_frames)
        self.confirm_button.grid(row=0, column=2)

        # Load and display the initial frame
        self.load_frame(self.frame_index)
        
        # Bind mouse click event to capture coordinates
        self.frame_panel.bind("<Button-1>", self.on_click)
        
    def load_frame(self, index):

        frame_path = os.path.join(self.frames_dir, f"{index:05d}.jpg")
        
        if not os.path.exists(frame_path):
            messagebox.showerror("Error", f"No frame found at {frame_path}")
            return
        
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame: {frame_path}")
        
        self.current_frame = self.video.process_frame(frame)

        # Update the image panel
        self.frame_panel.imgtk = self.current_frame
        self.frame_panel.config(image=self.current_frame)
        self.root.title(f"Frame Viewer - Frame {index}")
        
    def on_click(self, event):

        # Capture and store coordinates of the clicked point for the current frame
        x, y = event.x, event.y
        # self.coordinates[self.frame_index] = (x, y) # coordinates should be in the frames of the video, not the frame_viewer. 
        # self.video.coordinates[self.frame_index] = (x, y)
        # print(f"Selected point on frame {self.frame_index}: ({x}, {y})")

        # Get the actual dimensions of the image and panel
        frame_width, frame_height = self.current_frame.width(), self.current_frame.height()
        img_height, img_width = self.video.frame_size  # Assuming you store original size
        
        # Scale coordinates to original frame size
        scaled_x = int((x / frame_width) * img_width)
        scaled_y = int((y / frame_height) * img_height)

        self.coordinates[self.frame_index] = (scaled_x, scaled_y)
        self.video.coordinates[self.frame_index] = (scaled_x, scaled_y)    # Get the actual dimensions of the image and panel
     

    def next_frame(self):

        # Move to the next frame, if available 
        self.frame_index += 1 # TODO: if we have a list of selected frames, the following index is not +1. 
        self.load_frame(self.frame_index)

    def previous_frame(self):
        # Move to the previous frame, if available
        if self.frame_index > 0:
            self.frame_index -= 1
            self.load_frame(self.frame_index)

    def confirm_frames(self): 
        # Now i select the coordinates int eh video. 
        print(f"selected coordinates from frames{self.video.get_coordinates()}")
        
        if self.on_confirm: 
            self.on_confirm(self.video)

    def check_confirmed_frames(self): 
        """
        If the selected frames have been confirmed by the user, the window
        of the frame_viewer should be destroyed. 
        """
        if self.video.on_confirmed_frames: 
            self.root.destroy()   

    def get_coordinates(self):
        # Return the selected coordinates for each frame
        return self.coordinates
