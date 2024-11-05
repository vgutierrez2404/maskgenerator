import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import os

from src.video import Video

class FrameViewer: 
    def __init__(self, root, video: Video):
        self.root = root
        self.root.title("Frame Viewer - Select Point")
        
        # Directory containing the frames
        self.video = video
        self.frames_dir = self.video.frames_path
        self.frame_index = 0  # Start with the first frame
        self.coordinates = {}  # Dictionary of points selected in each of the frames.
        
        self.frame_panel = tk.Label(self.root)
        self.frame_panel.pack()
        self.current_frame = None
        
        # Buttons associated with the frame viewer
        btn_frame = tk.Frame(self.root)
        btn_frame.pack()
        
        self.prev_button = tk.Button(btn_frame, text="Previous Frame", command=self.previous_frame)
        self.prev_button.grid(row=0, column=0)
        
        self.next_button = tk.Button(btn_frame, text="Next Frame", command=self.next_frame)
        self.next_button.grid(row=0, column=1)

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
        self.coordinates[self.frame_index] = (x, y) # coordinates should be in the frames of the video, not the frame_viewer. 
        self.video.coordinates[self.frame_index] = (x, y)
        print(f"Selected point on frame {self.frame_index}: ({x}, {y})")

    def next_frame(self):
        # Move to the next frame, if available
        self.frame_index += 1
        self.load_frame(self.frame_index)

    def previous_frame(self):
        # Move to the previous frame, if available
        if self.frame_index > 0:
            self.frame_index -= 1
            self.load_frame(self.frame_index)

    def get_coordinates(self):
        # Return the selected coordinates for each frame
        return self.coordinates
