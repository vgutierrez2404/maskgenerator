import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import os

class FrameViewer:
    def __init__(self, root, frames_dir):
        self.root = root
        self.root.title("Frame Viewer - Select Point")
        
        # Directory containing the frames
        self.frames_dir = frames_dir
        self.frame_index = 0  # Start with the first frame
        self.coordinates = {}  # Store selected coordinates for each frame index
        
        self.image_panel = tk.Label(self.root)
        self.image_panel.pack()
        self.current_image = None
        # Load and display the initial frame
        self.load_frame(self.frame_index)
        
        # Bind mouse click event to capture coordinates
        self.image_panel.bind("<Button-1>", self.on_click)
        
        # Navigation buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack()
        
        self.prev_button = tk.Button(btn_frame, text="Previous Frame", command=self.previous_frame)
        self.prev_button.grid(row=0, column=0)
        
        self.next_button = tk.Button(btn_frame, text="Next Frame", command=self.next_frame)
        self.next_button.grid(row=0, column=1)
        
    def load_frame(self, index):
        # Load a specific frame by index
        frame_path = os.path.join(self.frames_dir, f"{index:05d}.jpg")
        
        if not os.path.exists(frame_path):
            messagebox.showerror("Error", f"No frame found at {frame_path}")
            return
        
        # Load the frame and convert it for Tkinter display
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame: {frame_path}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color for display
        frame = cv2.resize(frame, (640, 360))  # Resize for display
        
        # Convert the OpenCV image (numpy array) to PIL image for Tkinter
        image = Image.fromarray(frame)
        self.current_image = ImageTk.PhotoImage(image)
        
        # Update the image panel
        self.image_panel.config(image=self.current_image)
        self.image_panel.image = self.current_image  # Keep a reference to prevent GC
        self.root.title(f"Frame Viewer - Frame {index}")
        
    # def load_frame(self, index):
    #     # Load a specific frame by index
    #     frame_path = os.path.join(self.frames_dir, f"{index:05d}.jpg")
        
    #     if not os.path.exists(frame_path):
    #         messagebox.showerror("Error", f"No frame found at {frame_path}")
    #         return
        
    #     # Load the frame and convert it for Tkinter display
    #     frame = cv2.imread(frame_path)
    #     if frame is None:
    #         raise ValueError(f"Failed to load frame: {frame_path}")
        
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frame = cv2.resize(frame, (640, 360))  # Resize for display
    #     image = Image.open(frame)
    #     self.current_image = ImageTk.PhotoImage(image)
        
    #     # Update the image panel
    #     self.image_panel.config(image=self.current_image)
    #     self.image_panel.image = self.current_image  # Keep a reference to prevent GC
    #     self.root.title(f"Frame Viewer - Frame {index}")

    def on_click(self, event):
        # Capture and store coordinates of the clicked point for the current frame
        x, y = event.x, event.y
        self.coordinates[self.frame_index] = (x, y)
        print(f"Selected point on frame {self.frame_index}: ({x}, {y})")
        messagebox.showinfo("Point Selected", f"Coordinates: ({x}, {y})")

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
