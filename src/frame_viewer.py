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
        self.box = {} # Placeholder for the bounding box if needed
        
        # created a canvas instead of label to add bbox selection 
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack()

        self.current_frame = None
        self.on_confirm = on_confirm
        
        # Buttons associated with the frame viewer
        btn_frame = tk.Frame(self.root)
        btn_frame.pack()
        
        # methods for bbox 
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.is_dragging = False
        self.drag_threshold = 5  # Threhold to detect if is a bbox or a point selection 

        # self.prev_button = tk.Button(btn_frame, text="Previous Frame", command=self.previous_frame)
        self.prev_button = tk.Button(btn_frame, text="Previous Frame", command=lambda: self.change_frame(-1))
        self.prev_button.grid(row=0, column=0)
        
        # self.next_button = tk.Button(btn_frame, text="Next Frame", command=self.next_frame)
        self.next_button = tk.Button(btn_frame, text="Next Frame", command=lambda: self.change_frame(1))
        self.next_button.grid(row=0, column=1)

        self.confirm_button = tk.Button(btn_frame, text="Confirm frames", command=self.confirm_frames)
        self.confirm_button.grid(row=0, column=2)

        # Load and display the initial frame
        self.load_frame(self.frame_index)
        
        # Bind mouse click event to capture coordinates
        self.canvas.bind("<Button-1>", self.on_click)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
                
    def load_frame(self, index):

        # hacer esto cada vez que cargo un frame es un acto de terrorismo.
        def extract_number(filename):
            """
            Extract number from filename like 'frame10.png' or 'frame_10.png'
            """
            import re 
            match = re.search(r'\d+', filename)
            if match:
                return int(match.group())
            else:
                return -1  # fallback if no number found

        frame_files = [f for f in os.listdir(self.video.frames_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if frame_files:
            # frame = sorted(frame_files)[index]
            sorted_files = sorted(frame_files, key=extract_number)
            frame = sorted_files[index]
            
        frame_path = os.path.join(self.frames_dir, frame)
        
        if not os.path.exists(frame_path):
            messagebox.showerror("Error", f"No frame found at {frame_path}")
            return
        
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame: {frame_path}")
        
        self.current_frame = self.video.process_frame(frame)
    
        # Get the dimensions of the processed frame
        frame_width = self.current_frame.width()
        frame_height = self.current_frame.height()
        
        # Update canvas size to match the frame dimensions
        self.canvas.config(width=frame_width, height=frame_height)
        
        # Clear canvas and display the new frame
        self.canvas.delete("all")  
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_frame)
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
     
    def change_frame(self, direction=1):
        """
        Move to the next or previous frame based on the direction.
        
        :param direction: 1 for next frame, -1 for previous frame.
        """
        frame_names = self.video.get_frame_names()
        indexes = sorted(int(filename.split(".")[0].replace('frame', '')) for filename in frame_names)

        if self.frame_index in indexes:
            current_idx = indexes.index(self.frame_index)
            new_idx = current_idx + direction

            if 0 <= new_idx < len(indexes):  # Ensure within bounds
                self.frame_index = indexes[new_idx]
        else:
            self.frame_index = indexes[0] if direction == 1 else indexes[-1]  # Default to first or last frame

        self.load_frame(self.frame_index)  # Load the frame to the viewer


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
    
    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.is_dragging = False
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_move_press(self, event):
        dx = abs(event.x - self.start_x)
        dy = abs(event.y - self.start_y)
        if dx > self.drag_threshold or dy > self.drag_threshold:
            self.is_dragging = True
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):

        frame_width, frame_height = self.current_frame.width(), self.current_frame.height()
        img_width, img_height = self.video.frame_size  # Assuming you store original size
        
        if self.is_dragging:
            # User drew a box
            x0, y0, x1, y1 = self.canvas.coords(self.rect)
            scale_x0 = int((x0 / frame_width) * img_width)
            scale_y0 = int((y0 / frame_height) * img_height)
            scale_x1 = int((x1 / frame_width) * img_width)
            scale_y1 = int((y1 / frame_height) * img_height)
            
            self.box[self.frame_index] = (scale_x0, scale_y0, scale_x1, scale_y1)
            self.video.bounding_boxes[self.frame_index] = (scale_x0, scale_y0, scale_x1, scale_y1)
            print(f"Bounding box for frame {self.frame_index}: ({scale_x0}, {scale_y0}), ({scale_x1}, {scale_y1})")
        else:
            # if the draggig threshold is not met, the selection is trated as a point. 
            x, y = event.x, event.y
            scaled_x = int((x / frame_width) * img_width)
            scaled_y = int((y / frame_height) * img_height)

            # self.coordinates[self.frame_index] = (scaled_x, scaled_y)
            # self.video.coordinates[self.frame_index] = (scaled_x, scaled_y)
                    # Append instead of overwrite
            if self.frame_index not in self.coordinates:
                self.coordinates[self.frame_index] = []
            if self.frame_index not in self.video.coordinates:
                self.video.coordinates[self.frame_index] = []

            self.coordinates[self.frame_index].append((scaled_x, scaled_y))
            self.video.coordinates[self.frame_index].append((scaled_x, scaled_y))

            print(f"Point for frame {self.frame_index}: ({scaled_x}, {scaled_y})")

            self.canvas.delete(self.rect)  # Remove the tiny rectangle
