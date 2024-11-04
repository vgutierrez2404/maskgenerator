import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os 

from src.video import Video

class VideoViewer: 
    def __init__(self, root, results_dir): 
        self.root = root
        self.root.title("Video Segmenter - Previsualization")

        # Panel for displaying frames 
        self.video_panel = tk.Label(self.root)
        self.video_panel.pack()

        # Buttons associated with the window used 
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.video = Video(results_dir) # this class should be filled with the commands that are displayed below
        self.playing_video_path = None

        self.confirm_button = tk.Button(button_frame, text="Confirm Video", command=self.video.confirm_video)
        self.confirm_button.pack(side=tk.LEFT, padx=5)
        
        self.change_button = tk.Button(button_frame, text="Change Video", command=self.change_video)
        self.change_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.toggle_video)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.change_video()

    def change_video(self): 

        # Ask user to select a new video
        new_video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        self.playing_video_path = new_video_path

        # Check if a valid path is selected; if not, retain the current video
        if not new_video_path:
            if not self.video.video_path:
                messagebox.showwarning("Warning", "No video selected. Please select a video.")
            return
        
        self.video.video_changed(new_video_path)

        self.stop_button.config(text="Stop")

        self.update_video()

    def update_video(self):

        if self.video.playing and not self.video.paused and self.video.cap.isOpened():
            ret, frame = self.video.cap.read()
            if ret:
                # Resize frame and display it in Tkinter
                frame = cv2.resize(frame, (640, 360))  # Resize for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_panel.imgtk = imgtk
                self.video_panel.config(image=imgtk)
                
                # Schedule next frame update
                self.root.after(self.video.update_delay, self.update_video)

            else:
                # Loop video playback
                self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.update_video()

    def toggle_video(self): 

        # Toggle between paused and playing states
        if self.video.paused:
            self.video.paused = False
            self.stop_button.config(text="Stop")
            self.update_video()  # Resume video updates
        else:
            self.video.paused = True
            self.stop_button.config(text="Resume")

    def confirm_video(self): 
        """
        Update the view when a video is confirmed. 
        """
        if self.playing_video_path != None: 
            messagebox.showinfo("Video Confirmed", f"Video '{self.playing_video_path}' confirmed. Proceeding...")
            self.video.video_confirmed()
