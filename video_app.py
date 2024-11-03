import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os 

class VideoApp:
    def __init__(self, root, results_dir):
        self.root = root
        self.root.title("Video Segmenter - Previsualization")
        
        self.results_dir = results_dir
        # Video panel for displaying frames
        self.video_panel = tk.Label(self.root)
        self.video_panel.pack()
        
        # Buttons for confirming, changing, and stopping the video
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.confirm_button = tk.Button(button_frame, text="Confirm Video", command=self.confirm_video)
        self.confirm_button.pack(side=tk.LEFT, padx=5)
        
        self.change_button = tk.Button(button_frame, text="Change Video", command=self.change_video)
        self.change_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.toggle_video)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Variables for video capture and control
        self.video_path = None
        self.video_name = None # AÑADIDO DESPUES, CAMBIAR EN VIDEO_APP.PY¡
        self.cap = None
        self.playing = False
        self.paused = False
        self.update_delay = 30  # Frame update delay in milliseconds
        self.on_confirm = None
        self.output_dir = None
        
        # Start by prompting the user to select a video
        self.change_video()

    def change_video(self):
        # Prompt user to select a new video file
        new_video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        
        # Check if a valid path is selected; if not, retain the current video
        if not new_video_path:
            if not self.video_path:
                messagebox.showwarning("Warning", "No video selected. Please select a video.")
            return
        
        # Update the video path to the newly selected file
        self.video_path = new_video_path
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0] # AÑADIDO DESPUES, CAMBIAR EN VIDEO_APP.PY
        self.set_output_dir()
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
        self.stop_button.config(text="Stop")
        self.update_video()

    def update_video(self):
        if self.playing and not self.paused and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize frame and display it in Tkinter
                frame = cv2.resize(frame, (640, 360))  # Resize for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_panel.imgtk = imgtk
                self.video_panel.config(image=imgtk)
                
                # Schedule next frame update
                self.root.after(self.update_delay, self.update_video)
            else:
                # Loop video playback
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.update_video()

    def toggle_video(self):
        # Toggle between paused and playing states
        if self.paused:
            self.paused = False
            self.stop_button.config(text="Stop")
            self.update_video()  # Resume video updates
        else:
            self.paused = True
            self.stop_button.config(text="Resume")

    def confirm_video(self):
        if self.video_path:
            messagebox.showinfo("Video Confirmed", f"Video '{self.video_path}' confirmed. Proceeding...")
            self.playing = False
            self.cap.release()
            # Notify main application that video is confirmed
            if self.on_confirm:
                self.on_confirm(self.video_path, self.output_dir)

    def get_video_path(self):
        return self.video_path

    def set_output_dir(self):
        self.output_dir = os.path.join(self.results_dir, self.video_name)
