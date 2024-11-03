import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os 
from PIL import Image, ImageTk
import subprocess

# Local libraries 
from video_app import VideoApp
from frame_viewer import FrameViewer

# Global flag for root window
main_root = None

def slice_video(video_path, output_dir): 
    os.makedirs(output_dir, exist_ok=True)

    ffmpeg_command = [
        'ffmpeg',
        '-i', video_path,
        '-q:v', '2',  # Quality factor for image
        '-start_number', '0',  # Start numbering frames from 0
        os.path.join(output_dir, '%05d.jpg')  # Output pattern for frames
    ]

    try:
        # Run the ffmpeg command
        subprocess.run(ffmpeg_command, check=True)

        # Get all frames in the directory
        frame_names = [
            p for p in os.listdir(output_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        print(f"Frames extracted to {output_dir} successfully. Found {len(frame_names)} frames")

    except subprocess.CalledProcessError as e:
        print("An error occurred while running ffmpeg:", e)

def on_video_confirmed(video_path, output_dir): 
    global main_root

    print(f"Video confirmed: {video_path}")
    
    # Slice the video into frames
    slice_video(video_path, output_dir)

    # Destroy the previsualizer when the video is confirmed. 
    if main_root is not None:
        main_root.destroy()
    
    # Create a new Tkinter root window for the frame viewer
    frame_root = tk.Tk()
    frame_viewer = FrameViewer(frame_root, output_dir)  # output_dir is the frame directory
    frame_root.mainloop()

    # Retrieve selected coordinates from the frame viewer
    selected_coordinates = frame_viewer.get_coordinates()
    print(f"The selected coordinates in the video are: {selected_coordinates}")

def main():
    global main_root
    main_root = tk.Tk()

    res_dir = "./results"
    os.makedirs(res_dir, exist_ok=True)

    # Instance of the VideoApp class. This will allow to previsualize the video. 
    app = VideoApp(main_root, res_dir)
    
    # Callback when the video is confirmed. 
    app.on_confirm = on_video_confirmed
    
    main_root.mainloop()

if __name__ == "__main__": 
    main()
