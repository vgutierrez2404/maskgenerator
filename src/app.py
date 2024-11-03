import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np 
import os 
from PIL import Image, ImageTk
import subprocess
import torch 
import sys

# Append the absolute path of the sam2 directory to sys.path
sys.path.insert(0, os.path.abspath("./sam2"))

# Local libraries 
from src.video_app import VideoApp
from src.frame_viewer import FrameViewer
from sam2.build_sam import build_sam2_video_predictor
from src.predictor import Predictor
import src.functions as fnc
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
        # Slice the video in case that is not sliced 
        if len(os.listdir(output_dir)) == 0: 
            subprocess.run(ffmpeg_command, check=True)
            print(f"Frames extracted to {output_dir} successfully. ")
    
        # Get all frames in the directory
        frame_names = [
            p for p in os.listdir(output_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        print(f"Found {len(frame_names)} frames in {output_dir} directory.")

    except subprocess.CalledProcessError as e:
        print("An error occurred while running ffmpeg:", e)

    return frame_names

def check_device_used(): 
    """
    Checks if the device running that runs the model has CUDA available. If not, it uses the cpu or in case of a 
    apple silicon chip, mps. 
    """

    device = None

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    return device

def on_video_confirmed(video_path, output_dir): 
    global main_root

    print(f"Video confirmed: {video_path}")
    
    # Slice the video into frames. This should only happen in the 
    # case that the video is not already sliced. 
    frame_names = slice_video(video_path, output_dir)

    # Destroy the previsualizer when the video is confirmed. 
    if main_root is not None:
        main_root.destroy()
    
    # Create a new Tkinter root window for the frame viewer
    frame_root = tk.Tk()
    frame_viewer = FrameViewer(frame_root, output_dir)  # output_dir is the frame directory
    frame_root.mainloop()   
    
    # TODO: esto hay que llevarlo a otra funcion 
    device = check_device_used()
    predictor = Predictor(device=device) # this is the initialization of the SAM predictor. 

    inference_state = predictor.init_state(video_path=output_dir) # the direction of the video is the output directory?? 
    labels = np.array([1], np.int32) 
    points = frame_viewer.get_coordinates()
    print(f"The selected coordinates in the video are: {points}")

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1 

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
    )

    fnc.show_mask_on_frame(ann_frame_idx, output_dir, frame_names, points, labels, out_mask_logits, out_obj_ids)

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
