import tkinter as tk
import numpy as np 
import os 
from PIL import Image, ImageTk
import torch 
import sys

# Append the absolute path of the sam2 directory to sys.path
sys.path.insert(0, os.path.abspath("./sam2"))

# Local libraries 
from src.video_viewer import VideoViewer
from src.frame_viewer import FrameViewer
from sam2.build_sam import build_sam2_video_predictor
from src.predictor import Predictor
import src.functions as fnc
from src.video import Video 

# Global flag for root window
main_root = None

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

# def on_video_confirmed(video_path, output_dir): 
#     global main_root

#     print(f"Video confirmed: {video_path}")
    
#     # Slice the video into frames. This should only happen in the 
#     # case that the video is not already sliced. 
#     frame_names = fnc.slice_video(video_path, output_dir)

#     # Destroy the previsualizer when the video is confirmed. 
#     if main_root is not None:
#         main_root.destroy()
    
#     # Create a new Tkinter root window for the frame viewer
#     frame_root = tk.Tk()
#     frame_viewer = FrameViewer(frame_root, output_dir)  # output_dir is the frame directory
#     frame_root.mainloop()   

#     # TODO: esto hay que llevarlo a otra funcion - do_prediction() o algo asi. 
#     device = check_device_used()
#     predictor = Predictor(device=device) # this is the initialization of the SAM predictor. 

#     inference_state = predictor.init_state(video_path=output_dir) # the direction of the video is the output directory?? 
#     labels = np.array([1], np.int32) 
#     points = frame_viewer.get_coordinates()
#     print(f"The selected coordinates in the video are: {points}")

#     ann_frame_idx = 0  # the frame index we interact with
#     ann_obj_id = 1 

#     _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
#     )

#     fnc.show_mask_on_frame(ann_frame_idx, output_dir, frame_names, points, labels, out_mask_logits, out_obj_ids)

def on_video_confirmed(video:Video):

    global main_root

    print(f"Video confirmed: {video.get_video_path()}")

    # Destroy the previsualizer when the video is confirmed. 
    if main_root is not None:
        main_root.destroy()

    # Slice the video into frames
    frame_names = fnc.slice_video(video.get_video_path(), video.get_frames_path())

    # Create a new window for the frame_viewer. 
    frame_viewer_root = tk.Tk()
    frame_viewer = FrameViewer(frame_viewer_root, video)  # output_dir is the frame directory
    frame_viewer_root.mainloop()   


def main():
    global main_root
    main_root = tk.Tk()

    res_dir = "./results"
    os.makedirs(res_dir, exist_ok=True)

    # This code will be used before the refactor. 
    # Create an empty video and pass it to the video viewer 
    video = Video(res_dir)
    video_viewer = VideoViewer(main_root, video)
    video_viewer.video.on_confirm = on_video_confirmed

    main_root.mainloop()

if __name__ == "__main__": 
    main()
