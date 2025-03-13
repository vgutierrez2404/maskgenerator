import tkinter as tk
from tkinter import filedialog
import numpy as np 
import os 
from PIL import Image
import torch 
import sys
import matplotlib.pyplot as plt
from tkinter import messagebox
from tqdm import tqdm 

# Append the absolute path of the sam2 directory to sys.path
sys.path.insert(0, os.path.abspath("./sam2"))

# Local libraries 
from src.video_viewer import VideoViewer
from src.frame_viewer import FrameViewer
from sam2.build_sam import build_sam2_video_predictor
from src.predictor import Predictor
import src.functions as fnc
import src.utils.preprocessing as preprocessing
from src.video import Video 
from sam2.build_sam import build_sam2_video_predictor

######################
#
# Sources: https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb#scrollTo=1a572ea9-5b7e-479c-b30c-93c38b121131
#          https://github.com/facebookresearch/sam2#installation
#
######################

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

    # check if a subfolder with the selected frames exists. 
    selected_frames_path = os.path.join(video.frames_path, "selected_frames")
    if os.path.exists(selected_frames_path) and os.listdir(selected_frames_path):
        process = messagebox.askyesno( title="Process video?", message="A selection of frames already exists. Would you like to process again the video?")
        video.selected_frames_path = selected_frames_path   

    if process or not os.path.exists(selected_frames_path): # si se ha seleccionado processar o no existe la carpeta selectedframes
        # We can add a process video frames in order to not use all the available ones. 
        selected_frames = preprocessing.paralelize_list_processing(8, os.path.dirname(video.video_path), frame_names, 8)
        # falta meter los selected frames en alguna parte 
        video.get_selected_frames(selected_frames, selected_frames_path)


    # Create a new window for the frame_viewer. 
    frame_viewer_root = tk.Tk()
    frame_viewer = FrameViewer(frame_viewer_root, video, on_confirm=on_frames_confirmed)  # output_dir is the frame directory

    frame_viewer.check_confirmed_frames() # This function should be deleted in the future - no information 
    frame_viewer.video.on_confirmed_frames = on_frames_confirmed
    frame_viewer_root.mainloop()   


def reset_predictor_state(predictor:Predictor, inferance_state): 
    """
    Note: if you have run any previous tracking using this inference_state, please reset it first via reset_state.

    (The cell below is just for illustration; it's not needed to call reset_state here as this inference_state is 
    just freshly initialized above.)"""

    predictor.reset_state(inferance_state)



def on_frames_confirmed(video:Video):
    """"
    When the frames are confirmed, we throw the prediction on the coordinates 
    of the frames passed. 
    """
    # First we check what is the device where we are doing the predictions. 
    device = check_device_used()    
    print(f"used device{device}")

    # generate and initialize the predictor
    # predictor = Predictor(device=device) 
    checkpoints = os.path.join(os.getcwd(), "sam2/checkpoints", "sam2.1_hiera_small.pt") # video.model.model_checkpoints
    model_config = "sam2.1_hiera_s.yaml"
    config_path = os.path.join(os.getcwd(), "sam2", "sam2", "configs",  "sam2.1")
    predictor = build_sam2_video_predictor(model_config, checkpoints, device=device, config_path=config_path) # Step 1: load the video predictor 

    # que para hacer inferencia haya que pasarle el path de una carpeta de frames me parece una mierda, mejor sería pasarle los paths de los frames, pero bueno. 
    # esto hace que tenga que tener dos carpetas, una con los frames seleccionados y otra con todos los frames... 
    inference_state = predictor.init_state(video_path=video.selected_frames_path, async_loading_frames=True)   # quizás se puede activar el asynchronus_loading_frames para mejorar la eficiencia y que no se quede sin memoria 

    points = np.array(list(video.coordinates.values()), dtype=np.float32)
    
    labels = np.ones(points.shape[0], dtype=np.int32)
    # these are the indeces of the frames used in the inference
    frame_index = 0# = {int(k): v for k,v in video.coordinates.items()}
    ann_obj_id = 1

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=frame_index, obj_id=ann_obj_id, points=points, labels=labels,)

    frame_names = video.get_frame_names()
    # Lets see if this works properly
    
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_index}")
    plt.imshow(Image.open(os.path.join(os.path.dirname(video.video_path.rstrip("/")), frame_names[frame_index])))
    fnc.show_points(points, labels, plt.gca())
    fnc.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    # now i want to propagate the first mask through the entire video
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    masks_path = os.path.join(video.frames_path, "masks")   
    os.makedirs(masks_path, exist_ok=True)  

    vis_frame_stride = 30
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        # plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(os.path.dirname(video.video_path.rstrip("/")), frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            fnc.show_mask(out_mask, plt.gca(), obj_id=out_obj_id, black_mask=True)
        plt.savefig(os.path.join(masks_path, f"frame_{out_frame_idx:05d}.png"))

    for out_frame_idx in tqdm(range(0, len(frame_names))): 
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            image = Image.open(os.path.join(os.path.dirname(video.video_path.rstrip("/")), frame_names[out_frame_idx]))
            fnc.add_mask_and_save_image(masks_path, image, out_mask, out_frame_idx)
    
def main():
    global main_root    
    main_root = tk.Tk()

    res_dir = "./results"
    os.makedirs("./results", exist_ok=True)

    # This code will be used before the refactor. 
    # Create an empty video and pass it to the video viewer 
    video = Video(res_dir)
    video_viewer = VideoViewer(main_root, video)
    video_viewer.video.on_confirm = on_video_confirmed

    main_root.mainloop()

if __name__ == "__main__": 
    main()
