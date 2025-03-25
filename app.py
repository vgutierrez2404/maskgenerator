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
import tempfile 
import cv2 

# Append the absolute path of the sam2 directory to sys.path
sys.path.insert(0, os.path.abspath("./sam2"))

# Local libraries 
from src.video_viewer import VideoViewer
from src.frame_viewer import FrameViewer
from sam2.build_sam import build_sam2_video_predictor
import src.utils.functions as fnc
import src.utils.preprocessing as preprocessing
from src.video import Video 

# Global flag for root window
main_root = None

# Parameters 
DISPLAY_MASK_IN_IMAGES = False

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

def on_video_confirmed(video:Video):

    global main_root

    print(f"Video confirmed: {video.get_video_path()}")

    # Destroy the previsualizer when the video is confirmed. 
    if main_root is not None:
        main_root.destroy()
        
    # check if a subfolder with the selected frames exists. 
    selected_frames_path = os.path.join(video.frames_path, "selected_frames")
    if os.path.exists(selected_frames_path) and os.listdir(selected_frames_path):
        process = messagebox.askyesno( title="Process video?", message="A selection of frames already exists. Would you like to process again the video?")
        video.selected_frames_path = selected_frames_path   

    if process or not os.path.exists(selected_frames_path): # si se ha seleccionado processar o no existe la carpeta selectedframes
        # We can add a process video frames in order to not use all the available ones. 
        # Slice the video into frames
        frame_names = fnc.slice_video(video.get_video_path(), video.get_frames_path())
        selected_frames = preprocessing.paralelize_list_processing(8, os.path.dirname(video.video_path), frame_names, 8)
        # falta meter los selected frames en alguna parte 
        video.get_selected_frames(selected_frames, selected_frames_path)


    # Create a new window for the frame_viewer. 
    main_root = tk.Tk()
    frame_viewer = FrameViewer(main_root, video, on_confirm=on_frames_confirmed)  # output_dir is the frame directory

    frame_viewer.check_confirmed_frames() # This function should be deleted in the future - no information 
    frame_viewer.video.on_confirmed_frames = on_frames_confirmed
    main_root.mainloop()   


def inference_by_batches(video:Video, batch_size:int, predictor): 
    """
    lo que debería hacer es coger todos los frames del video, separarlos en batches 
    y llamar a init_state y propagate_in_video con cada batch. Luego, coger la 
    máscara del ultimo frame del batch n y pasarsela como input al batch n+1. 
 
    """
    def batch_generator(frame_paths, batch_size):
        """Yield successive batches from frame_paths."""
        for i in range(0, len(frame_paths), batch_size):
            yield frame_paths[i:i + batch_size]

    video_segmentations = {} # dictionary containing the index of the frame and its mask. 
    last_mask = None # dictionary containing single key-value pair?? 
    for index, batch in enumerate(tqdm(batch_generator(video.get_frame_names(), batch_size))):
        with tempfile.TemporaryDirectory() as temp_dir: # create a temp directory to store the frames selected in the patch and infere them. Save the last mask. 
            for frame_path in batch: 
                frame_path = os.path.join(video.selected_frames_path, frame_path)
                os.symlink(frame_path, os.path.join(temp_dir, os.path.basename(frame_path)))
    
            # batch_inference_state = predictor.init_state(temp_dir, async_loading_frames=True) # batch tieen que ser una pseudocarpeta con los frames. 
            batch_inference_state = predictor.init_state(temp_dir,) # batch tieen que ser una pseudocarpeta con los frames. 

            frame_idx = fnc.get_frame_idx(batch[0])
            # For the first batch we use coordinates for the prediction of the mask 
            if last_mask is None or (isinstance(last_mask, np.ndarray) and last_mask.size == 0):   # this should mean that the batch that is being process is the first one and we have points selected from them.
                    # since we will select points from the first frame, we process it in a different way. 
                points = np.array(list(video.coordinates.values()), dtype=np.float32) # this should only have coordinates of the first frame... or 
                                                                                    # at least coordinates of images from the first batch. 
                labels = np.ones(points.shape[0], dtype=np.int32)
                ann_obj_id = 1
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state=batch_inference_state,
                                                                                   frame_idx=frame_idx, obj_id=ann_obj_id, points=points, labels=labels)

                # add to the dictionary that will have the segmentation masks of the frames and that will be used the 
                # last_key, last_value = next(reversed(my_dict.items()))
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(batch_inference_state):
                    video_segmentations[(index * batch_size) + out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    
                if video_segmentations: 
                    last_mask = next(reversed(video_segmentations[(index * batch_size) + out_frame_idx].values())).squeeze()
                else: 
                    raise ValueError("Error: No mask found for the last frame in the batch.")
                # once we store the last mask of the batch we can continue with the next batch. 
                # should we reset the state of the predictor? 
                # predictor.reset_state(batch_inference_state)
                # continue 
            
            else: # for next batche, we use the mask from the last frame of the previous batch.  

                # ahora ya no se propagan ppuntos, solo se tiene la primera máscara de la imagen del batch n+1  como input, que es la la mascara del batch n
                ann_obj_id = 1 
                frame_idx = 0
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(batch_inference_state, frame_idx, ann_obj_id, last_mask) # es este frame index?  
                
                # ahora se supone que ya está cargada la mascara en el predictor, debería tener que poder propagarse en el video. 
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(batch_inference_state):
                    video_segmentations[(index * batch_size) + out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    
                last_mask = next(reversed(video_segmentations[(index * batch_size) + out_frame_idx].values())).squeeze() # dave the last mask to propagate it to the next batch. 
                print(f"{len(video_segmentations)} segmentations found")
            #should reset state after each batch? This is the last thing that should be done. 
            predictor.reset_state(batch_inference_state)
        print(f"Processing next batch {index + 1 }\n")  

    return video_segmentations      

def normal_inference(video:Video, predictor): 
    """
    As done in the original sam demo notebook

    Return: 
        - video_segments: dictionary containing the index of the frame and its mask. 
    """
    inference_state = predictor.init_state(video_path=video.selected_frames_path, async_loading_frames=True)  # async_loading_frames: introduce en memoria los frames de forma asyncrona mientras hace inferencia con vide_prediction. 

    points = np.array(list(video.coordinates.values()), dtype=np.float32)
    
    labels = np.ones(points.shape[0], dtype=np.int32)
    frame_index = 0
    ann_obj_id = 1 # give a unique id to each object we interact with (it can be any integers). A single id for each object to track in the prediction. 

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=frame_index, obj_id=ann_obj_id, points=points, labels=labels)
    
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments 

def render_segmentation(video:Video, video_segmentations:dict, frame_names:list): 
    """
    Post process the images and mask after the inference of the model. Add to the images
    the masks and save them. Optionally, display some of the images and masks.  
    """
    # search for mask path in the directory of frames. 
    masks_path = os.path.join(video.frames_path, "masks")   
    os.makedirs(masks_path, exist_ok=True)  

    vis_frame_stride = 30 # each vis_frame_strides plot mask and image. 
    plt.close("all") # close all previous figures.     

    for out_frame_idx in tqdm(range(0, len(frame_names))): 
        image = Image.open(os.path.join(video.video_path, frame_names[out_frame_idx]))
        
        if out_frame_idx % vis_frame_stride == 0: # visaulization as in the sam demo 
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video.video_path, frame_names[out_frame_idx])))

        for out_obj_id, out_mask in video_segmentations[out_frame_idx].items(): # add mask to each of the frames and save them. 
            fnc.add_mask_and_save_image(masks_path, image, out_mask, out_frame_idx)
            
            if out_frame_idx % vis_frame_stride == 0: 
                fnc.show_mask(out_mask, plt.gca(), obj_id=out_obj_id, black_mask=True)

def on_frames_confirmed(video:Video):
    """"
    When the frames are confirmed, we throw the prediction on the coordinates 
    of the frames passed. 
    
    TODO: this should be done in batches of frames in order to be able to infer 
    larger videos with the memory of a single gpu. We can propagate masks, so we 
    will save the last mask generated by a batch and propagate to the next batcb. 
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

    batch_size = 10
    video_segments = inference_by_batches(video , batch_size, predictor) 

    frame_names = video.get_frame_names()

    render_segmentation(video, video_segments, frame_names)
    # render the segmentation results every few frames
    masks_path = os.path.join(video.frames_path, "masks")   
    os.makedirs(masks_path, exist_ok=True)  

    vis_frame_stride = 30
    plt.close("all")    
    for out_frame_idx in tqdm(range(0, len(frame_names))): 
        image = Image.open(os.path.join(video.video_path, frame_names[out_frame_idx]))

        # visualization as in the sam demo  
        if out_frame_idx % vis_frame_stride == 0: # this does not work in the ffirst video. 
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video.video_path, frame_names[out_frame_idx])))

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            fnc.add_mask_and_save_image(masks_path, image, out_mask, out_frame_idx)
            
            if out_frame_idx % vis_frame_stride == 0: # visualization as in the sam demo 
                fnc.show_mask(out_mask, plt.gca(), obj_id=out_obj_id, black_mask=True)

    load_diff_video = messagebox.askyesno(title="Load different video?", message="Would you like to load a different video?")   
    main_root.destroy()
    if load_diff_video:
        # When eveything finishes, reset the state of the predictor in order to not deallocate the memory. 
        ## predictor.reset_state(inference_state) # Seemos like this is only needed if other video is added to the tool. 
        # if its the same video, frames are stored in cache. 
        main()
    else: 
        sys.exit(0) 

def main():
    """
    Main loop of the application. 
    """
    global main_root    
    main_root = tk.Tk()

    res_dir = "./results"
    os.makedirs("./results", exist_ok=True)
 
    # Create an empty video object with no properties. It will be later update by the videoViewer/frameViewer depend on the input. 
    video = Video(res_dir)

    selection_type = fnc.select_input_type()
    if selection_type: 
        video_viewer = VideoViewer(main_root, video)
        video_viewer.video.on_confirm = on_video_confirmed

    else: 
        # in case that we select a folder of frames, we display directly the frame viewer.
        # TODO: esto debería esta fuera de la aplicacion principal -> Clean code pls. 
        frames_path = filedialog.askdirectory() 
        video.input_type = "selected_frames"
        video.frames_path = frames_path
        video.selected_frames_path = frames_path
        video.video_path = frames_path 
        video.get_frame_size()

        frame_viewer = FrameViewer(main_root, video, on_confirm=on_frames_confirmed)
        frame_viewer.check_confirmed_frames()
        frame_viewer.video.on_confirmed_frames = on_frames_confirmed


    main_root.mainloop()

if __name__ == "__main__": 
    main()
