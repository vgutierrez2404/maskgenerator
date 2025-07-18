import tkinter as tk
from tkinter import filedialog  
import os 
from PIL import Image
import sys
import matplotlib.pyplot as plt
from tkinter import messagebox
from tqdm import tqdm 

# Append the absolute path of the sam2 directory to sys.path
sys.path.insert(0, os.path.abspath("./sam2")) # esto sigue siendo necesario?? 

# Local libraries 
from src.video_viewer import VideoViewer
from src.frame_viewer import FrameViewer
import src.utils.functions as fnc
import src.utils.preprocessing as preprocessing
from src.video import Video 
from src.inference_processor import InferenceProcessor 

# Global flag for root window
main_root = None

# Parameters 
DISPLAY_MASK_IN_IMAGES = False

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
    
    """

    video_segments, segmentation_logits = InferenceProcessor(video=video).run_inference() # run the inference. 

    frame_names = video.get_frame_names()

    # esto es un post processing de las imagenes. 
    render_segmentation(video, video_segments, frame_names)

    # render the segmentation results every few frames
    masks_path = os.path.join(video.frames_path, "masks")   
    os.makedirs(masks_path, exist_ok=True)  

    logits_path = os.path.join(video.frames_path, "segmentation_logits")
    os.makedirs(logits_path, exist_ok=True)
    fnc.save_masks_logits(segmentation_logits, logits_path) 

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
            # TODO: fnc.save_mask_as_image(masks_path, out_mask, out_frame_idx)
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
