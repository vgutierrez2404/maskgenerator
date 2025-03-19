import matplotlib.pyplot as plt  
import numpy as np 
from PIL import Image
import os 
import random 
import subprocess
import cv2

def find_frames(output_dir: str) -> list: 
    """
    Finds and returns the frames in a directory and sorts from min to max
    being assigned each number an integer in based of its position 
    in the video. 
    """

    frame_names = [
            p for p in os.listdir(output_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    return frame_names 

def slice_video(video_path, output_dir): 
    """
    This function inputs a video and slice it in frames that 
    will be used for the frame prediction later. 

    It should use a class Video where all the frames of that video are stored. 
    """
    # Generate output directory if it does not exist. 
    os.makedirs(output_dir, exist_ok=True)

    frame_names = []
    # The ffmpge commnad that will be used to slice the video. Retrieved
    # from META notebook. 
    ffmpeg_command = [
        'ffmpeg',
        '-i', video_path,
        '-q:v', '2',  # Quality factor for image
        '-start_number', '0',  # Start numbering frames from 0
        os.path.join(output_dir, 'frame_%05d.jpg')  # Output pattern for frames
    ]

    try:   
        if len(os.listdir(output_dir)) == 0:  # Slice the video in frames
            subprocess.run(ffmpeg_command, check=True)
            print(f"Frames extracted to {output_dir} successfully. ")

        frame_names = find_frames(output_dir)
        print(f"Found {len(frame_names)} frames in {output_dir} directory.")
        
        return frame_names 
    
    except subprocess.CalledProcessError as e:
        print("An error occurred while running ffmpeg:", e)  
        return frame_names 

def show_mask(mask, ax, obj_id=None, random_color=False, black_mask=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif black_mask:
        color = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_mask_on_frame(ann_frame_idx, video_dir, frame_names, points, labels, out_mask_logits,out_obj_ids ): 

    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

def add_mask_and_save_image(masks_path: str, image:Image, mask:np.array, out_frame_idx:int) -> None:
    """
    For each frame, takes the mask obtained by the model and applies it to the image 
    and saves it in the masks_path directory.  
    """

    h, w = mask.shape[-2:]
    mask_image = (mask.reshape(h, w, 1)*255).astype(np.uint8)
    mask_image = cv2.bitwise_not(mask_image)
    res_image = cv2.bitwise_and(np.asarray(image),np.asarray(image), mask=mask_image)

    masked_image = Image.fromarray(res_image)
    masked_image.save(os.path.join(masks_path, f"frame_{out_frame_idx:05d}.jpg")) # this is resetting the index of the frames!! CARE

def check_for_preprocesed_frames(): 
    """ before loading a video to the app check if some other application 
    has preprocessed the frames (adding metadata for example)
    
    The idea of using this for DEQ probably requieres that the metadata has not 
    been added so it should be added in the middle of the pipeline of processing. 

    returns: 
        - bool: true if the folder exists and has frames. 
    """
    exist_processed_frames = False
    # we check if the folder selected_frames exists and contains frames. 
    if os.path.exists("./selected_frames") and (len(os.listdir("./selected_frames")) > 0):
        exist_processed_frames = True   
        frames_list = os.listdir("./selected_frames")
        number_of_frames = len(frames_list)
        print(f"Found {number_of_frames} frames in the directory")
        
        # https://exiftool.org/examples.html 
        # for checking if it has the frames we select a random sample of 5 frames and 
        # check if all of them have metadata inside with exiftool. 
        # if they do, we return true.

        random_frames = random.sample(frames_list, 5)
        for frame in random_frames: 
            check_metadata_cdm = "exiftool {}".format(frame)
            metadata = subprocess.run(check_metadata_cdm, shell=True, stdout=subprocess.PIPE)
            if not metadata: 
                exist_processed_frames = False
                break 

    return exist_processed_frames

    