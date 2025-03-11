import cv2
import time 
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil 
import os 

def compute_and_compare_histograms(frame_1:str, frame_2:str): 
    """
    This function takes as input the paths of two iamges and computes the histogram
    between both 
    """
    # from https://medium.com/scrapehero/exploring-image-similarity-approaches-in-python-b8ca0a3ed5a3
    image_1 = cv2.imread(frame_1)
    image_2 = cv2.imread(frame_2)

    hist_image_1 = cv2.calcHist([image_1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_image_1[255, 255, 255] = 0 
    cv2.normalize(hist_image_1, hist_image_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_image_2 = cv2.calcHist([image_2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_image_2[255, 255, 255] = 0 
    cv2.normalize(hist_image_2, hist_image_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Compare both histograms using some metric 

    metric_val = cv2.compareHist(hist_image_1, hist_image_2, cv2.HISTCMP_CORREL)
    # print(f"Similarity Score for frames {frame_1} and {frame_2} is: ", round(metric_val, 2))

    return metric_val

def crop_frames(skip_ratio:str, frames_files:list[str]): 
    """
    Selects every x-th frame in a list of frames. 

    Inputs: 
        - skip_ratio: integer that defines the ratio of frames skipped each 
        iteration. 
        - frames_files: list of strings containing all the frames of the video 
        selected. 

    Output: 
        - selected_frames: list of frames which is a cropped version of 
        frames_files. 
    """
    if not frames_files:
        return []

    # Initialize the list to store the selected frames
    selected_frames = []

    # Iterate through the frames and select every x-th frame
    for i in range(0, len(frames_files), skip_ratio):
        selected_frames.append(frames_files[i])

    # Ensure the last frame is included if it's not already in the selected frames
    if frames_files[-1] not in selected_frames:
        selected_frames.append(frames_files[-1])

    return selected_frames

def select_desired_frames(selection_type:str, frame_files:list[str], input_folder:str, threshold:int): 
    """
    Podríamos usar varios approaches para seleccionar los frames antes de añadirles
    la informacion de EXIF. ¿Distancia? ¿Similitud entre frames: DL, metricas, histogramas, 
    features como SIFT o SURF que ya se hacen en opensfm? ¿Tiempo? ¿cada x frames? 
    Vamos a probar varios. 

    Input arguments: 
        - selection_type: the way the selection of similar frames is going to be done. 
        - input_folder: where the all the original frames are stored 
        - frame_files: list of strings containing the names of the frames stored
        in the input folder. 
    """

    if selection_type == "histogram": 
        # threshold = 0.7 -> It's passed as an argument of the function
        last_saved_frame = None 
        selected_frames = []
        
        start_time = time.time()
        for i, frame_name in enumerate(frame_files): 
            frame_path = os.path.join(input_folder, frame_name)

            if last_saved_frame is None:
                # Save the first frame unconditionally
                last_saved_frame = frame_path
                # Lo que quiero guardar es el nombre del frame, no su path. 
                selected_frames.append(frame_name)
                print(f"Saved: {frame_path}")
                continue
            
            similarity = compute_and_compare_histograms(last_saved_frame, frame_path)

            if similarity < threshold:  
                selected_frames.append(frame_name)
                print(f"Similarity between last frame{last_saved_frame} and frame {frame_name} at iteration {i} is {similarity}")
                last_saved_frame = frame_path
                print(f"Saved frame {frame_name}")

            else: 
                print(f"Skipped frame {frame_name}") 
        end_time = time.time()

        print(f"Elapsed time to select the desirable frames: {end_time - start_time}")

        return selected_frames
    
    elif selection_type == "time": 
        # este solo sera posible una vez tengamos las exif añadidas
        pass 

    elif selection_type == "distance":
        # este solo sera posible una vez tengamos las exif añadidas
        pass

    elif selection_type == "skip_frames": 

        skip_ratio = 5 # Select 1 each x frames 
        selected_frames = crop_frames(skip_ratio, frame_files)
        
        return selected_frames
    

def paralelize_list_processing(n_subgroups:int, input_folder:str, frame_files:list[str], max_workers:int=None): 
    """
    max_workers: 
    """
    start_time = time.time()
    subgroup_size = ceil(len(frame_files) / n_subgroups)
    subgroups = [frame_files[i:i + subgroup_size] for i in range(0, len(frame_files), subgroup_size)]
    threshold = 0.7 
    selected_frames = [] # List that will contain all the selected frames by each subgroup 

    with ProcessPoolExecutor(max_workers=max_workers) as executor: 
        futures = [executor.submit(select_desired_frames, "histogram", subgroup, input_folder, threshold)
            for subgroup in subgroups ]

        # Wait for all subgorups to finish processing 
        for future in as_completed(futures): 
            try: 
                selected_frames.extend(future.result())
            except Exception as e: 
                print(f"Error occurred during subgroup processing: {e}")
    
    end_time = time.time()

    print(f"Elapsed time to process: {end_time - start_time}")

    return selected_frames 