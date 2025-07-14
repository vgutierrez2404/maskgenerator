import torch 
import numpy as np 
from tqdm import tqdm 
import tempfile
import os 

from src.config_loader import ConfigLoader  
from src.video import Video 
from sam2.build_sam import build_sam2_video_predictor
from src.utils.functions import get_frame_idx

class InferenceProcessor:
    """ 
    Defines the way that the inference will be done. In a batch way or a normal (sam2 notebook example) way. 
    """
    def __init__(self, video: Video):
        """
        Initializes the inference processor with the video and configuration.

        Args:
            video (Video): The video object containing the input data.
            config (ConfigLoader): The configuration loader object.
        """
        self.video = video
        self.config = ConfigLoader()
        self.device = self._check_device_used()
        self.predictor = self._load_predictor()

    def _check_device_used(self): 
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

    def _load_predictor(self):
        """
        Loads the video predictor model based on the configuration.
        """
        return build_sam2_video_predictor(
            self.config.model_config,
            self.config.checkpoints,
            device=self.device,
            config_path=self.config.config_path
        )
    
    def _normal_inference(self): 
        """
        As done in the original sam demo notebook. It can run out of memory if the 
        video has too many frames. 

        Return: 
            - video_segments: dictionary containing the index of the frame and its mask. 
        """
        inference_state = self.predictor.init_state(video_path=self.video.selected_frames_path, async_loading_frames=True)  # async_loading_frames: introduce en memoria los frames de forma asyncrona mientras hace inferencia con vide_prediction. 

        points = np.array(list(self.video.coordinates.values()), dtype=np.float32)
        
        labels = np.ones(points.shape[0], dtype=np.int32)
        frame_index = 0
        ann_obj_id = 1 # give a unique id to each object we interact with (it can be any integers). A single id for each object to track in the prediction. 

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=frame_index, obj_id=ann_obj_id, points=points, labels=labels)
        
        video_segmentations = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segmentations[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        return video_segmentations 

    def _inference_by_batches(self): 
        """
        Infereces by batches the video, allowing to process larger videos in a single gpu. 
        Divides frames path into batches. The batch is processed and the mask of the last frame is saved. 
        In the next batch, the saved mask is used as input (two adjoining frames should not be very different).

        Return: 
            - video_segments: dictionary containing the index of the frame and its mask. 
        """
        
        def batch_generator(frame_paths, batch_size):
            """Yield successive batches from frame_paths."""
            for i in range(0, len(frame_paths), batch_size):
                yield frame_paths[i:i + batch_size]

        video_segmentations = {} # frame_idx-mask. 
        last_mask = None 
        for index, batch in enumerate(tqdm(batch_generator(self.video.get_frame_names(), self.config.batch_size))):
            with tempfile.TemporaryDirectory() as temp_dir: # temp dir to store frames from the batch (sam2 predictor needs a directory, not abs paths)
                for frame_path in batch: 
                    frame_path = os.path.join(self.video.selected_frames_path, frame_path)
                    os.symlink(frame_path, os.path.join(temp_dir, os.path.basename(frame_path)))
        
                batch_inference_state = self.predictor.init_state(temp_dir) # no need for async loading of frames if processed by batches. 

                frame_idx = get_frame_idx(batch[0])
                if last_mask is None or (isinstance(last_mask, np.ndarray) and last_mask.size == 0): # for first batch we use coord or bbox as prompt to predict the mask. 

                    if self.video.has_bounding_box(frame_idx): 
                         
                        # label 1 indicates a positive click (to add a region) 
                        # while label 0 indicates a negative click (to remove a region).
                    
                        ann_obj_id = 1 # unique id for the object we are annotating.
                        bbox = self.video.get_bounding_box(frame_idx)
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(inference_state=batch_inference_state,
                                                                                        frame_idx=frame_idx, obj_id=ann_obj_id, box=bbox)

                    else: 
                        points = np.array(list(self.video.coordinates.values()), dtype=np.float32)
                        # label 1 indicates a positive click (to add a region) 
                        # while label 0 indicates a negative click (to remove a region).
                        labels = np.ones(points.shape[0], dtype=np.int32) 
                        ann_obj_id = 1 
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(inference_state=batch_inference_state,
                                                                                        frame_idx=frame_idx, obj_id=ann_obj_id, points=points, labels=labels)

                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(batch_inference_state):
                            video_segmentations[(index * self.config.batch_size) + out_frame_idx] = {
                                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                                for i, out_obj_id in enumerate(out_obj_ids)
                                }
                            
                    if video_segmentations: 
                        last_mask = next(reversed(video_segmentations[(index * self.config.batch_size) + out_frame_idx].values())).squeeze() # save the last mask. 

                    else: 
                        raise ValueError("Error: No mask found for the last frame in the batch.")
                    
                else: # rest of batches we propagate the mask of the last batch. 
                    ann_obj_id = 1 # the index of the object we are masking.  
                    frame_idx = 0 # in the first frame of the batch. 
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(batch_inference_state, frame_idx, ann_obj_id, last_mask) # es este frame index?  
                    
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(batch_inference_state):
                        video_segmentations[(index * self.config.batch_size) + out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                            }
                        
                    last_mask = next(reversed(video_segmentations[(index * self.config.batch_size) + out_frame_idx].values())).squeeze()
                
                # always reset the state of the predictor when the batch ends. / 
                self.predictor.reset_state(batch_inference_state)

            print(f"Processing next batch {index + 1 }\n")  

        return video_segmentations      

    def run_inference(self):
        """
        Runs inference based on the selected inference type in the configuration.
        """
        if self.config.inference_type == "batches":
            return self._inference_by_batches()
        
        else:
            
            video_segmentations = self._normal_inference()
            if not video_segmentations: 
                raise ValueError("Error while trying to run inference. No video_segmentations found.")
            
