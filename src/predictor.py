from sam2.build_sam import build_sam2_video_predictor
import os 
# this code is retrieved from META's notebooks (https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb#scrollTo=f5f3245e-b4d6-418b-a42a-a67e0b3b5aec)

class Predictor: 
    def __init__(self, device): 
        # self.checkpoints = "../sam2/checkpoints/sam2.1_hiera_large.pt" 
        # self.model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
                # Define the root directory of your project
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

        # Paths to checkpoints and model configuration
        self.checkpoints = os.path.join(project_root, "sam2", "checkpoints", "sam2.1_hiera_large.pt")
        self.model_config = os.path.join(project_root,"sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")
        self.model = build_sam2_video_predictor(self.model_config, self.checkpoints, device = device)
