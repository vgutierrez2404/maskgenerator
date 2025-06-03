import os
import yaml

class ConfigLoader:
    """
    Load configuration for the application from a YAML file. 
    """
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()

        # by default we use the small model. 
        self.checkpoints = os.path.join(os.getcwd(), self.config.get("model", {}).get("checkpoints", "sam2/checkpoints/sam2.1_hiera_small.pt"))
        self.model_config = self.config.get("model", {}).get("architecture", "sam2.1_hiera_l.yaml") 
        self.config_path = os.path.join(os.getcwd(), self.config.get("model", {}).get("architecture_path", "sam2/sam2/configs/sam2.1"))
        self.batch_size = self.config.get("processing", {}).get("batch_size", 10)
        self.inference_type = self.config.get("processing", {}).get("inference_type", "normal")  

    def _load_config(self): # underscore at start means private method 
        """Private method to load the YAML configuration file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file '{self.config_file}' not found.")

        with open(self.config_file, "r") as file:
            return yaml.safe_load(file)

    def get(self, key_path, default=None):
        """
        Get a value from the config using a dot-separated key path.
        Example: config.get("video.model.model_checkpoints")
        """
        keys = key_path.split(".")
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            return default

    def reload(self):
        """Reload the configuration from the file."""
        self.config = self._load_config()
        self.checkpoints = os.path.join(os.getcwd(), self.config["video"]["model"]["model_checkpoints"])
        self.model_config = self.config["model"]["config_file"]
        self.config_path = os.path.join(os.getcwd(), self.config["paths"]["config_directory"])
