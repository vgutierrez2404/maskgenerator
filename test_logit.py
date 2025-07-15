import numpy as np
from matplotlib import pyplot as plt

def load_and_display_segmentation_logit(logit_path: str):
    """
    Load a segmentation logit from .npy file and display it as a binary image.
    Values < 0 are displayed as black, values > 0 as white.
    
    Args:
        logit_path (str): Path to the .npy file containing the logit
    """
    try:
        # Load the logit file
        logit_data = np.load(logit_path, allow_pickle=True)
        
        # Handle the case where logit_data is a numpy array containing a dictionary
        if isinstance(logit_data, np.ndarray) and logit_data.dtype == object:
            # Extract the dictionary from the numpy array
            logit_dict = logit_data.item()
            
            # Get the first (and likely only) value from the dictionary
            if isinstance(logit_dict, dict):
                print(f"Dictionary keys: {list(logit_dict.keys())}")
                # Get the logit array from the dictionary
                logit = next(iter(logit_dict.values()))
            else:
                logit = logit_dict
        else:
            logit = logit_data
        
        # Print logit information
        print(f"Logit shape: {logit.shape}")
        print(f"Logit dtype: {logit.dtype}")
        print(f"Min value: {logit.min():.4f}")
        print(f"Max value: {logit.max():.4f}")
        print(f"Values < 0: {np.sum(logit < 0)}")
        print(f"Values > 0: {np.sum(logit > 0)}")
        print(f"Values = 0: {np.sum(logit == 0)}")
        
        # If logit has multiple dimensions, take the first channel/slice
        if logit.ndim > 2:
            display_logit = logit[0] if logit.shape[0] == 1 else logit
            if display_logit.ndim > 2:
                display_logit = display_logit.squeeze()
        else:
            display_logit = logit
        
        # Create binary image: values < 0 = black (0), values > 0 = white (1)
        binary_image = (display_logit > 0).astype(np.uint8)
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(binary_image, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Segmentation Logit - Shape: {logit.shape}\nBlack: values < 0, White: values > 0")
        plt.axis("off")
        plt.colorbar(label="Binary mask (0=black, 1=white)")
        plt.show()
        
        return logit, binary_image
        
    except FileNotFoundError:
        print(f"Error: File not found at {logit_path}")
        return None, None
    except Exception as e:
        print(f"Error loading logit file: {e}")
        return None, None

def parse_logit_structure(logit_path: str):
    """
    Helper function to understand the structure of the logit file
    """
    try:
        data = np.load(logit_path, allow_pickle=True)
        
        print(f"Top level data type: {type(data)}")
        print(f"Top level data shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")
        print(f"Top level data dtype: {data.dtype if hasattr(data, 'dtype') else 'No dtype'}")
        
        # If it's an object array, extract the content
        if isinstance(data, np.ndarray) and data.dtype == object:
            content = data.item()
            print(f"Content type: {type(content)}")
            
            if isinstance(content, dict):
                print(f"Dictionary keys: {list(content.keys())}")
                for key, value in content.items():
                    print(f"Key {key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'No shape'}")
                    if hasattr(value, 'dtype'):
                        print(f"  dtype: {value.dtype}")
                    if hasattr(value, 'min'):
                        print(f"  min: {value.min():.4f}, max: {value.max():.4f}")
        
    except Exception as e:
        print(f"Error parsing structure: {e}")

if __name__ == "__main__":
    # Replace with your actual logit file path
    logit_path = ""

    print("=== Analyzing logit structure ===")
    parse_logit_structure(logit_path)
    
    print("\n=== Loading and displaying logit ===")
    logit, binary_mask = load_and_display_segmentation_logit(logit_path)