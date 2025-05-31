From a virtual environment
* Installed cuda version 12.1 https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local 
* cd sam2 and installed sam2 requirements: https://github.com/facebookresearch/sam2 
```
    pip install -e . 
```
## TODO 
Add a bounding box feature to select the foreground of the scene in case that more than 1 object exists and the user doesn't want to use points. 

### Usage 

After installation, run: 
```
    python3 app.py # o el nombre que acabe teniendo 
``` 
and select the desired video. 
