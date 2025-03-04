from ultralytics.models.sam import SAM

# Load a model
model = SAM("sam2_b.pt")

# Display model information (optional)
model.info()

# Segment with bounding box prompt
#results = model("path/to/image.jpg", bboxes=[100, 100, 200, 200])

# Segment with point prompt
results = model("/Users/victorgutierrezgarcia/Desktop/SAM2_annotator/istockphoto-1896758161-640_adpp_is.mp4")
