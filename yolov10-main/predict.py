from ultralytics import YOLOv10

# Load a pretrained YOLOv10n model
model = YOLOv10("runs/detect/train_v10/weights/best.pt")

# Perform object detection on an image
# results = model("test1.jpg")
results = model.predict("ultralytics/assets/bus.jpg")
# Display the results
results[0].show()