from ultralytics import YOLOv10

model_yaml_path = "ultralytics/cfg/models/v10/yolov10n.yaml"

data_yaml_path = 'ultralytics/cfg/datasets/helmet.yaml'

pre_model_name = 'weights/yolov10n.pt'

if __name__ == '__main__':
    model = YOLOv10("ultralytics/cfg/models/v10/yolov10n.yaml").load('weights/yolov10n.pt')

    results = model.train(data=data_yaml_path, epochs=10, batch=8, name='train_v10')
