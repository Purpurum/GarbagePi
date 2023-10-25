import os
import yolov5
from ultralytics import YOLO
from PIL import Image

def load_detector_model(modelname):
    #TODO
    #Ифы на каждую модель через аргумент, лишние по итогу убрать
    #Функция возвращает загруженную модель
    if modelname == "YOLOv5":
        yolov5_models_folder = "models/yolo/v5/"
        yolov5_models = [os.path.join(yolov5_models_folder, f) for f in os.listdir(yolov5_models_folder) if os.path.isfile(os.path.join(yolov5_models_folder, f))]
        model_yolov5 = yolov5.load(yolov5_models[0])
        return model_yolov5

    if modelname == "YOLOv8":
        yolov8_models_folder = "models/yolo/v8/"
        yolov8_models = [os.path.join(yolov8_models_folder, f) for f in os.listdir(yolov8_models_folder) if os.path.isfile(os.path.join(yolov8_models_folder, f))]
        model_yolov8 = YOLO(yolov8_models[0])
        return model_yolov8

def detect(modelname, model, image):
    #TODO
    #Ифы на каждую модель через аргумент, лишние по итогу убрать
    #Функция возвращает загруженную модель
    if modelname == "YOLOv8":
        results = model(image)
        im_array = results[0].plot()
        img_with_boxes = Image.fromarray(im_array)
        return img_with_boxes, results
    
    if modelname == "YOLOv5":
        results = model(image, size=640)
        predictions = results.pred[0]
        boxes = predictions[:, :4] # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        r_img = results.render()
        img_with_boxes = r_img[0]
        return img_with_boxes, results