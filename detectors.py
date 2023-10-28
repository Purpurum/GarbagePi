import os
import yolov5
from ultralytics import YOLO
from PIL import Image

def load_detector_model(model_path, model_arch = "YOLOv8"):
    #TODO
    #Ифы на каждую модель через аргумент, лишние по итогу убрать
    #Функция возвращает загруженную модель

    if model_arch == "YOLOv8":
        model_yolov8 = YOLO(model_path)
        return model_yolov8

def detect(model, image, modelname = "YOLOv8"):
    #TODO
    #Ифы на каждую модель через аргумент, лишние по итогу убрать
    #Функция возвращает загруженную модель
    if modelname == "YOLOv8":
        
        results = model(image)
        im_array = results[0].plot()
        img_with_boxes = Image.fromarray(im_array[..., ::-1])
        try:
            img_with_boxes.save("results.png")
        except:
            pass
        return img_with_boxes, results
    