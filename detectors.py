from ultralytics import YOLO
from PIL import Image

def load_detector_model(model_path, model_arch = "YOLOv8"):
    if model_arch == "YOLOv8":
        model_yolov8 = YOLO(model_path)
        return model_yolov8
#Функция Локального детекта
def detect(model, image):
    results = model(image, verbose=False)
    im_array = results[0].plot()
    img_with_boxes = Image.fromarray(im_array[..., ::-1])
    try:
        img_with_boxes.save("results.png")
    except:
        pass
    return img_with_boxes
#Функция ансамбля
def ensemble_detect(models_pack, image, time = None, type = "png"):
    try:
        image = image[...,::-1]
    except:
        pass
    if type == "png":
        results_list = []
        for model in models_pack:
            results = model(image)
            names_count = {}
            names = model.names
            for name in names:
                names_count[name] = results[0].boxes.cls.tolist().count(name)
            results_list.append(names_count)

        most_common_values = {}

        for d in results_list:
            for key, value in d.items():
                if key in most_common_values:
                    counts = most_common_values[key]
                    if value in counts:
                        counts[value] += 1
                    else:
                        counts[value] = 1
                else:
                    most_common_values[key] = {value: 1}

        output_list = []

        for key, counts in most_common_values.items():
            most_common = max(counts, key=counts.get)
            output_list.append(most_common)
        image_with_boxes = detect(models_pack[1], image)
        
        return image_with_boxes, output_list
