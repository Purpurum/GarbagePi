from io import BytesIO
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import time
import tempfile
import zipfile
import csv
import tifffile
from ultralytics import YOLO

import detectors
from res.style import page_style


def main(models):
    
    page_style()
    
    st.title('Garbage detection by Ping')
    
    def load():
        load_type = ['jpg','png','jpeg','mkv','mp4','mpg','mpeg','mpeg4','tif','tiff']
        uploaded_data = st.file_uploader(label="Выберите файл для распознавания",type=load_type)
        if uploaded_data is not None:
            file_type = get_file_type(uploaded_data.name)
            print(file_type)
            if any(file_type == i for i in load_type[:3]):load_img(uploaded_data)
            elif any(file_type == i for i in load_type[3:8]):load_video(uploaded_data)
            elif any(file_type == i for i in load_type[8:]):load_tif(uploaded_data)
            else:st.text("Ошибка чтения файла, возможно неподходящий формат\n(убедитесь что в названии файла нет точек и специальных символов)")          
        else:
            return None
    
    def get_file_type(file):
        return os.path.splitext(file)[1][1:]
    
    def load_img(img_file):
        img = img_file.getvalue()
        st.image(img)
        image = Image.open(BytesIO(img))
        image, results = detectors.ensemble_detect(models, image)
        st.image(image)
        st.text(f"Дерево:{results[0]}")
        st.text(f"Стекло:{results[1]}")
        st.text(f"Пластик:{results[2]}")
        st.text(f"Металл:{results[3]}")
    
    def load_tif(tif_file):
        end_point = st.select_slider(
            'Выберите длину волны',
            options=['400', '430', '460', '490', '520', '550', '580', '610', '640', '670', '700'])
        points = {'400':0, '430':1, '460':2, '490':3, '520':4, '550':5, '580':6, '610':7, '640':8, '670':9, '700':10}
        chosen_point =  points[end_point]
        image = tifffile.imread(tif_file)
        channel = image[chosen_point, :, :]
        channel = (channel * 255).astype(np.uint8)
        channel_image = Image.fromarray(channel)       
        st.image(channel_image)
        image, results = detectors.ensemble_detect(models, channel_image, type="png")
        st.image(image)
        st.text(f"Дерево:{results[0]}")
        st.text(f"Стекло:{results[1]}")
        st.text(f"Пластик:{results[2]}")
        st.text(f"Металл:{results[3]}")

    def load_video(video_file):
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        video_capture = cv2.VideoCapture(tfile.name)
        FRAME_WINDOW = st.image([])
        res = []
        counter = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break  
             
            counter += 1    
            processed_frame, results = detectors.ensemble_detect(models, frame)
            if counter % 25 == 0:   
                res.append(results)

            FRAME_WINDOW.image(processed_frame)
            sums = [sum(x) for x in zip(*res)]
        
        st.text(f"Дерево:{sums[0]}")
        st.text(f"Стекло:{sums[1]}")
        st.text(f"Пластик:{sums[2]}")
        st.text(f"Металл:{sums[3]}")
    def load_camera():
        run = st.checkbox('Стрим потока с камеры')
        if run:    
            RESULTS_WIDNOW = st.image([])
            camera = cv2.VideoCapture(0)

        while run:
            _, frame = camera.read()
            image, results = detectors.ensemble_detect(models, frame)
            new_image = np.array(image.convert('RGBA'), dtype=np.uint8)
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
            RESULTS_WIDNOW.image(new_image)

        else:
            pass
    
    load_camera()
    load()
    
print("loading_models")
model1 = YOLO("models/YOLOv8/yolov8_combined_pre_s.pt")
model2 = YOLO("models/YOLOv8/yolov8_combined_pre_s.pt")
model3 = YOLO("models/YOLOv8/yolov8_combined_s.pt")
models = [model1, model2, model3]

if __name__ == '__main__':
    main(models)