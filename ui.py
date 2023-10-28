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
#Импорты внутренних скриптов
from detectors import load_detector_model, detect
from utils import get_files_in_directory


def main():
    selector_option = st.sidebar.selectbox("Выберите модель детекции", options=["YOLOv5","YOLOv8n_GC","YOLOv8n_thesis","YOLOv8s"]) #Она выбирается в мейне ЫЛЬЯ!
    st.sidebar.header('Ввод') 
    st.text(body=selector_option) #Она выбирается в мейне ЫЛЬЯ!
    selected_model = load_detector_model(selector_option) #Она выбирается в мейне ЫЛЬЯ!
    print(selected_model)
    st.title('Вывод')
  
    

    def load():
        load_type = ['jpg','png','jpeg','zip','rar','mkv','mp4','mpg','mpeg','mpeg4']
        uploaded_data= st.sidebar.file_uploader(label="Выберете файл для распознования",type = load_type)
        if uploaded_data is not None:
            file_type = get_file_type(uploaded_data.name)
            print(file_type)
            if any(file_type == i for i in load_type[:3]):load_img(uploaded_data)
            elif any(file_type == i for i in load_type[3:5]):load_zip(uploaded_data)
            elif any(file_type == i for i in load_type[5:]):load_video(uploaded_data)
            else:st.text("Ошибка чтения файла, возможно неподходящий формат\n(убедитесь что в названии файла нет точек и специальных символов)")          
        else:
            return None
    

    def get_file_type(file):
        return os.path.splitext(file)[1][1:]

    def load_img(img_file):
        img = img_file.getvalue()
        st.image(img)
        image = Image.open(BytesIO(img))
        #image = np.asarray(image)
        image, results = detect(selector_option, selected_model, image)
        st.image(image)
        st.text(results)

    def load_zip(zip_file):
        results_list = []
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)      
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                image, result = detect(selector_option, selected_model,file_path)
                results_list.append(result)

        with open('results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(results_list)

    def load_video(video_file):
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        video_capture = cv2.VideoCapture(tfile.name)
        FRAME_WINDOW = st.image([])

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break 
            
            processed_frame, results = detect(selector_option, selected_model, frame)
            FRAME_WINDOW.image(processed_frame)
    
    def load_camera():
        run = st.sidebar.checkbox('Run')
        if run:    
            FRAME_WINDOW = st.image([])
            RESULTS_WIDNOW = st.image([])
            camera = cv2.VideoCapture(0)

        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image, results = detect(selector_option, selected_model, frame)
            FRAME_WINDOW.image(frame)
            RESULTS_WIDNOW.image(image)

        else:
            st.write('Stopped')
    
    #load_camera() 
    load()
    


if __name__ == '__main__':
    main()