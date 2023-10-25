from io import BytesIO
import os
import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live
from PIL import Image
import time
import tempfile
import zipfile
import csv
#Импорты внутренних скриптов
from detectors import load_detector_model, detect


def main():
    st.sidebar.header('Ввод')
    st.title('Вывод')
    load_camera()
        
    load()

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
    image = np.asarray(image)
    model = load_detector_model("YOLOv8")
    image, results = detect("YOLOv8", model, image)
    st.image(image)
    st.text(results)

def load_zip(zip_file):
    model = load_detector_model("YOLOv8")
    results_list = []

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            image, result = detect("YOLOv8",model,file_path)
            results_list.append(result)

    with open('results.csv', 'a', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerows(results_list)

def load_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())

    model = load_detector_model("YOLOv8")
    video_capture = cv2.VideoCapture(tfile.name)
    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break 
        
        processed_frame, results = detect("YOLOv8", model, frame)
        FRAME_WINDOW.image(processed_frame)
    
def load_camera():
    run = st.sidebar.checkbox('Run')
    if run:
      model = load_detector_model("YOLOv5")
      FRAME_WINDOW = st.image([])
      RESULTS_WIDNOW = st.image([])
      camera = cv2.VideoCapture(0)

      while run:
          _, frame = camera.read()
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image, results = detect("YOLOv5", model, frame)
          FRAME_WINDOW.image(frame)
          RESULTS_WIDNOW.image(image)

      else:
          st.write('Stopped')
    


if __name__ == '__main__':
	main()