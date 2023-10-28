<div align="center">
  <a href="https://ibb.co/ZX537yv"><img src="https://i.ibb.co/dMFSxnX/banner2.png" alt="banner2" border="0" /></a>
</div>

## <div align="center">Стэк технологий📑</div>
<div align="center">
  <a href="https://www.python.org/doc/"><img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"></a>
  <a href="https://pytorch.org/docs/stable/index.html"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"></a>
  <a href="https://opencv.github.io/cvat/docs/"><img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white"></a>
  <br>
  <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg"></a>
  <a href="https://docs.streamlit.io/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"></a>
</div>

## <div align="center">О нашем решении📝</div>
<p>Мы представляем модели для автоматического распознавания и подсчёта уникальных элементов четырёх категорий ТБО: дерево, пластик, стекло, металл с RGB + мультиспектральных изображений. Уникальность нашего решения заключается в использовании нескольких моделей, предобученных на разных датасетах в ансамбле(одновременно), для более точного распознавания категорий ТБО. А также в универсальности, так как присутствует возможность распознавания как с изображения, так и с видео или прямого потока с камеры устройства.
</p>

## <div align="center">Быстрый старт🎢</div>
<details open>
  
#### Установка зависимостей
<p>
Для запуска проекта необходимо установить зависимости. Необходимые для работы проекта библиотеки можно посмотреть в файле <a href="https://docs.ultralytics.com/">🧨requirements.txt</a> и установить их вручную. Также можно воспользоваться командой:
</p>
  
```bash
$ pip install -r requirements.txt
```

#### Запуск пользовательского интерфейса
<p>
  Для работы с моделями детекции ТБО можно воспользоваться разработанным нашей командой пользовательским интерфейсом, для его запуска необходимо выполнить следующую команду: 
</p>
  
```bash
$ streamlit run newUi.py
```
<p>
  Данная команда запускает веб приложение разработанное на базе фреймворка <a href="https://streamlit.io/">Streamlit</a>. 
</p>   

#### Предпросмотр работы моделей
<p>
  Для предпросмотра работы моделей без запуска пользовательского интерфейса можно воспользоваться <a href="https://jupyter.org/">Jupiter</a> блокнотом, данный блокнот лежит в репозитории проекта <a href="https://docs.ultralytics.com/">🧨testrun.ipynb</a> 
</p> 
</details>

## <div align="center">Модели детекции🪀</div>
<p>
  В своем проекте мы использовали ансамблевый метод. Было использовано три модели детекции объектов архитектуры YOLOv8small. Каждая модель была предобучена на различных датасетах, затем была дообучена на предоставленных наборах данных. Архитектура ансамбля представлена на схеме:  
</p>
<div align="center">

  #### Схема работы ансамбля 🧨🧨🧨
  <img src="https://i.ibb.co/dMFSxnX/banner2.png"/>
</div>

## <div align="center">Результат работы моделей🔮</div>
<p>
  В разработанном приложении есть возможность детекции ТБО на картинках(jpg,png,tif,tiff) и видео(mp4, mkv, камера реального времени), результатом работы алгоритма является подсчет уникальных элементов, таких классов как: дерево, стекло, металл и пластик. Примеры работы приложения на разных типах данных:
</p>

| Фото                                                                                         | Видео                 | Камера реального времени | 
| -------------------------------------------------------------------------------------------- | --------------------- | ------------------------ | 
|🧨                                                                                            | 🧨                    | 🧨                    | 
