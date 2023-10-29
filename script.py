import os
import csv

folder_path = 'results/'  # путь к папке с файлами
csv_filename = 'submission.csv'  # название CSV-файла

# Получение списка файлов в папке
file_list = sorted(os.listdir(folder_path), key=lambda name: len(name.split(";")[0]))
print(file_list)
# Создание CSV-файла и запись данных
with open(csv_filename, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
   
    # Запись названий файлов
    for file_name in file_list:
        writer.writerow([file_name])
        
print(f"CSV-файл '{csv_filename}' успешно создан.")