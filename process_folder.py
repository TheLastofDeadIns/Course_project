import os
import re
import csv
from analyze_audio import analyze_audio


def extract_number(file_name):
    """
    Извлекает число из имени файла, например, rec123.wav -> 123.
    :param file_name: Имя файла.
    :return: Число, извлеченное из имени файла.
    """
    match = re.search(r'rec(\d+)\.wav', file_name)
    if match:
        return int(match.group(1))
    return -1  # Если число не найдено


def read_descriptions(description_file):
    """
    Читает описания из текстового файла.
    :param description_file: Путь к файлу с описаниями.
    :return: Список описаний.
    """
    try:
        # Пробуем прочитать файл в кодировке UTF-8
        with open(description_file, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines()]
    except UnicodeDecodeError:
        # Если UTF-8 не работает, пробуем Windows-1251
        with open(description_file, 'r', encoding='windows-1251') as file:
            return [line.strip() for line in file.readlines()]


def filter_files(files, descriptions):
    """
    Фильтрует файлы по описаниям.
    :param files: Список файлов.
    :param descriptions: Список описаний.
    :return: Отфильтрованный список файлов и соответствующие описания.
    """
    filtered_files = []
    filtered_descriptions = []

    for i, file_name in enumerate(files):
        if i < len(descriptions):
            description = descriptions[i]
            if description.startswith(("good", "Тромбоз", "Стеноз", "Дисфункция")):
                filtered_files.append(file_name)
                filtered_descriptions.append(description)
        else:
            break  # Если описаний меньше, чем файлов

    return filtered_files, filtered_descriptions


def process_folder(folder_path, description_file, m, output_file):
    """
    Обрабатывает все аудиофайлы в папке.
    :param folder_path: Путь к папке с аудиофайлами.
    :param description_file: Путь к файлу с описаниями.
    :param m: Длина z-векторов.
    :param output_file: Путь к файлу для сохранения результатов.
    """
    # Получаем список файлов в папке
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    if not files:
        print("В папке нет .wav файлов.")
        return

    # Сортируем файлы по числовому значению номера
    files.sort(key=extract_number)

    # Читаем описания
    descriptions = read_descriptions(description_file)

    # Фильтруем файлы по описаниям
    filtered_files, filtered_descriptions = filter_files(files, descriptions)

    if not filtered_files:
        print("Нет файлов, соответствующих критериям.")
        return

    # Создаем матрицу для результатов
    results_matrix = []

    # Обрабатываем каждый отфильтрованный файл
    for file_name, description in zip(filtered_files, filtered_descriptions):
        file_path = os.path.join(folder_path, file_name)
        try:
            print(f"\nАнализ файла: {file_name}")
            normalized_entropy, C, H = analyze_audio(file_path, m)

            # Определяем значение для четвертого столбца
            label = 1 if description.startswith("good") else 0

            # Добавляем строку в матрицу
            results_matrix.append([normalized_entropy, C, H, label])

            print("Результат (нормированная энтропия, сложность, индекс Херста, метка):")
            print(results_matrix[-1])
        except Exception as e:
            print(f"Ошибка при обработке файла {file_name}: {e}")

    # Сохраняем матрицу в CSV-файл
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Нормированная энтропия", "Сложность", "Индекс Херста", "Метка"])
        writer.writerows(results_matrix)

    print(f"\nРезультаты сохранены в файл: {output_file}")


if __name__ == "__main__":
    # Укажите путь к папке с аудиофайлами
    folder_path = "records"

    # Укажите путь к файлу с описаниями
    description_file = "records/conclusions.txt"

    # Укажите путь к файлу для сохранения результатов
    output_file = "results2.csv"

    m = 10

    # Обработка папки
    process_folder(folder_path, description_file, m, output_file)