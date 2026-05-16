import librosa
import warnings

def load_audio(file_path: str, target_sr: int = 22050):
    """
    Загружает аудиофайл, конвертирует в моно и приводит к заданной частоте дискретизации.
    
    :param file_path: Путь к аудиофайлу.
    :param target_sr: Желаемая частота дискретизации (по умолчанию 22050 Гц для баланса скорости и точности).
    :return: Кортеж (audio_time_series, sample_rate) или (None, None) в случае ошибки.
    """
    # Игнорируем предупреждения PySoundFile, которые часто возникают при чтении mp3
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print(f"Загрузка аудиофайла: {file_path}...")
    try:
        # sr=target_sr приводит аудио к единому sample rate
        # mono=True переводит в моно для упрощения поиска частотных паттернов
        audio_time_series, sample_rate = librosa.load(file_path, sr=target_sr, mono=True)
        print("Загрузка успешно завершена.")
        return audio_time_series, sample_rate
    except Exception as e:
        print(f"Ошибка при загрузке файла {file_path}: {e}")
        return None, None

import librosa
import numpy as np

def extract_features(y: np.ndarray, sr: int):
    """
    Извлекает биты и хромаграмму из аудио для последующего анализа.
    
    :param y: Временной ряд аудио (numpy array).
    :param sr: Частота дискретизации.
    :return: Кортеж (beat_frames, beat_times, chromagram) или (None, None, None) при ошибке.
    """
    print("Анализ трека: извлечение битов и хромаграммы...")
    try:
        # Получаем индексы кадров (фреймов), где происходят удары (биты), и темп
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        # Конвертируем индексы кадров в реальное время (в секундах)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Строим хромаграмму (распределение энергии по 12 нотам)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Обработка темпа для корректного вывода в разных версиях librosa
        tempo_val = tempo[0] if isinstance(tempo, np.ndarray) else tempo
        print(f"Анализ завершен. Найдено {len(beat_frames)} битов. Оценочный темп: {tempo_val:.2f} BPM.")
        
        return beat_frames, beat_times, chromagram
    except Exception as e:
        print(f"Ошибка при извлечении характеристик аудио: {e}")
        return None, None, None

import numpy as np

def find_best_loop(beat_frames: np.ndarray, beat_times: np.ndarray, chromagram: np.ndarray, min_duration: float = 10.0, max_duration: float = 30.0):
    """
    Ищет идеальный луп, сравнивая хромаграммы на разных битах.
    
    :param beat_frames: Массив индексов кадров битов.
    :param beat_times: Массив времени битов в секундах.
    :param chromagram: Хромаграмма трека.
    :param min_duration: Минимальная длина лупа в секундах.
    :param max_duration: Максимальная длина лупа в секундах.
    :return: Кортеж (start_time, end_time) лучшего лупа или (None, None).
    """
    print(f"Поиск идеального лупа от {min_duration} до {max_duration} секунд...")
    
    best_score = float('inf')  # Ищем минимальное расстояние (максимальную схожесть)
    best_loop = (None, None)
    
    # Размер окна для сравнения (в кадрах). Мы берем небольшой срез после бита.
    # 10 кадров при нашем sample_rate — это около доли секунды, 
    # ровно столько, чтобы "захватить" звучание аккорда или баса.
    window_size = 10 
    
    num_beats = len(beat_frames)
    max_frames = chromagram.shape[1]
    
    for i in range(num_beats):
        start_frame = beat_frames[i]
        start_time = beat_times[i]
        
        # Проверяем, не выходит ли окно за пределы хромаграммы
        if start_frame + window_size >= max_frames:
            continue
            
        start_features = chromagram[:, start_frame : start_frame + window_size]
        
        for j in range(i + 1, num_beats):
            end_frame = beat_frames[j]
            end_time = beat_times[j]
            duration = end_time - start_time
            
            if duration < min_duration:
                continue
            if duration > max_duration:
                break # Биты отсортированы по времени, дальше проверять нет смысла
                
            if end_frame + window_size >= max_frames:
                continue
                
            end_features = chromagram[:, end_frame : end_frame + window_size]
            
            # Вычисляем Евклидово расстояние между матрицами хромаграмм.
            # Чем ближе значение к нулю, тем идентичнее звучат эти моменты.
            distance = np.linalg.norm(start_features - end_features)
            
            if distance < best_score:
                best_score = distance
                best_loop = (start_time, end_time)
                
    if best_loop[0] is not None:
        print(f"Найден идеальный луп: {best_loop[0]:.3f} сек. -> {best_loop[1]:.3f} сек. (Длительность: {best_loop[1]-best_loop[0]:.3f} сек.)")
    else:
        print("Не удалось найти подходящий луп в заданном диапазоне времени.")
        
    return best_loop[0], best_loop[1]