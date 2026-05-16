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