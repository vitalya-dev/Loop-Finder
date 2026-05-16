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