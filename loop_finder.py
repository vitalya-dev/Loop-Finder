#!/usr/bin/env python3
import os
import sys
import warnings
import argparse
from typing import Tuple, Optional, Any

import librosa
import numpy as np
import soundfile as sf

def load_audio(file_path: str, target_sr: int = 22050) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Загружает аудиофайл, конвертирует в моно и приводит к заданной частоте дискретизации.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    print(f"[*] Загрузка аудиофайла: {file_path}...")
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        print(f"[+] Загрузка завершена успешно. Сэмплов: {len(y)}")
        # Принудительно возвращаем sr как int, чтобы Pyright не сомневался
        return y, int(sr)
    except Exception as e:
        print(f"[-] Ошибка при загрузке файла: {e}")
        return None, None

def extract_features(y: np.ndarray, sr: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Извлекает биты и хромаграмму из аудио.
    """
    print("[*] Анализ трека: извлечение темпа, битов и гармонии...")
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        
        tempo_val = tempo[0] if isinstance(tempo, np.ndarray) else tempo
        print(f"[+] Анализ завершен. Найдено {len(beat_frames)} битов. Темп: {tempo_val:.2f} BPM.")
        return beat_frames, beat_times, chromagram
    except Exception as e:
        print(f"[-] Ошибка при извлечении характеристик: {e}")
        return None, None, None

def find_best_loop(beat_frames: np.ndarray, beat_times: np.ndarray, chromagram: np.ndarray, min_duration: float, max_duration: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Ищет идеальный луп по матрице хромаграмм среди доступных битов.
    """
    print(f"[*] Поиск идеальной петли в диапазоне {min_duration}-{max_duration} сек...")
    best_score = float('inf')
    # Изначально задаем тип как Optional[float]
    best_loop: Tuple[Optional[float], Optional[float]] = (None, None)
    window_size = 10 
    
    num_beats = len(beat_frames)
    max_frames = chromagram.shape[1]
    
    for i in range(num_beats):
        start_frame = beat_frames[i]
        start_time = beat_times[i]
        
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
                break
                
            if end_frame + window_size >= max_frames:
                continue
                
            end_features = chromagram[:, end_frame : end_frame + window_size]
            distance = np.linalg.norm(start_features - end_features)
            
            if distance < best_score:
                best_score = float(distance)
                best_loop = (float(start_time), float(end_time))
                
    if best_loop[0] is not None and best_loop[1] is not None:
        print(f"[+] Успех! Найден лучший повтор:")
        print(f"    Старт: {best_loop[0]:.3f} сек.")
        print(f"    Конец: {best_loop[1]:.3f} сек.")
        print(f"    Длина петли: {best_loop[1] - best_loop[0]:.3f} сек. (Score: {best_score:.4f})")
    else:
        print("[-] Не удалось найти подходящую петлю в заданном диапазоне времени.")
        
    return best_loop[0], best_loop[1]

def export_loop(y: np.ndarray, sr: int, start_time: float, end_time: float, output_path: str) -> bool:
    """
    Вырезает кусок аудио и сохраняет его на диск.
    """
    print(f"[*] Экспорт фрагмента в {output_path}...")
    try:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        loop_audio = y[start_sample:end_sample]
        sf.write(output_path, loop_audio, sr)
        print(f"[+] Экспорт успешно завершен.")
        return True
    except Exception as e:
        print(f"[-] Ошибка при экспорте файла: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Автоматический поиск идеального аудио-лупа для зацикленных видео и гифок.")
    parser.add_argument("input", help="Путь к исходному аудиофайлу (mp3, wav, flac и т.д.)")
    parser.add_argument("-o", "--output", default="perfect_loop.wav", help="Путь для сохранения лупа (по умолчанию: perfect_loop.wav)")
    parser.add_argument("--min", type=float, default=10.0, help="Минимальная длительность лупа в секундах (по умолчанию: 10.0)")
    parser.add_argument("--max", type=float, default=30.0, help="Максимальная длительность лупа в секундах (по умолчанию: 30.0)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[-] Ошибка: Файл {args.input} не найден.")
        sys.exit(1)
        
    y, sr = load_audio(args.input)
    # Явная проверка обеих переменных убедит Pyright, что дальше пойдут только корректные данные
    if y is None or sr is None: 
        sys.exit(1)
    
    beat_frames, beat_times, chromagram = extract_features(y, sr)
    if beat_frames is None or beat_times is None or chromagram is None: 
        sys.exit(1)
    
    start_t, end_t = find_best_loop(beat_frames, beat_times, chromagram, args.min, args.max)
    # Аналогично проверяем обе переменные времени на None
    if start_t is None or end_t is None: 
        sys.exit(1)
    
    # Теперь Pyright на 100% уверен, что передаются нужные типы: np.ndarray, int, float, float
    export_loop(y, sr, start_t, end_t, args.output)
    print("[+] Готово! Скрипт завершил работу.")

if __name__ == '__main__':
    main()