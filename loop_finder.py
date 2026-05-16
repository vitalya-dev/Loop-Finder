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

def find_best_loop(beat_frames: np.ndarray, beat_times: np.ndarray, chromagram: np.ndarray, min_duration: float, max_duration: float, count: int = 5, window_size: int = 10, min_dist: float = 3.0) -> list:
    """
    Ищет несколько уникальных идеальных лупов по матрице хромаграмм среди доступных битов,
    избегая дубликатов, расположенных слишком близко друг к другу.
    """
    print(f"[*] Поиск топ-{count} уникальных петель в диапазоне {min_duration}-{max_duration} сек (размер окна: {window_size})...")
    found_loops = []
    
    num_beats = len(beat_frames)
    max_frames = chromagram.shape[1]
    
    # 1. Собираем абсолютно все возможные комбинации
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
            distance = float(np.linalg.norm(start_features - end_features))
            
            found_loops.append((distance, float(start_time), float(end_time)))
            
    # 2. Сортируем все найденные петли по возрастанию score (от лучших к худшим)
    found_loops.sort(key=lambda x: x[0])
    
    # 3. Отбираем только уникальные петли (отсеиваем близкие дубликаты)
    top_loops = []
    for loop_data in found_loops:
        # Если мы уже набрали нужное количество, останавливаем поиск
        if len(top_loops) >= count:
            break
            
        score, start_t, end_t = loop_data
        is_too_close = False
        
        # Проверяем дистанцию до уже добавленных в топ лупов
        for sel_start, sel_end in top_loops:
            if abs(start_t - sel_start) < min_dist:
                is_too_close = True
                break
                
        # Если луп достаточно далеко от других, берем его
        if not is_too_close:
            top_loops.append((start_t, end_t))
            print(f"[+] Найден уникальный луп #{len(top_loops)}: Старт: {start_t:.3f} сек, Конец: {end_t:.3f} сек, Длина: {end_t - start_t:.3f} сек (Score: {score:.4f})")
            
    if not top_loops:
        print("[-] Не удалось найти подходящие уникальные петли в заданном диапазоне времени.")
        
    return top_loops

def export_loop(y: np.ndarray, sr: int, start_time: float, end_time: float, output_path: str, crossfade_duration: float = 0.05) -> bool:
    """
    Вырезает кусок аудио, применяет кроссфейд для бесшовной склейки и сохраняет на диск.
    """
    print(f"[*] Экспорт фрагмента в {output_path} (crossfade: {crossfade_duration} сек)...")
    try:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        # Копируем основу лупа
        loop_audio = y[start_sample:end_sample].copy()
        
        fade_samples = int(crossfade_duration * sr)
        
        # Проверяем, есть ли в оригинальном аудио данные после конца лупа для "хвоста"
        if end_sample + fade_samples > len(y):
            fade_samples = len(y) - end_sample
            
        # Защита: кроссфейд не должен быть длиннее половины самого лупа
        if fade_samples > len(loop_audio) // 2:
            fade_samples = len(loop_audio) // 2
            
        if fade_samples > 0:
            # Берем "хвост" оригинального трека сразу после конца лупа
            tail = y[end_sample : end_sample + fade_samples].copy()
            
            # Создаем кривые: нарастание для начала лупа, затухание для хвоста
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            
            # Магический микс: накладываем затухающий хвост на нарастающее начало лупа
            loop_audio[:fade_samples] = (loop_audio[:fade_samples] * fade_in) + (tail * fade_out)
            
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
    parser.add_argument("-c", "--count", type=int, default=5, help="Количество лупов для поиска (по умолчанию: 5)")
    parser.add_argument("-w", "--window", type=int, default=10, help="Строгость поиска/размер окна в кадрах (по умолчанию: 10)")
    parser.add_argument("--min-dist", type=float, default=3.0, help="Минимальная дистанция между лупами в секундах (по умолчанию: 3.0)")
    parser.add_argument("--crossfade", type=float, default=0.05, help="Длительность кроссфейда в секундах (по умолчанию: 0.05)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[-] Ошибка: Файл {args.input} не найден.")
        sys.exit(1)
        
    y, sr = load_audio(args.input)
    if y is None or sr is None: 
        sys.exit(1)
    
    beat_frames, beat_times, chromagram = extract_features(y, sr)
    if beat_frames is None or beat_times is None or chromagram is None: 
        sys.exit(1)
    
    top_loops = find_best_loop(beat_frames, beat_times, chromagram, args.min, args.max, args.count, args.window, args.min_dist)
    if not top_loops: 
        sys.exit(1)
    
    base_name, ext = os.path.splitext(args.output)
    
    for i, (start_t, end_t) in enumerate(top_loops):
        if args.count == 1:
            output_path = args.output
        else:
            output_path = f"{base_name}_{i+1}{ext}"
            
        # Передаем новый параметр args.crossfade в функцию экспорта
        export_loop(y, sr, start_t, end_t, output_path, args.crossfade)
        
    print("[+] Готово! Скрипт завершил работу.")

if __name__ == '__main__':
    main()