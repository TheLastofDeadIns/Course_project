import numpy as np
from collections import Counter
import math
from scipy.io import wavfile
from scipy.signal import butter, filtfilt


def load_audio(file_path):
    """Загружает аудиофайл и возвращает сырой сигнал и частоту дискретизации."""
    sample_rate, data = wavfile.read(file_path)
    # Обработка моно/стерео сигнала
    y = data if data.ndim == 1 else data[:, 0]
    # Нормализация амплитуды
    return y / np.max(np.abs(y)), sample_rate


def bandpass_filter(y, sample_rate, lowcut=1000, highcut=1450):
    """Применяет полосовой фильтр 680-1000 Гц к сигналу."""
    if sample_rate < 2 * highcut:
        raise ValueError(
            f"Частота дискретизации ({sample_rate} Гц) слишком мала. "
            f"Требуется минимум {2 * highcut} Гц для анализа до {highcut} Гц."
        )

    nyquist = 0.5 * sample_rate
    b, a = butter(
        N=5,  # Порядок фильтра
        Wn=[lowcut / nyquist, highcut / nyquist],
        btype='band',
        analog=False
    )
    return filtfilt(b, a, y)


def split_and_trim_audio(y, trim_percent=0.07):
    """Разделяет аудио на 3 части и обрезает 7% с каждого конца первой части."""
    part_size = len(y) // 3
    first_part = y[:part_size]
    trim_size = int(len(first_part) * trim_percent)
    return first_part[trim_size:-trim_size] if trim_size > 0 else first_part


def calculate_entropy_metrics(y_filtered, m):
    """Вычисляет энтропийные метрики для отфильтрованного сигнала."""
    # Создание z-векторов
    z_vectors = [y_filtered[i:i + m] for i in range(len(y_filtered) - m + 1)]

    # Создание xi-векторов (перестановочных паттернов)
    xi_vectors = [tuple(np.argsort(z)) for z in z_vectors]

    # Расчет вероятностей
    counter = Counter(xi_vectors)
    M = len(counter)
    probabilities = {xi: count / len(xi_vectors) for xi, count in counter.items()}
    P = list(probabilities.values())

    # Энтропия
    entropy_P = -sum(p * math.log(p) for p in P if p > 0)
    entropy_P_e = math.log(M) if M > 0 else 0
    normalized_entropy = entropy_P / entropy_P_e if entropy_P_e > 0 else 0

    # Сложность
    if M > 0:
        P_e = [1 / M] * M
        mixed = [(p + pe) / 2 for p, pe in zip(P, P_e)]
        entropy_mixed = -sum(pm * math.log(pm) for pm in mixed if pm > 0)
        jsd = entropy_mixed - (entropy_P / 2) - (entropy_P_e / 2)
        denominator = ((M + 1) / M) * math.log(M + 1) - 2 * math.log(2 * M) + math.log(M)
        C = (-2 / denominator * jsd * normalized_entropy) if denominator != 0 else 0
    else:
        C = 0

    return normalized_entropy, C


def hurst_exponent(time_series):
    """Вычисляет индекс Херста методом R/S анализа."""
    n = len(time_series)
    min_block_size = max(10, n // 100)  # Минимум 10 точек или 1% от длины
    max_block_size = n // 4
    log_rs, log_n = [], []

    for block_size in range(min_block_size, max_block_size + 1, max(min_block_size // 5, 1)):
        num_blocks = n // block_size
        rs_values = []

        for i in range(num_blocks):
            block = time_series[i * block_size:(i + 1) * block_size]
            if len(block) < 2:
                continue

            mean_block = np.mean(block)
            cumulative_deviation = np.cumsum(block - mean_block)
            R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
            S = np.std(block, ddof=1)
            if S > 0:
                rs_values.append(R / S)

        if rs_values:
            log_rs.append(np.log(np.mean(rs_values)))
            log_n.append(np.log(block_size))

    return np.polyfit(log_n, log_rs, 1)[0] if len(log_n) >= 2 else 0.5


def analyze_audio(file_path, m=3):
    """
    Полный анализ аудиофайла в диапазоне 680-1000 Гц:
    1. Загрузка -> 2. Обрезка -> 3. Фильтрация -> 4. Анализ
    """
    try:
        # 1. Загрузка
        y_raw, sr = load_audio(file_path)

        # Проверка частоты дискретизации
        if sr < 2000:
            raise ValueError(f"Частота дискретизации {sr} Гц слишком мала")

        # 2. Разделение и обрезка
        y_trimmed = split_and_trim_audio(y_raw)
        if len(y_trimmed) < 100:  # Минимум 100 отсчетов
            raise ValueError("Слишком короткий сигнал после обрезки")

        # 3. Фильтрация (600-1000 Гц)
        y_filtered = bandpass_filter(y_trimmed, sr)

        # 4. Расчет метрик
        m = min(m, len(y_filtered) // 10)  # Автоподбор m
        entropy, complexity = calculate_entropy_metrics(y_filtered, m)
        H = hurst_exponent(y_filtered)

        return entropy, complexity, H

    except Exception as e:
        print(f"Ошибка анализа {file_path}: {str(e)}")
        return None, None, None