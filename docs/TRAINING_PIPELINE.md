# Training Pipeline

## 1. Цель

Этот документ описывает полный workflow обучения новой модели:

1. скачать датасет
2. построить manifest
3. отобрать subset
4. извлечь landmark features
5. обучить PyTorch-модель
6. экспортировать модель в ONNX
7. подменить browser model в проекте

Основной рекомендуемый датасет сейчас:

- `ASL Citizen`

Почему:

- единый официальный архив
- меньше проблем, чем у WLASL со старыми внешними ссылками
- подходит для isolated sign recognition

## 2. Требования

### 2.1 Локально

- Python `3.10+`
- `pip`
- `venv`

### 2.2 На Linux GPU-сервере

Минимально полезно:

- NVIDIA GPU
- `4+ CPU`
- `16+ GB RAM`
- достаточно диска для архива, распаковки и features

Для OpenCV на Ubuntu может понадобиться:

```bash
sudo apt update
sudo apt install -y unzip libgl1 libglib2.0-0
```

## 3. Установка Python dependencies

```bash
cd ~/workspace/Signs-Recognition/python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Скачать ASL Citizen

```bash
mkdir -p ~/workspace/datasets/asl_citizen
cd ~/workspace/datasets/asl_citizen
wget https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip
unzip -q ASL_Citizen.zip
```

Ожидаемая структура:

```text
~/workspace/datasets/asl_citizen/
  ASL_Citizen.zip
  ASL_Citizen/
    splits/
    videos/
```

## 5. Построить manifest

```bash
cd ~/workspace/Signs-Recognition/python
source .venv/bin/activate

python build_asl_citizen_manifest.py \
  --dataset-root ~/workspace/datasets/asl_citizen/ASL_Citizen \
  --output ~/workspace/datasets/asl_citizen/asl_citizen_manifest.jsonl
```

Этот шаг:

- читает официальные split CSV
- нормализует пути к видео
- пишет JSONL manifest

## 6. Собрать subset

### 6.1 Автовыбор top classes

```bash
python prepare_wlasl_subset.py \
  --manifest ~/workspace/datasets/asl_citizen/asl_citizen_manifest.jsonl \
  --output ~/workspace/datasets/asl_citizen/asl_citizen_50_manifest.jsonl \
  --stats-output ~/workspace/datasets/asl_citizen/asl_citizen_50_stats.json \
  --num-classes 50 \
  --min-samples-per-class 20 \
  --max-train-per-class 80 \
  --max-val-per-class 20
```

### 6.2 Curated subset по explicit labels

Для everyday signs:

```bash
python prepare_wlasl_subset.py \
  --manifest ~/workspace/datasets/asl_citizen/asl_citizen_manifest.jsonl \
  --labels-file ~/workspace/Signs-Recognition/python/label_sets/asl_citizen_daily_v1.txt \
  --output ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_manifest.jsonl \
  --stats-output ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_stats.json \
  --max-train-per-class 120 \
  --max-val-per-class 30
```

Важно:

- если в самих данных у знака только `30` примеров, большие лимиты не помогут
- лимиты — это потолок, а не гарантированный размер класса

## 7. Проверить размер subset

Пример:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_stats.json').expanduser()
data = json.loads(path.read_text(encoding='utf-8'))

print('num_classes =', data['num_classes'])
print('total_samples =', data['total_samples'])
for item in data['classes']:
    print(item['label'], item['train_samples'], item['val_samples'], item['total_samples'])
PY
```

## 8. Извлечь features

```bash
python extract_sign_features.py \
  --manifest ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_manifest.jsonl \
  --output ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_features.npz \
  --max-frames 40
```

Что делает extraction:

- открывает каждое видео
- прогоняет кадры через MediaPipe Holistic
- берёт:
  - левую руку
  - правую руку
  - верхнюю часть pose landmarks
- нормализует координаты
- собирает fixed-length sequence
- сохраняет `.npz`

Результат:

- `sequences`
- `labels`
- `splits`
- `label_names`
- `feature_size`

## 9. Проверить feature-файл

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

p = Path('~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_features.npz').expanduser()
data = np.load(p, allow_pickle=True)
print('sequences shape =', data['sequences'].shape)
print('labels shape =', data['labels'].shape)
print('num_classes =', len(data['label_names']))
print('feature_size =', int(data['feature_size'][0]))
PY
```

## 10. Обучить модель

Пример baseline:

```bash
python train_sign_model.py \
  --features ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_features.npz \
  --output-dir ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_run \
  --epochs 16 \
  --batch-size 32 \
  --hidden-size 192
```

Ожидаемые выходы:

- `best_model.pt`
- `training_metrics.json`

## 11. Проверить метрики

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path('~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_run/training_metrics.json').expanduser()
data = json.loads(p.read_text(encoding='utf-8'))
print('best_val_accuracy =', data['best_val_accuracy'])
print('last_epoch =', data['history'][-1])
PY
```

## 12. Экспортировать в ONNX

```bash
python export_sign_model_onnx.py \
  --checkpoint ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_run/best_model.pt \
  --output ~/workspace/Signs-Recognition/models/asl_citizen_daily_v1.onnx \
  --metadata-output ~/workspace/Signs-Recognition/models/asl_citizen_daily_v1_metadata.json \
  --sequence-length 40 \
  --top-k 5
```

Что создаётся:

- `.onnx` файл
- `.json` metadata с `label_names`, `feature_size`, `sequence_length`

## 13. Подменить браузерную модель

Сейчас runtime ссылается на:

- `models/asl_citizen_50.onnx`
- `models/asl_citizen_50_metadata.json`

Есть два пути:

### Вариант A. Заменить существующие файлы

- оставить те же имена
- просто положить новые версии поверх старых

### Вариант B. Добавить новую модель и сменить ссылки в runtime

Поменять пути в:

- [`js/sign-model-runtime.js`](/D:/Integration-Game/gesture-trainer-web/js/sign-model-runtime.js)

Этот путь чище, если нужно держать несколько моделей.

## 14. Почему лучше не смешивать буквы и бытовые слова сразу

Практический вывод по данным:

- буквенные классы встречаются редко
- бытовые слова и буквы образуют разные по сложности задачи
- смешивание обычно ухудшает первую модель

Рекомендуемая стратегия:

- модель 1: бытовые слова
- модель 2: буквы / fingerspelling

## 15. Практические рекомендации

### Если мало времени

- не гнаться сразу за `100+` классами
- сначала обучить `20-50` полезных знаков
- проверить confusion pairs
- только потом расширять словарь

### Если данных мало на класс

- лучше уменьшить число классов
- чем держать много очень похожих классов с `~30` видео

### Если inference в браузере кажется слабым

Проверить:

- совпадает ли sequence length с metadata
- используется ли правильный ONNX и metadata JSON
- не осталось ли старое имя модели в runtime
- не открыт ли сайт из кэша

