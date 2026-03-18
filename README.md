# ASL Browser Trainer

Интерактивный тренажёр жестов на базе MediaPipe Holistic и `onnxruntime-web`.

Проект больше не зависит от Python-бэкенда для веб-интерфейса: текущая модель запускается прямо в браузере. Это позволяет хостить сайт как обычный статический проект на GitHub Pages, Netlify или через локальный `http.server`.

## Что сейчас есть

- браузерный тренажёр с целевым знаком, прогрессом удержания, подсказками и диагностикой камеры
- отдельная страница live-теста модели
- экспортированная ONNX-модель в репозитории
- Python pipeline для подготовки датасета, extraction landmark-фич, обучения GRU-модели и экспорта в ONNX
- curated label set для бытовых ASL-знаков

Текущая встроенная модель:

- файл: [`models/asl_citizen_50.onnx`](/D:/Integration-Game/gesture-trainer-web/models/asl_citizen_50.onnx)
- metadata: [`models/asl_citizen_50_metadata.json`](/D:/Integration-Game/gesture-trainer-web/models/asl_citizen_50_metadata.json)
- источник: baseline, обученный на `ASL Citizen` subset из `50` классов

## Быстрый старт

### Веб-версия

```powershell
cd D:\Integration-Game\gesture-trainer-web
python -m http.server 4174
```

Открыть:

- `http://127.0.0.1:4174/`
- `http://127.0.0.1:4174/model-test.html`

### Что открывать

- `index.html` — основной trainer с целевыми знаками, подсказками, удержанием и коучингом
- `model-test.html` — минимальная страница для проверки модели, камеры и top-predictions

## Текущая архитектура

### В браузере

- `index.html` и `model-test.html` загружают:
  - `onnxruntime-web`
  - MediaPipe Holistic
  - локальные JS-модули из [`js`](/D:/Integration-Game/gesture-trainer-web/js)
- камера и landmarks обрабатываются на клиенте
- landmark-последовательность длиной `40` кадров подаётся в ONNX-модель
- top predictions считаются прямо в браузере

### В Python

Python сейчас нужен для:

- подготовки dataset manifests
- построения subset по explicit label list
- extraction features из видео
- обучения модели
- экспорта PyTorch checkpoint в ONNX

Локальный FastAPI-бэкенд всё ещё лежит в проекте как вспомогательный инструмент, но для веб-интерфейса он больше не обязателен.

## Структура проекта

```text
gesture-trainer-web/
  index.html
  model-test.html
  styles.css
  js/
    gesture-trainer.js
    model-test.js
    sign-model-runtime.js
  models/
    asl_citizen_50.onnx
    asl_citizen_50_metadata.json
  python/
    build_asl_citizen_manifest.py
    build_wlasl_manifest.py
    prepare_wlasl_subset.py
    extract_sign_features.py
    train_sign_model.py
    export_sign_model_onnx.py
    local_inference_server.py
    label_sets/
      asl_citizen_daily_v1.txt
  docs/
    ARCHITECTURE.md
    TRAINING_PIPELINE.md
    TROUBLESHOOTING.md
```

## Основные файлы

- [`js/sign-model-runtime.js`](/D:/Integration-Game/gesture-trainer-web/js/sign-model-runtime.js)  
  Общий runtime для browser inference: загрузка ONNX, softmax, нормализация landmarks, сбор feature vectors, запуск модели, работа с камерой.

- [`js/gesture-trainer.js`](/D:/Integration-Game/gesture-trainer-web/js/gesture-trainer.js)  
  Основной trainer UI: целевые знаки, hold progress, коучинг по видимости, zone checking, guide card, диагностическая панель.

- [`js/model-test.js`](/D:/Integration-Game/gesture-trainer-web/js/model-test.js)  
  Упрощённый тест модели без механики тренировки.

- [`python/train_sign_model.py`](/D:/Integration-Game/gesture-trainer-web/python/train_sign_model.py)  
  Обучение GRU-классификатора на landmark-последовательностях.

- [`python/export_sign_model_onnx.py`](/D:/Integration-Game/gesture-trainer-web/python/export_sign_model_onnx.py)  
  Экспорт чекпойнта в ONNX + metadata JSON для браузера.

- [`python/label_sets/asl_citizen_daily_v1.txt`](/D:/Integration-Game/gesture-trainer-web/python/label_sets/asl_citizen_daily_v1.txt)  
  Подготовленный список бытовых ASL-знаков для следующего curated training run.

## Текущий статус модели

Базовый результат, который уже был обучен:

- датасет: `ASL Citizen`
- subset: `50` классов
- объём: `1901` видео
- extracted features shape: `(1901, 40, 154)`
- baseline validation accuracy: примерно `56.35%`

Что это значит:

- pipeline рабочий
- модель пригодна как prototype / demo baseline
- качество ещё не production-level
- для более стабильной UX лучше переходить на curated everyday subset и/или уменьшать количество конфликтующих классов

## Curated everyday labels

Для следующего запуска уже подготовлен curated everyday set:

- [`python/label_sets/asl_citizen_daily_v1.txt`](/D:/Integration-Game/gesture-trainer-web/python/label_sets/asl_citizen_daily_v1.txt)

Он содержит `40` практичных знаков:

- `HELLO`
- `BYE`
- `YES`
- `NO`
- `PLEASE`
- `SORRY`
- `HELP`
- `THANKYOU`
- `WELCOME1`
- `EAT1`
- `DRINK1`
- `WATER`
- `MOTHER`
- `FATHER`
- `FAMILY`
- `HOME`
- `HOUSE`
- `SCHOOL`
- `WORK`
- `FRIEND`
- `LOVE`
- `WANT1`
- `NEED`
- `COME`
- `COMEHERE`
- `GO`
- `STOP`
- `FINISH`
- `GOOD`
- `BAD`
- `HAPPY`
- `SAD`
- `NOW`
- `MORE`
- `NOT`
- `KNOW`
- `DONTKNOW`
- `NOTUNDERSTAND`
- `GOAHEAD`
- `GREAT`

Важно:

- буквы пока не смешиваются с этой моделью
- по буквам в доступных данных мало примеров, поэтому алфавит лучше учить отдельным экспериментом или на отдельном датасете

## Локальный Python backend

Хотя веб-интерфейс уже работает без backend, локальный FastAPI-сервер всё ещё можно поднять для:

- smoke tests старого API
- отладки PyTorch checkpoint без экспорта в ONNX
- локальной проверки `/api/health` и `/api/predict`

Пример:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn local_inference_server:app --host 127.0.0.1 --port 8000
```

Но для обычного использования сайта это уже не нужно.

## Подробная документация

- [Architecture](./docs/ARCHITECTURE.md)
- [Training Pipeline](./docs/TRAINING_PIPELINE.md)
- [Troubleshooting](./docs/TROUBLESHOOTING.md)

## Краткий workflow для переобучения

1. Скачать и распаковать `ASL Citizen`
2. Построить полный manifest
3. Собрать subset через `prepare_wlasl_subset.py`
4. Извлечь features через `extract_sign_features.py`
5. Обучить модель через `train_sign_model.py`
6. Экспортировать checkpoint в ONNX через `export_sign_model_onnx.py`
7. Подменить файлы в `models/`
8. Перезапустить или заново открыть статический сайт

Полные команды и пояснения лежат в [Training Pipeline](./docs/TRAINING_PIPELINE.md).

## Ограничения

- текущая встроенная модель всё ещё baseline
- многие похожие знаки путаются
- качество сильно зависит от устойчивой видимости корпуса, рук и лица
- для words-only модели нельзя автоматически ожидать хорошее распознавание букв
- GitHub Pages и Netlify подходят только для статической browser inference версии, но не для Python inference

## Развёртывание

Подходит любой статический хостинг:

- GitHub Pages
- Netlify
- Vercel static
- локальный `python -m http.server`

Главное условие:

- сайт должен иметь доступ к файлам из `models/`
- браузер должен разрешать камеру
- страница должна загружать CDN-зависимости `onnxruntime-web` и MediaPipe Holistic

## Полезные заметки

- если на странице `404` по `/api/health`, значит открыт старый backend-ориентированный URL или старый закэшированный build
- если камера даёт `NotFoundError`, нужно проверить разрешения браузера, выбрать реальное устройство и освободить камеру от Zoom/Discord/OBS
- если Python extraction падает на `libGL.so.1`, нужно установить системный пакет `libgl1`

