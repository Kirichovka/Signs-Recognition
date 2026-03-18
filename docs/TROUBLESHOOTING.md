# Troubleshooting

## 1. Камера не запускается в браузере

### Симптом

- `NotFoundError: Requested device not found`
- `Camera stream is not ready yet`
- trainer зависает в состоянии `Waiting`

### Что проверить

1. Разрешение камеры в браузере:
   - иконка слева от адреса
   - `Camera -> Allow`

2. Не занята ли камера:
   - Zoom
   - Discord
   - OBS
   - Teams
   - Windows Camera

3. Открыт ли сайт по реальному адресу, а не старому табу/кэшу

4. Поддерживает ли устройство выбранные constraints

### Что уже сделано в коде

В runtime добавлены fallback-попытки:

- explicit `deviceId`
- `facingMode: "user"`
- video без `facingMode`
- `video: true`

Если даже после этого ошибка остаётся, проблема почти наверняка вне JS-логики:

- права браузера
- устройство камеры
- драйвер
- камера занята другой программой

## 2. GitHub Pages открывается, но `/api/health` даёт 404

### Причина

Открыта старая схема, рассчитанная на Python backend.

### Решение

Использовать текущую browser-only версию.

Актуальная архитектура:

- нет обязательного `/api/health`
- нет обязательного `/api/predict`
- inference идёт в браузере

Если видишь 404 на `/api/health`, значит:

- открыт старый URL
- или старая закэшированная версия JS

Сделай:

- `Ctrl+F5`
- открой новую вкладку
- проверь, что загрузился актуальный build

## 3. Серверный extraction падает с `libGL.so.1`

### Симптом

```text
ImportError: libGL.so.1: cannot open shared object file
```

### Решение

На Ubuntu:

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0
```

## 4. На сервере нет `unzip`

### Решение

```bash
sudo apt update
sudo apt install -y unzip
```

## 5. WLASL downloader даёт много `Unsuccessful downloading`

### Причина

Это нормальная проблема старого WLASL workflow:

- часть ссылок устарела
- часть видео были на внешних сайтах
- часть ссылок вела на `swf` или мёртвые ресурсы

### Практическое решение

Использовать `ASL Citizen` как основной датасет для быстрых и надёжных train runs.

## 6. Модель в браузере грузится, но качество кажется слабым

### Возможные причины

- baseline accuracy ограничена качеством исходной модели
- в текущем наборе слишком много конфликтующих классов
- пользователь делает знак не в той зоне
- камера плохо видит лицо/корпус/руки
- модель и metadata не совпадают

### Что делать

1. Проверить, что `.onnx` и metadata относятся к одной и той же модели
2. Проверить coaching/diagnostics на странице
3. Попробовать curated subset с более бытовыми и визуально различимыми знаками
4. Уменьшить число классов

## 7. Почему большие лимиты в subset не увеличивают число видео

### Пример

Указано:

- `--max-train-per-class 120`
- `--max-val-per-class 30`

Но по факту получается только около `30` видео на класс.

### Причина

В самих данных для этого знака больше нет примеров.

Лимиты в `prepare_wlasl_subset.py`:

- обрезают сверху
- но не создают новые данные

## 8. Почему буквы пока не входят в everyday-модель

### Причина

- по буквам мало примеров
- часть букв отсутствует как хороший полноценный класс
- смешивание букв с everyday words почти наверняка испортит первую модель

### Что делать

Обучать отдельно:

- words model
- alphabet model

## 9. Python backend поднимается, но сайт на GitHub Pages его не видит

### Причина

GitHub Pages — это статический хостинг.

Он не умеет:

- запускать FastAPI
- выполнять PyTorch inference
- обслуживать локальный процесс Python как “встроенный backend”

### Решение

Либо:

- использовать browser-only ONNX inference

Либо:

- держать backend отдельно на сервере

## 10. Как понять, что текущая страница работает на новой модели

Проверить:

- какие label names отображаются
- какое имя модели/metadata зашито в runtime
- нет ли старых API-вызовов
- не закэшировалась ли старая версия JS

Полезные точки:

- [`js/sign-model-runtime.js`](/D:/Integration-Game/gesture-trainer-web/js/sign-model-runtime.js)
- [`models/asl_citizen_50.onnx`](/D:/Integration-Game/gesture-trainer-web/models/asl_citizen_50.onnx)
- [`models/asl_citizen_50_metadata.json`](/D:/Integration-Game/gesture-trainer-web/models/asl_citizen_50_metadata.json)

