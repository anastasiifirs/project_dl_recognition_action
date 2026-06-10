# Полное описание проекта: регистрация событий на видеозаписи

## 1. Название проекта

**Регистрация событий на видеозаписи**

Проект относится к области Deep Learning и Computer Vision. Его цель - построить систему, которая анализирует видеозапись с человеком и автоматически регистрирует действия как временные события.

Итоговый результат работы модели - не просто одна метка для всего видео, а список событий:

```json
{
  "label": "push_ups",
  "start_sec": 31.62,
  "end_sec": 38.42,
  "avg_confidence": 0.967,
  "max_confidence": 0.999,
  "repetition_count": 4
}
```

## 2. Основная идея

Обычная классификация видео отвечает на вопрос: **"Что происходит в этом видео?"**

В этом проекте решается более практическая задача: **"Какие действия происходили, в какие моменты они начались и закончились, и сколько повторений было выполнено?"**

Система должна:

- открыть длинное видео;
- нарезать его на временные окна;
- прогнать окна через нейросетевую модель;
- сгладить поток предсказаний;
- объединить стабильные предсказания в события;
- сохранить результат в машинно-читаемом виде;
- построить визуальную timeline;
- создать видео с overlay-разметкой.

## 3. Классы действий

В проекте используется 8 классов:

```text
stand
walk
run
jump
push_ups
squat
bend
other
```

Описание классов:

| Класс | Значение |
|---|---|
| `stand` | человек стоит |
| `walk` | ходьба |
| `run` | бег |
| `jump` | прыжки |
| `push_ups` | отжимания |
| `squat` | приседания |
| `bend` | наклоны |
| `other` | любые другие движения, не входящие в основные классы |

Класс `other` нужен как fallback-класс. Он сложнее остальных, потому что включает много разных движений: махи руками, потягивания, удары ногой, переходы между упражнениями и другие нестандартные действия.

## 4. Использованные данные

Финальная версия проекта использует локальный датасет:

```text
data/dev/
```

Несмотря на название `dev`, сейчас это основной рабочий набор данных проекта.

Структура:

```text
data/dev/
  raw/
    1.MOV
    2.MOV
    ...
    IMG_3644.mov
  annotations.csv
  splits/
    train.txt
    val.txt
    test.txt
  processed/
    train.csv
    val.csv
    test.csv
```

Количество данных:

```text
82 видео
716 размеченных событий
8 классов
```

CSV-разметка:

```text
data/dev/annotations.csv
```

Формат CSV:

```csv
video,start_sec,end_sec,label
IMG_3609.mov,0,5,stand
IMG_3609.mov,5,10,walk
IMG_3609.mov,10,15,run
```

Колонки:

| Колонка | Описание |
|---|---|
| `video` | имя видеофайла из `data/dev/raw/` |
| `start_sec` | начало события в секундах |
| `end_sec` | конец события в секундах |
| `label` | класс действия |

Поддерживаемые форматы видео:

```text
.mp4
.mov
.avi
.mkv
```

Звук в проекте не используется. Модель работает только с визуальными кадрами.

## 5. Распределение событий по классам

По результатам validation:

```text
bend:     80 событий
jump:     91 событие
other:    99 событий
push_ups: 62 события
run:      93 события
squat:    83 события
stand:    122 события
walk:     86 событий
```

Суммарная длительность событий по классам:

```text
bend:     648.0 сек
jump:     629.0 сек
other:    631.6 сек
push_ups: 491.0 сек
run:      655.5 сек
squat:    622.0 сек
stand:    554.5 сек
walk:     579.0 сек
```

## 6. Validation датасета

Для проверки данных реализован модуль:

```text
src/event_video_recognition/validation.py
```

И CLI-скрипт:

```text
scripts/validate_dataset.py
```

Validation проверяет:

- существует ли `annotations.csv`;
- есть ли обязательные колонки `video,start_sec,end_sec,label`;
- существуют ли все видео из CSV;
- поддерживается ли расширение видео;
- открывается ли видео через OpenCV;
- можно ли получить `fps`, `width`, `height`, `frames`, `duration_sec`;
- нет ли отрицательных времен;
- выполняется ли `end_sec > start_sec`;
- входят ли labels в список допустимых классов;
- не выходит ли событие за длительность видео.

Финальный validation сохранен:

```text
outputs/final_dataset_validation.json
```

Результат:

```text
ok: true
errors: []
warnings: []
```

## 7. Разделение данных

Split выполняется по исходным видео, а не случайно по событиям.

Это важно: если случайно делить события, то разные интервалы одного и того же видео могут попасть и в train, и в test. Тогда метрики будут завышены, потому что модель уже видела тот же фон, человека, одежду и ракурс.

Финальное разделение:

```text
train: 57 видео, 489 событий
val:   12 видео, 107 событий
test:  13 видео, 120 событий
```

Файлы split:

```text
data/dev/splits/train.txt
data/dev/splits/val.txt
data/dev/splits/test.txt
```

CSV для обучения:

```text
data/dev/processed/train.csv
data/dev/processed/val.csv
data/dev/processed/test.csv
```

Скрипт:

```text
scripts/prepare_splits.py
```

Команда:

```bash
PYTHONPATH=src python scripts/prepare_splits.py \
  --config configs/final.yaml \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

## 8. Структура проекта

Финальная структура ключевых файлов:

```text
project_dl_recognition_action/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore

  configs/
    final.yaml

  data/
    dev/
      raw/
      annotations.csv
      splits/
      processed/

  src/
    event_video_recognition/
      __init__.py
      config.py
      validation.py
      video.py
      dataset.py
      models.py
      events.py
      metrics.py
      visualization.py
      pipeline.py
      repetitions.py

  scripts/
    validate_dataset.py
    cut_clips.py
    prepare_splits.py
    train.py
    infer_video.py
    evaluate.py
    evaluate_all.py
    evaluate_split.py
    infer_webcam.py
    add_repetition_counts.py
    calibrate_temperature.py
    plot_training_curves.py
    render_annotated_from_events.py
    run_strong_pipeline.py
    train_ensemble.py

  app/
    streamlit_app.py

  models/
    final_checkpoint.pt
    final_metrics.json
    temperature.json
    training_curves.png

  outputs/
    final_dataset_validation.json
    final/
      test_eval/
        summary_metrics.json
        per_video_metrics.csv
        confusion_matrix.png

  tests/
    test_events.py
    test_metrics.py
    test_repetitions.py
    test_validation.py
    test_visualization.py

  docs/
    full_project_description.md
    project_summary.md
    report.md
    training_strategy.md
```

## 9. Назначение основных модулей

### `config.py`

Загрузка YAML-конфига проекта.

Главный конфиг:

```text
configs/final.yaml
```

### `validation.py`

Проверка CSV-разметки и видеофайлов.

### `video.py`

Видео-утилиты:

- чтение кадров;
- letterbox-препроцессинг;
- работа с видео;
- подготовка кадров к модели.

### `dataset.py`

PyTorch Dataset для обучения на размеченных временных интервалах.

Особенность: чтение кадров оптимизировано так, чтобы не делать `capture.set(...)` для каждого кадра. Это ускоряет работу с `.MOV`.

### `models.py`

Создание модели:

```text
torchvision.models.video.r3d_18
```

Финальный слой заменяется под 8 классов.

### `events.py`

Event Registry. Превращает поток предсказаний модели в устойчивые события.

### `metrics.py`

Метрики:

- segment IoU;
- event-level precision;
- event-level recall;
- event-level F1;
- mean IoU;
- latency.

### `visualization.py`

Построение `timeline.png`.

### `pipeline.py`

Общий offline inference pipeline:

```text
video -> sliding windows -> model predictions -> EventRegistry -> outputs
```

### `repetitions.py`

Подсчет повторений для периодических действий:

- `push_ups`;
- `squat`;
- `bend`;
- `jump`.

## 10. Конфигурация

Файл:

```text
configs/final.yaml
```

Ключевые параметры:

```yaml
model:
  architecture: r3d_18
  checkpoint: models/final_checkpoint.pt
  pretrained: true
  clip_len: 16
  frame_stride: 2
  image_size: 112
```

Модель принимает вход:

```text
[B, C, T, H, W]
```

Где:

- `B` - batch size;
- `C` - каналы изображения;
- `T` - число кадров;
- `H` - высота;
- `W` - ширина.

Inference-параметры:

```yaml
inference:
  infer_every_frames: 4
  confidence_threshold: 0.55
  smoothing_window: 5
  min_event_duration_sec: 1.2
  merge_gap_sec: 1.0
  draw_overlay: true
  fill_gaps_with_other: false
```

Для честного inference пропуски не заполняются автоматически классом `other`.

В Streamlit-режиме `Presentation full` можно включить заполнение пропусков для более непрерывной демонстрационной timeline, но это именно демонстрационная настройка.

## 11. Модель

Используемая архитектура:

```text
r3d_18
```

Источник:

```text
torchvision.models.video
```

Почему выбрана 3D-CNN:

- учитывает не только отдельные кадры, но и движение во времени;
- подходит для action recognition;
- доступна pretrained-версия;
- работает без отдельной skeleton-разметки;
- проще поддерживать, чем старые TensorFlow/OpenPose pipelines.

Финальный checkpoint:

```text
models/final_checkpoint.pt
```

Размер checkpoint:

```text
примерно 127-128 MB
```

## 12. Стратегия обучения

Обучение сделано не "с нуля", а через fine-tuning pretrained video backbone.

Используется staged fine-tuning:

1. `head_warmup`
   - backbone заморожен;
   - обучается новая классификационная голова;
   - задача этапа: быстро адаптировать выход модели под 8 классов.

2. `last_block_finetune`
   - размораживается последний 3D-блок `layer4`;
   - модель аккуратно адаптируется к локальному датасету;
   - риск переобучения ниже, чем при полном размораживании всей сети.

Дополнительные элементы:

- pretrained `r3d_18`;
- `WeightedRandomSampler`;
- class weights;
- label smoothing;
- augmentations;
- cosine learning rate schedule;
- сохранение лучшего checkpoint;
- поддержка `cpu`, `cuda`, `mps`.

Команда обучения:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/final.yaml \
  --output-dir models \
  --device mps
```

## 13. История обучения

Файл:

```text
models/final_metrics.json
```

Итог:

```text
best_val_acc  = 0.9813084112
best_val_loss = 0.3415339447
```

Последняя эпоха:

```text
epoch:      10
stage:      last_block_finetune
train_loss: 0.58924
train_acc:  0.89571
val_loss:   0.34153
val_acc:    0.98131
lr:         0.00001
```

Важно: validation-метрика высокая, но ее нельзя считать абсолютным доказательством production-качества. Данные сняты в близких условиях, поэтому главная честная оценка - test split.

## 14. Test evaluation

Финальная оценка:

```text
outputs/final/test_eval/summary_metrics.json
```

Test split:

```text
13 видео
120 ground-truth событий
127 predicted событий
```

Итоговые метрики:

```text
true_positives:  104
false_positives: 23
false_negatives: 16

precision: 0.8189
recall:    0.8667
f1:        0.8421
mean_iou:  0.6658
latency:   1.5105 sec
```

Интерпретация:

- `precision = 0.8189`: примерно 82% найденных событий являются корректными;
- `recall = 0.8667`: модель находит примерно 87% реальных событий;
- `f1 = 0.8421`: хороший баланс между лишними и пропущенными событиями;
- `mean_iou = 0.6658`: временные границы событий совпадают с разметкой умеренно хорошо;
- `latency = 1.5105 sec`: среднее смещение границ события относительно ground truth.

## 15. Метрики по классам

```text
push_ups: precision 1.0000, recall 1.0000, F1 1.0000
squat:   precision 1.0000, recall 0.9333, F1 0.9655
run:     precision 0.8824, recall 0.8824, F1 0.8824
jump:    precision 0.8667, recall 0.8667, F1 0.8667
other:   precision 0.8571, recall 0.8571, F1 0.8571
stand:   precision 0.7241, recall 0.8400, F1 0.7778
walk:    precision 0.7143, recall 0.8333, F1 0.7692
bend:    precision 0.6429, recall 0.7500, F1 0.6923
```

Per-class IoU:

```text
stand:    0.5311
walk:     0.6906
run:      0.7198
jump:     0.6734
push_ups: 0.6506
squat:    0.7284
bend:     0.7679
other:    0.6684
```

Сильные классы:

- `push_ups`;
- `squat`;
- `run`;
- `jump`;
- `other`.

Более слабые классы:

- `bend`;
- `walk`;
- `stand`.

Причины:

- `bend` визуально похож на переходы, подготовку к упражнению и часть движений `other`;
- `walk` и `stand` могут смешиваться на коротких переходах;
- временные границы между спокойными действиями субъективнее, чем между активными упражнениями.

## 16. Дополнительные улучшения после финального обучения

В проект добавлены дополнительные инженерные улучшения:

- `boundary_refinement`: уточняет границы события по raw predictions в окне около `start_sec` и `end_sec`;
- `temperature_scaling`: калибрует confidence модели по validation split и сохраняет `models/temperature.json`;
- `per_class_iou`: добавляет mean IoU по каждому классу;
- `confusion_matrix.png`: строит event-level confusion matrix по matched events;
- `training_curves.png`: визуализирует train/val loss и accuracy по эпохам;
- `load_ensemble`: добавляет инфраструктуру для нескольких checkpoints без поломки одиночного режима;
- safe pose fallback: если `models/pose_landmarker_full.task` отсутствует, inference не падает, а счетчик повторений помечается как `method="unavailable"`.

Новые артефакты:

```text
models/temperature.json
models/training_curves.png
outputs/final/test_eval/confusion_matrix.png
```

## 17. Слабые видео

В test split есть отдельные видео, где результат хуже среднего:

```text
10.MOV
7.MOV
```

На них модель хуже переносит выученные признаки. Это нормальная и честная картина для пользовательского датасета: среднее качество хорошее, но отдельные нестандартные видео остаются сложными.

## 18. Inference

Скрипт:

```text
scripts/infer_video.py
```

Команда:

```bash
PYTHONPATH=src python scripts/infer_video.py \
  --config configs/final.yaml \
  --video data/dev/raw/IMG_3609.mov \
  --checkpoint models/final_checkpoint.pt \
  --output-dir outputs/final/IMG_3609 \
  --device mps
```

Результаты inference:

```text
events.json
events.csv
timeline.png
annotated.mp4
inference_stats.json
```

## 19. Event Registry

Event Registry - один из главных компонентов проекта.

Он принимает поток предсказаний:

```python
Prediction(time_sec, label, confidence)
```

И возвращает список событий:

```python
Event(label, start_sec, end_sec, avg_confidence, max_confidence)
```

Логика:

- если confidence ниже порога, предсказание считается `unknown`;
- несколько соседних предсказаний сглаживаются через `smoothing_window`;
- короткие шумовые события отсекаются через `min_event_duration_sec`;
- одинаковые события через короткий разрыв склеиваются через `merge_gap_sec`;
- последнее событие корректно закрывается в конце видео.

Это нужно, потому что модель делает предсказания по окнам, а пользователю нужен понятный журнал действий.

## 20. Visualization

Модуль:

```text
src/event_video_recognition/visualization.py
```

Он строит timeline:

```text
timeline.png
```

Timeline показывает:

- какие события модель нашла;
- где они начинаются;
- где заканчиваются;
- какие классы были зарегистрированы.

При evaluation можно сравнивать predicted events и ground truth.

## 21. Annotated video

После inference создается:

```text
annotated.mp4
```

На видео отображается:

- текущий label действия;
- confidence;
- при наличии - счетчик повторений.

Это нужно для демонстрации проекта: можно открыть видео и визуально увидеть, как система регистрирует события.

## 22. Подсчет повторений

Добавлен отдельный post-processing модуль:

```text
src/event_video_recognition/repetitions.py
```

Он считает повторения для:

```text
jump
push_ups
squat
bend
```

Важно: счетчик не заменяет модель action recognition. Сначала 3D-CNN находит событие и его границы, затем счетчик анализирует этот интервал.

Подход:

- для `push_ups`, `squat`, `bend` используется pose-based логика, если доступна MediaPipe pose-модель;
- для `jump` используется motion-energy;
- если pose-сигнал слабый, применяется fallback по motion-energy.

В конфиге:

```yaml
repetition_counting:
  enabled: true
  method: auto
```

Pose model path:

```text
models/pose_landmarker_full.task
```

Папка `files-mentioned-by-the-user-telegram` не изменялась. Из нее были взяты идеи, но абсолютный путь к pose-модели из проекта убран. Если `models/pose_landmarker_full.task` отсутствует, inference продолжает работу, а repetition counting для pose-классов получает `method="unavailable"`.

## 23. Streamlit demo

Файл:

```text
app/streamlit_app.py
```

Запуск:

```bash
cd project_dl_recognition_action
PYTHONPATH=src streamlit run app/streamlit_app.py
```

По умолчанию Streamlit использует:

```text
configs/final.yaml
models/final_checkpoint.pt
```

Режимы:

| Режим | Назначение |
|---|---|
| `Presentation full` | полный демонстрационный режим с overlay и счетчиком |
| `Balanced` | компромисс скорости и качества |
| `Speed check` | быстрая проверка, не для оценки качества |

Важно: `Speed check` может давать плохую разметку, потому что делает меньше model calls. Для финальной демонстрации нужен `Presentation full`.

## 24. Оценка одного видео

Скрипт:

```text
scripts/evaluate.py
```

Пример:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/final.yaml \
  --predicted outputs/final/IMG_3609/events.json \
  --video-name IMG_3609.mov \
  --output-dir outputs/final/IMG_3609/eval
```

## 25. Оценка split

Скрипт:

```text
scripts/evaluate_split.py
```

Финальная команда:

```bash
PYTHONPATH=src python scripts/evaluate_split.py \
  --config configs/final.yaml \
  --checkpoint models/final_checkpoint.pt \
  --split test \
  --output-dir outputs/final/test_eval \
  --device mps
```

Результаты:

```text
outputs/final/test_eval/summary_metrics.json
outputs/final/test_eval/per_video_metrics.csv
```

## 26. Тесты

Папка:

```text
tests/
```

Тесты:

```text
test_events.py
test_metrics.py
test_repetitions.py
test_validation.py
test_visualization.py
```

Что проверяется:

- EventRegistry фильтрует короткий шум;
- EventRegistry применяет confidence threshold;
- EventRegistry склеивает одинаковые события;
- segment IoU считается корректно;
- validation ловит неверные labels;
- validation ловит неверные интервалы;
- timeline creation не падает на пустом списке;
- repetition counting работает на базовых случаях.

В текущем окружении `pytest` не установлен, поэтому полный pytest-run не запускался. Компиляция Python-файлов и smoke-check ключевых модулей прошли успешно.

Команда для запуска тестов после установки dev-зависимостей:

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

## 27. Проверка проекта

Была выполнена компиляция:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/project_dl_pycache_final python -m compileall src scripts app
```

Результат:

```text
успешно
```

Также был выполнен smoke-check:

```text
smoke ok: r3d_18 8 stand
```

## 28. Использованные референсы

Были изучены два GitHub-проекта:

1. `IIharlamovv/action-recognition-v1.0`
2. `felixchenfy/Realtime-Action-Recognition`

## 29. Что взято из `IIharlamovv/action-recognition-v1.0`

Из первого проекта переосмыслены идеи:

- action recognition по видеофрагментам;
- 3D-CNN / R3D / X3D подход;
- sliding window inference;
- CSV-разметка `video,start_sec,end_sec,label`;
- offline inference `video -> predictions`;
- экспорт результатов в JSON;
- letterbox-препроцессинг.

Важно: датасет из этого репозитория не использовался как полноценный обучающий датасет, потому что в репозитории нет всех обучающих видео. Он использовался только как референс формата и pipeline.

## 30. Что взято из `Realtime-Action-Recognition`

Из второго проекта взяты только универсальные инженерные идеи:

- temporal window;
- smoothing;
- confidence threshold;
- стабильная метка действия во времени;
- online/real-time inference как возможное расширение.

Не использовались:

- TensorFlow 1.x;
- OpenPose/tf-pose-estimation;
- PCA;
- ручные skeleton features;
- старый MLP-классификатор.

Причина: проект старый, и его технологическая база устарела. Для финальной версии использован современный PyTorch pipeline.

## 31. Что взято из локальной папки `files-mentioned-by-the-user-telegram`

Папка:

```text
files-mentioned-by-the-user-telegram
```

Файлы в этой папке не изменялись.

Полезные идеи:

- MediaPipe Pose;
- pose-based признаки;
- анализ последовательности поз;
- overlay предсказаний на видео;
- счетчик повторений;
- журнал событий.

В финальный проект перенесена не копия кода, а сама идея: использовать pose/motion post-processing для подсчета повторений поверх уже найденных событий.

## 32. Почему не только pose-модель

Pose-модель полезна для упражнений, где важны суставы:

- отжимания;
- приседания;
- наклоны.

Но только pose-подход может быть нестабилен:

- если человек частично закрыт;
- если плохое освещение;
- если камера далеко;
- если вертикальное видео с неидеальным ракурсом;
- если действие лучше определяется по общему движению, а не по углам суставов.

Поэтому в проекте основной классификатор - RGB video model `r3d_18`, а pose/motion используется как дополнительный post-processing для счетчика.

## 33. Почему не нужно использовать звук

Звук не нужен, потому что классы действий определяются визуально:

- ходьба;
- бег;
- прыжки;
- отжимания;
- приседания;
- наклоны;
- стойка;
- другие движения.

Использование звука усложнило бы проект, потребовало бы аудио pipeline и не дало бы надежной пользы для этих классов.

## 34. Почему `.mov` поддерживается

Исходные видео часто сняты на iPhone и имеют формат `.MOV`.

Проект поддерживает `.mov`, но для переносимости на другие машины `.mp4` может быть удобнее.

Если на другой машине `.MOV` плохо открывается через OpenCV, можно конвертировать видео в `.mp4` через ffmpeg.

## 35. Ограничения проекта

Текущая версия:

- рассчитана на одного человека в кадре;
- не решает multi-person tracking;
- зависит от качества разметки;
- может ошибаться на новых ракурсах и условиях;
- хуже различает `bend`, `walk`, `stand`;
- может давать задержку границ событий около 1-2 секунд.

Multi-person tracking оставлен как future work.

## 36. Что можно улучшить дальше

Наиболее полезные улучшения:

1. Добавить больше видео для слабых классов:
   - `bend`;
   - `walk`;
   - `stand`.

2. Добавить разнообразие:
   - другие люди;
   - другие фоны;
   - другое освещение;
   - другая одежда;
   - разные расстояния до камеры;
   - разные порядки действий.

3. Проверить и, если нужно, уточнить разметку:
   - границы событий;
   - короткие переходы;
   - класс `other`.

4. Сделать отдельную calibration-процедуру:
   - подобрать `confidence_threshold`;
   - подобрать `smoothing_window`;
   - подобрать `min_event_duration_sec`;
   - подобрать `merge_gap_sec`.

5. Добавить multi-person mode:
   - person detector;
   - tracker;
   - action recognition по каждому track.

6. Рассмотреть более сильные модели:
   - X3D;
   - Video Swin Transformer;
   - TimeSformer;
   - SlowFast.

Но для учебного проекта текущая версия уже показывает полный современный pipeline.

## 37. Главные команды

Проверка датасета:

```bash
PYTHONPATH=src python scripts/validate_dataset.py \
  --config configs/final.yaml \
  --output outputs/final_dataset_validation.json
```

Подготовка split:

```bash
PYTHONPATH=src python scripts/prepare_splits.py \
  --config configs/final.yaml \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

Обучение:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/final.yaml \
  --output-dir models \
  --device mps
```

Оценка test split:

```bash
PYTHONPATH=src python scripts/evaluate_split.py \
  --config configs/final.yaml \
  --checkpoint models/final_checkpoint.pt \
  --split test \
  --output-dir outputs/final/test_eval \
  --device mps
```

Inference:

```bash
PYTHONPATH=src python scripts/infer_video.py \
  --config configs/final.yaml \
  --video data/dev/raw/IMG_3609.mov \
  --checkpoint models/final_checkpoint.pt \
  --output-dir outputs/final/IMG_3609 \
  --device mps
```

Streamlit:

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

## 38. Эксперимент X3D-S

Для проверки альтернативной современной video architecture был добавлен отдельный эксперимент с `X3D-S` из `pytorchvideo`.

Артефакты эксперимента:

```text
configs/x3d_s.yaml
models/x3d_s_checkpoint.pt
models/x3d_s_metrics.json
models/x3d_s_temperature.json
models/x3d_s_training_curves.png
outputs/x3d_s/test_eval/summary_metrics.json
outputs/x3d_s/test_eval_tuned/summary_metrics.json
outputs/x3d_s/threshold_sweep_val/threshold_sweep.csv
outputs/x3d_s/test_eval/confusion_matrix.png
```

Сравнение с финальной `R3D-18`:

| Модель | Precision | Recall | F1 | Mean IoU | Latency | Размер checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| R3D-18 | 0.8189 | 0.8667 | 0.8421 | 0.6658 | 1.5105 sec | ~127 MB |
| X3D-S, threshold 0.55 | 0.7591 | 0.8667 | 0.8093 | 0.7355 | 1.1316 sec | ~12 MB |
| X3D-S, threshold 0.70 | 0.7704 | 0.8667 | 0.8157 | 0.7321 | 1.1573 sec | ~12 MB |

Финальная `R3D-18` обучалась с `image_size=112`, а `X3D-S` - с `image_size=160`. Это нужно явно учитывать при интерпретации: split и event-level evaluation одинаковые, но spatial context у моделей разный.

Для `X3D-S` была сделана отдельная temperature calibration:

```text
models/x3d_s_temperature.json
temperature = 0.1643
```

Для `R3D-18` используется:

```text
models/temperature.json
temperature = 0.6008
```

Для `X3D-S` был отдельно выполнен `confidence_threshold` sweep на validation split:

```text
outputs/x3d_s/threshold_sweep_val/threshold_sweep.csv
лучший threshold = 0.70
val F1 = 0.9358
val mean_iou = 0.7815
```

После выбора threshold на validation модель была один раз оценена на test split:

```text
outputs/x3d_s/test_eval_tuned/summary_metrics.json
```

Вывод по эксперименту: `R3D-18` остается финальной моделью, потому что дает лучший event-level `F1` и меньше ложных событий. Но `X3D-S` даже после честной настройки threshold показала существенно лучший `Mean IoU` (`0.6658 -> 0.7321`) и примерно в 10 раз меньший размер checkpoint. Для задачи регистрации событий это сильный результат: `X3D-S` перспективна для дальнейшей работы над точностью временных границ и edge/mobile deployment.

## 39. Финальная интерпретация результата

Проект можно считать успешно собранным как полноценный Deep Learning pipeline:

- данные проверяются;
- split сделан корректно;
- модель обучена на полном доступном наборе;
- checkpoint сохранен;
- inference работает;
- события экспортируются;
- timeline строится;
- annotated video создается;
- Streamlit demo есть;
- счетчик повторений добавлен;
- test evaluation посчитан.

Качество:

```text
F1 на test split = 0.8421
```

Это хороший результат для учебного проекта на пользовательском датасете из 82 видео. При этом нужно честно говорить, что модель еще не production-grade: ей нужно больше разнообразных данных для надежной работы на любых новых видео.

## 40. Где лежат главные результаты

```text
configs/final.yaml
models/final_checkpoint.pt
models/final_metrics.json
models/temperature.json
models/training_curves.png
configs/x3d_s.yaml
models/x3d_s_checkpoint.pt
models/x3d_s_metrics.json
models/x3d_s_temperature.json
models/x3d_s_training_curves.png
outputs/final_dataset_validation.json
outputs/final/test_eval/summary_metrics.json
outputs/final/test_eval/per_video_metrics.csv
outputs/final/test_eval/confusion_matrix.png
outputs/x3d_s/test_eval/summary_metrics.json
outputs/x3d_s/test_eval_tuned/summary_metrics.json
outputs/x3d_s/threshold_sweep_val/threshold_sweep.csv
outputs/x3d_s/test_eval/confusion_matrix.png
docs/full_project_description.md
```

## 41. Краткий вывод

Итоговый проект - это самостоятельная система регистрации событий на видеозаписи, построенная на современном PyTorch video action recognition pipeline. В проекте объединены идеи 3D-CNN, sliding window inference, temporal smoothing, event registry, evaluation, visualization, Streamlit demo и repetition counting.

Главная ценность проекта - он показывает не только обучение модели, но и полный путь от сырых видео и CSV-разметки до готового журнала событий и демонстрационного видео с разметкой.
