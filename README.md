# Регистрация событий на видеозаписи

Deep Learning проект для задачи `video -> events`: система анализирует длинное видео с одним человеком и возвращает журнал действий с временными границами.

Пример события:

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

Итоговый pipeline:

- проверяет видео и CSV-разметку;
- делает train/val/test split по исходным видео;
- обучает современную 3D-CNN модель;
- запускает offline inference по sliding window;
- регистрирует события через сглаживание и пороги уверенности;
- сохраняет `events.json`, `events.csv`, `timeline.png`, `annotated.mp4`;
- считает event-level метрики;
- считает per-class IoU и строит confusion matrix;
- калибрует confidence через temperature scaling;
- уточняет границы событий через boundary refinement;
- добавляет счетчик повторений для `push_ups`, `squat`, `bend`, `jump`;
- запускает Streamlit demo для показа результата.

## Данные

Скинули в личные сообщения датасет)

Финальный локальный набор данных лежит в:

```text
data/dev/
  raw/                 # 82 исходных видео, не включены в GitHub из-за размера
  annotations.csv      # 716 размеченных событий
  splits/
    train.txt
    val.txt
    test.txt
  processed/
    train.csv
    val.csv
    test.csv
```

Формат `annotations.csv`:

```csv
video,start_sec,end_sec,label
IMG_3609.mov,0,5,stand
IMG_3609.mov,5,10,walk
IMG_3609.mov,10,15,run
```

Классы:

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

Поддерживаются `.mp4`, `.mov`, `.avi`, `.mkv`. Звук не используется: модель работает только с визуальными кадрами.

Важно для GitHub-версии: сырые видео `data/dev/raw/` не включены в репозиторий. Для полного воспроизведения обучения или оценки их нужно передать отдельно и положить в `data/dev/raw/`.

## Разделение Данных

Split сделан по исходным видео, а не случайно по событиям. Это важно, чтобы одинаковые фрагменты одного ролика не попадали одновременно в train и test.

Текущий split:

```text
train: 57 видео, 489 событий
val:   12 видео, 107 событий
test:  13 видео, 120 событий
```

## Модель

Основная модель:

```text
torchvision.models.video.r3d_18
```

Используется pretrained video backbone, финальный слой заменен под 8 классов проекта. Вход модели:

```text
[B, C, T, H, W]
```

Финальная конфигурация:

```text
configs/final.yaml
```

Финальный checkpoint:

```text
models/final_checkpoint.pt
```

Важно для GitHub-версии: `final_checkpoint.pt` весит больше 100 MB, поэтому добавлен через Git LFS. После клонирования репозитория нужно убедиться, что установлен Git LFS, иначе вместо настоящего checkpoint может скачаться только маленький pointer-файл.

История обучения:

```text
models/final_metrics.json
```

Калибровка confidence:

```text
models/temperature.json
```

## Обучение

Применена staged fine-tuning стратегия:

1. `head_warmup`: обучается новая классификационная голова.
2. `last_block_finetune`: размораживается последний 3D-блок `layer4` и голова.

Дополнительно используются:

- pretrained backbone;
- split по видео;
- `WeightedRandomSampler`;
- class weights;
- label smoothing;
- augmentation;
- cosine learning rate schedule;
- сохранение лучшего checkpoint по `val_acc`, при равенстве по меньшему `val_loss`;
- поддержка `cpu`, `cuda`, `mps`.

Команда финального обучения:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/final.yaml \
  --output-dir models \
  --device mps
```

Лучший результат на validation:

```text
best_val_acc  = 0.9813
best_val_loss = 0.3415
```

Важно: validation-результат высокий, но он может быть оптимистичным, потому что видео сняты в близких условиях. Главная честная проверка - test split.

## Test Метрики

Оценка на held-out test split из 13 видео:

```text
precision = 0.8189
recall    = 0.8667
f1        = 0.8421
mean_iou  = 0.6658
latency   = 1.5105 sec
```

По классам:

```text
push_ups: F1 1.0000
squat:   F1 0.9655
run:     F1 0.8824
jump:    F1 0.8667
other:   F1 0.8571
stand:   F1 0.7778
walk:    F1 0.7692
bend:    F1 0.6923
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

Дополнительные финальные артефакты:

```text
models/training_curves.png
models/temperature.json
outputs/final/test_eval/confusion_matrix.png
```

## Дополнительные Улучшения

После финального обучения добавлены инженерные улучшения:

- `boundary_refinement`: уточнение `start_sec` и `end_sec` по raw predictions около границ события;
- `temperature_scaling`: калибровка confidence модели по validation split;
- `per_class_iou`: отдельный mean IoU для каждого класса;
- `confusion_matrix.png`: event-level confusion matrix по matched events;
- `training_curves.png`: график train/val loss и accuracy по эпохам;
- `load_ensemble`: инфраструктура для нескольких checkpoints без поломки одиночного режима;
- safe pose fallback: если `models/pose_landmarker_full.task` отсутствует, inference не падает, а repetition counting получает `method="unavailable"`.

## Запуск Pipeline

Проверить датасет:

```bash
PYTHONPATH=src python scripts/validate_dataset.py \
  --config configs/final.yaml \
  --output outputs/final_dataset_validation.json
```

Подготовить split:

```bash
PYTHONPATH=src python scripts/prepare_splits.py \
  --config configs/final.yaml \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

Оценить test split:

```bash
PYTHONPATH=src python scripts/evaluate_split.py \
  --config configs/final.yaml \
  --checkpoint models/final_checkpoint.pt \
  --split test \
  --output-dir outputs/final/test_eval \
  --device mps
```

Inference на одном видео:

```bash
PYTHONPATH=src python scripts/infer_video.py \
  --config configs/final.yaml \
  --video data/dev/raw/IMG_3609.mov \
  --checkpoint models/final_checkpoint.pt \
  --output-dir outputs/final/IMG_3609 \
  --device mps
```

Streamlit demo:

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

## Outputs

После inference создаются:

```text
events.json
events.csv
timeline.png
annotated.mp4
inference_stats.json
```

После evaluation:

```text
metrics.json
summary_metrics.json
per_video_metrics.csv
confusion_matrix.png
```


## Ограничения

Текущая версия рассчитана на single-person видео. Multi-person tracking оставлен как future work.

Модель стала намного сильнее после расширения до 82 видео, но для более надежного переноса на полностью новые условия желательно добавить еще больше разнообразия: другие люди, фоны, ракурсы, одежду, освещение и порядок действий.
