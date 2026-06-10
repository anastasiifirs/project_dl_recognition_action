from event_video_recognition.events import Event
from event_video_recognition.metrics import event_level_metrics, segment_iou


def test_segment_iou():
    assert segment_iou(0, 10, 5, 15) == 5 / 15
    assert segment_iou(0, 1, 2, 3) == 0


def test_event_level_metrics_counts_matches():
    gt = [Event("walk", 0, 5, 1, 1), Event("run", 6, 10, 1, 1)]
    pred = [Event("walk", 0.5, 5.5, 0.8, 0.9), Event("jump", 6, 10, 0.7, 0.8)]

    metrics = event_level_metrics(gt, pred, iou_threshold=0.3)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
