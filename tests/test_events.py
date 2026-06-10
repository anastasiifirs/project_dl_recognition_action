from event_video_recognition.events import EventRegistry, Prediction


def test_registry_filters_short_low_confidence_noise():
    registry = EventRegistry(
        confidence_threshold=0.6,
        smoothing_window=3,
        min_event_duration_sec=0.5,
        merge_gap_sec=0.2,
    )
    registry.add_prediction(Prediction(0.0, "walk", 0.95))
    registry.add_prediction(Prediction(0.2, "jump", 0.40))
    registry.add_prediction(Prediction(0.4, "walk", 0.93))
    registry.add_prediction(Prediction(0.8, "walk", 0.92))

    events = registry.close(1.0)

    assert len(events) == 1
    assert events[0].label == "walk"
    assert events[0].start_sec == 0.0


def test_registry_merges_same_label_across_small_gap():
    registry = EventRegistry(
        confidence_threshold=0.6,
        smoothing_window=1,
        min_event_duration_sec=0.1,
        merge_gap_sec=0.5,
    )
    registry.add_prediction(Prediction(0.0, "squat", 0.9))
    registry.add_prediction(Prediction(1.0, "unknown", 0.2))
    registry.add_prediction(Prediction(1.2, "squat", 0.88))

    events = registry.close(2.0)

    assert len(events) == 1
    assert events[0].label == "squat"
    assert events[0].end_sec == 2.0


def test_registry_applies_confidence_threshold():
    registry = EventRegistry(
        confidence_threshold=0.8,
        smoothing_window=1,
        min_event_duration_sec=0.1,
        merge_gap_sec=0.2,
    )
    registry.add_prediction(Prediction(0.0, "run", 0.4))
    registry.add_prediction(Prediction(1.0, "run", 0.4))

    events = registry.close(2.0)

    assert events == []
