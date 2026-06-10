from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, replace


@dataclass
class Prediction:
    time_sec: float
    label: str
    confidence: float


@dataclass
class Event:
    label: str
    start_sec: float
    end_sec: float
    avg_confidence: float
    max_confidence: float


class EventRegistry:
    def __init__(
        self,
        confidence_threshold: float,
        smoothing_window: int,
        min_event_duration_sec: float,
        merge_gap_sec: float,
        unknown_label: str = "unknown",
    ):
        self.confidence_threshold = confidence_threshold
        self.min_event_duration_sec = min_event_duration_sec
        self.merge_gap_sec = merge_gap_sec
        self.unknown_label = unknown_label
        self.window: deque[Prediction] = deque(maxlen=smoothing_window)
        self.current_label: str | None = None
        self.current_start = 0.0
        self.current_confidences: list[float] = []
        self.events: list[Event] = []
        self.last_time = 0.0

    def add_prediction(self, prediction: Prediction) -> str:
        self.last_time = prediction.time_sec
        observed = prediction
        if observed.confidence < self.confidence_threshold:
            observed = Prediction(observed.time_sec, self.unknown_label, observed.confidence)
        self.window.append(observed)
        stable = self._stable_prediction(observed.time_sec)
        self._update_active_event(stable)
        return stable.label

    def close(self, end_time_sec: float | None = None) -> list[Event]:
        if self.current_label and self.current_label != self.unknown_label:
            self._finish_current(end_time_sec if end_time_sec is not None else self.last_time)
        self.current_label = None
        return self._merge_events(self.events)

    def _stable_prediction(self, time_sec: float) -> Prediction:
        labels = [item.label for item in self.window]
        label = Counter(labels).most_common(1)[0][0]
        confs = [item.confidence for item in self.window if item.label == label]
        return Prediction(time_sec=time_sec, label=label, confidence=sum(confs) / max(1, len(confs)))

    def _update_active_event(self, prediction: Prediction) -> None:
        if self.current_label is None:
            self.current_label = prediction.label
            self.current_start = prediction.time_sec
            self.current_confidences = [prediction.confidence]
            return

        if prediction.label == self.current_label:
            self.current_confidences.append(prediction.confidence)
            return

        if self.current_label != self.unknown_label:
            self._finish_current(prediction.time_sec)
        self.current_label = prediction.label
        self.current_start = prediction.time_sec
        self.current_confidences = [prediction.confidence]

    def _finish_current(self, end_sec: float) -> None:
        duration = end_sec - self.current_start
        if duration < self.min_event_duration_sec:
            return
        self.events.append(
            Event(
                label=str(self.current_label),
                start_sec=round(self.current_start, 3),
                end_sec=round(end_sec, 3),
                avg_confidence=round(sum(self.current_confidences) / len(self.current_confidences), 4),
                max_confidence=round(max(self.current_confidences), 4),
            )
        )

    def _merge_events(self, events: list[Event]) -> list[Event]:
        merged: list[Event] = []
        for event in events:
            if merged and merged[-1].label == event.label and event.start_sec - merged[-1].end_sec <= self.merge_gap_sec:
                prev = merged[-1]
                total_duration = (prev.end_sec - prev.start_sec) + (event.end_sec - event.start_sec)
                if total_duration > 0:
                    prev.avg_confidence = round(
                        (
                            prev.avg_confidence * (prev.end_sec - prev.start_sec)
                            + event.avg_confidence * (event.end_sec - event.start_sec)
                        )
                        / total_duration,
                        4,
                    )
                prev.end_sec = event.end_sec
                prev.max_confidence = max(prev.max_confidence, event.max_confidence)
            else:
                merged.append(event)
        return merged

    @staticmethod
    def to_dicts(events: list[Event]) -> list[dict[str, float | str]]:
        return [asdict(event) for event in events]


def _consecutive_times(predictions: list[Prediction], label: str, threshold: float) -> list[float]:
    times: list[float] = []
    streak: list[Prediction] = []
    for prediction in sorted(predictions, key=lambda item: item.time_sec):
        if prediction.label == label and prediction.confidence >= threshold:
            streak.append(prediction)
            if len(streak) >= 2:
                times.extend([item.time_sec for item in streak[-2:]])
        else:
            streak = []
    return times


def refine_event_boundaries(
    event: Event,
    raw_predictions: list[Prediction],
    confidence_threshold: float,
) -> Event:
    start_window = [
        prediction
        for prediction in raw_predictions
        if event.start_sec - 2.0 <= prediction.time_sec <= event.start_sec + 2.0
    ]
    end_window = [
        prediction
        for prediction in raw_predictions
        if event.end_sec - 2.0 <= prediction.time_sec <= event.end_sec + 2.0
    ]
    start_candidates = _consecutive_times(start_window, event.label, confidence_threshold)
    end_candidates = _consecutive_times(end_window, event.label, confidence_threshold)
    new_start = min(start_candidates) if start_candidates else event.start_sec
    new_end = max(end_candidates) if end_candidates else event.end_sec
    if new_start >= new_end:
        return event
    return replace(event, start_sec=round(new_start, 3), end_sec=round(new_end, 3))
