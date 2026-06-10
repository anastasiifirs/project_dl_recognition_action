from event_video_recognition.visualization import plot_timeline


def test_empty_timeline_creation(tmp_path):
    output = plot_timeline([], tmp_path / "timeline.png")

    assert output.exists()
    assert output.stat().st_size > 0
