import threading

from hf_dm.camera.spinnaker import _PreviewDispatcher


def test_preview_dispatcher_keeps_latest_frame_when_callback_is_busy():
    received: list[int] = []
    callback_started = threading.Event()
    release_callback = threading.Event()

    def callback(frame: dict) -> None:
        received.append(int(frame["frame_index"]))
        callback_started.set()
        release_callback.wait(timeout=1.0)

    dispatcher = _PreviewDispatcher(callback, thread_name="test-preview-dispatcher")
    dispatcher.start()
    try:
        assert dispatcher.submit({"frame_index": 1})
        assert callback_started.wait(timeout=1.0)
        for frame_index in range(2, 10):
            assert dispatcher.submit({"frame_index": frame_index})
        release_callback.set()
    finally:
        dispatcher.stop()

    assert received[0] == 1
    assert received[-1] == 9
    assert dispatcher.submitted_count == 9
    assert dispatcher.delivered_count == len(received)
