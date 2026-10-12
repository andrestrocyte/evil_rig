"""Read timestamped optical edges from bpod-core's event buffer, without serial I/O."""

from __future__ import annotations

from typing import Callable
import threading
import time

import polars as pl


class BpodLickReader:
    """Consume each hardware event once, including final events after an FSM exits.

    peek_data is a thread-safe memory snapshot. Never call Input.read while
    the Bpod FSM reader owns its USB stream.
    """

    def __init__(self, bpod, port: int, callback: Callable[[dict], None]):
        self.bpod = bpod
        self.port = int(port)
        self.callback = callback
        # bpod-core labels electrical edges High/Low; MATLAB-style event
        # imports may instead use In/Out. A blocked optical beam is high.
        self._entry_events = {f"Port{self.port}_High", f"Port{self.port}In"}
        self._exit_events = {f"Port{self.port}_Low", f"Port{self.port}Out"}
        self._thread = None
        self._cursor = 0
        self.edge_count = 0
        self._base_us = int(bpod._time_reference.reset_system_time_ns) // 1000

    def poll(self) -> None:
        thread = getattr(self.bpod, "_event_thread", None)
        if thread is None:
            return
        frame = self.bpod.peek_data()
        # A new FSM may replace the buffer while the snapshot is being copied.
        if thread is not getattr(self.bpod, "_event_thread", None):
            return
        if thread is not self._thread:
            self._thread = thread
            self._cursor = 0
        end = frame.height
        pending = (
            frame.slice(self._cursor)
            .filter(pl.col("event").is_in(self._entry_events | self._exit_events))
            .with_columns(pl.col("time").dt.epoch("us").alias("epoch_us"))
        )
        for row in pending.iter_rows(named=True):
            timestamp_us = int(row["epoch_us"]) - self._base_us
            self.callback(
                {
                    "channel": self.port,
                    "pin": None,
                    "value": int(row["event"] in self._entry_events),
                    "event": row["event"],
                    "bpod_trial": int(row["trial"]),
                    "timestamp_us": timestamp_us,
                    "relative_time_s": timestamp_us / 1e6,
                    "bpod_epoch_us": int(row["epoch_us"]),
                    "source": "bpod_optical",
                    "clock": "bpod_session",
                }
            )
            self.edge_count += 1
        self._cursor = end


class BpodLickForwarder:
    """Pair one Bpod edge with its StreamDIO UART-receipt timestamp.

    Only one marker is outstanding. Missing or mismatched receipts abort the
    acquisition rather than silently substituting a host/Bpod timestamp.
    """

    def __init__(self, send_marker: Callable[[int, int], None], timeout_s: float = 0.15):
        self.send_marker = send_marker
        self.timeout_s = timeout_s
        self._condition = threading.Condition()
        self._pending: dict | None = None
        self._receipt: dict | None = None
        self._error: str | None = None

    def accept_receipt(
        self,
        channel: int,
        value: int,
        timestamp_us: int,
        relative_time_s: float,
        host_mono_s: float,
    ) -> None:
        with self._condition:
            if self._pending is None or channel != self._pending["channel"]:
                return
            if value != self._pending["value"]:
                self._error = "StreamDIO optical receipt does not match the Bpod edge"
            elif self._receipt is not None:
                self._error = "Duplicate StreamDIO optical receipt"
            else:
                self._receipt = {
                    "timestamp_us": timestamp_us,
                    "relative_time_s": relative_time_s,
                    "receipt_host_mono_s": host_mono_s,
                }
            self._condition.notify_all()

    def forward(self, event: dict) -> dict:
        sent_at = time.perf_counter()
        with self._condition:
            if self._pending is not None:
                raise RuntimeError("An optical marker is already outstanding")
            self._pending = dict(event)
            self._receipt = None
            self._error = None
        try:
            # No input reads or acknowledgements on Bpod's active FSM USB stream.
            self.send_marker(int(event["channel"]), int(event["value"]))
            deadline = sent_at + self.timeout_s
            with self._condition:
                while self._receipt is None and self._error is None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise TimeoutError("No StreamDIO timestamp received for optical lick")
                    self._condition.wait(remaining)
                if self._error is not None:
                    raise RuntimeError(self._error)
                assert self._receipt is not None
                return {
                    **event,
                    "bpod_timestamp_us": event["timestamp_us"],
                    "bpod_relative_time_s": event["relative_time_s"],
                    **self._receipt,
                    "clock": "streamdio",
                    "timestamp_origin": "streamdio_uart_receive",
                    "relay_send_host_mono_s": sent_at,
                    "relay_receipt_wait_s": self._receipt["receipt_host_mono_s"] - sent_at,
                    "relay_transport": "bpod_usb_to_module_uart",
                }
        finally:
            with self._condition:
                self._pending = None
