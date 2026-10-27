"""External FIFO consumer for Kotekan lifecycle tests; not a pipeline stage."""

from __future__ import annotations

import json
import os
import select
from pathlib import Path
import threading
import time
import urllib.request


class FifoBackpressure:
    """Hold an existing rawFileWrite consumer while observing existing /buffers.

    One output product uses FIFOs instead of regular files. A reader opens the
    first FIFO but deliberately does not drain it until the test observes full
    rings. No Kotekan stage, ownership rule, or scheduler is modified.
    """

    def __init__(self, output_dir: Path, received_dir: Path, product: str,
                 shard: int, frames: int, record_bytes: int):
        self.output_dir, self.received_dir = output_dir, received_dir
        self.product, self.shard = product, shard
        self.frames, self.record_bytes = frames, record_bytes
        self.prefix = f"{product}.shard{shard}"
        self.release = threading.Event()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._read, name="fifo-test-reader", daemon=True)
        self.error = None
        self.records = []
        received_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(frames):
            os.mkfifo(self.path(output_dir, frame))

    def path(self, directory, frame):
        return directory / f"{self.prefix}_{frame:07d}.bin"

    def start(self):
        self.thread.start()
        if not self.ready.wait(5):
            raise RuntimeError("FIFO reader did not open its first pipe")

    def _read(self):
        try:
            for frame in range(self.frames):
                fd = os.open(self.path(self.output_dir, frame), os.O_RDONLY | os.O_NONBLOCK)
                try:
                    if frame == 0:
                        self.ready.set()
                        while not self.release.wait(0.05):
                            if self.stop.is_set():
                                return
                    else:
                        self.stop.wait(0.01)  # Keep draining slower than an ordinary file sink.
                    data = bytearray()
                    deadline = time.monotonic() + 120
                    while not self.stop.is_set():
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"timed out receiving {self.prefix} frame {frame}")
                        try:
                            chunk = os.read(fd, 65536)
                        except BlockingIOError:
                            select.select([fd], [], [], 0.05)
                            continue
                        if chunk:
                            data.extend(chunk)
                        elif data:  # EOF after the actual writer closed this record.
                            break
                        else:  # No writer has opened this future frame yet.
                            self.stop.wait(0.001)
                    if self.stop.is_set():
                        return
                    if len(data) != self.record_bytes:
                        raise AssertionError(f"FIFO record {frame}: {len(data)} != {self.record_bytes}")
                    self.path(self.received_dir, frame).write_bytes(data)
                    self.records.append({"frame": frame, "bytes": len(data)})
                finally:
                    os.close(fd)
        except Exception as error:
            self.error = error
            self.ready.set()

    def observe_stall(self, process, port, config, other_record_sizes):
        """Require full output/input rings and completion of the unblocked shard.

        Two stable snapshots while the reader is held establish an actual stall;
        elapsed sleeps or configured delays alone are not backpressure evidence.
        /buffers is diagnostic and not an atomic cross-buffer snapshot.
        """
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        target = f"{self.product}_{self.shard}"
        names = (target, f"voltage_{self.shard}", f"missing_{self.shard}")
        other = 1 - self.shard
        deadline = time.monotonic() + 90
        first = None
        samples = []
        while time.monotonic() < deadline:
            if self.error:
                raise self.error
            if process.poll() is not None:
                raise AssertionError("Kotekan exited before the backpressure observation")
            try:
                with opener.open(f"http://127.0.0.1:{port}/buffers", timeout=1) as response:
                    snapshot = json.load(response)
            except (OSError, ValueError):
                time.sleep(0.05)
                continue
            full = all(snapshot[n]["num_full_frame"] == config[n]["num_frames"] for n in names)
            other_done = all(
                (self.output_dir / f"{product}.shard{other}_{self.frames - 1:07d}.bin").exists()
                and (self.output_dir / f"{product}.shard{other}_{self.frames - 1:07d}.bin").stat().st_size == size
                for product, size in other_record_sizes.items())
            if full and other_done:
                compact = {n: snapshot[n] for n in names}
                now = time.monotonic()
                if first is None:
                    first = (now, compact)
                    samples.append({"time": now, "buffers": compact})
                elif compact == first[1] and now - first[0] >= 0.25:
                    samples.append({"time": now, "buffers": compact})
                    return {"held_product": self.prefix, "stable_full_seconds": now - first[0],
                            "other_shard_completed_while_held": True, "snapshots": samples}
                elif compact != first[1]:
                    first = (now, compact)
                    samples = [{"time": now, "buffers": compact}]
            time.sleep(0.05)
        raise TimeoutError("did not observe full output/upstream rings and independent shard completion")

    def finish(self):
        self.thread.join(10)
        if self.thread.is_alive():
            raise TimeoutError("FIFO reader failed to finish after Kotekan exited")
        if self.error:
            raise self.error
        if len(self.records) != self.frames:
            raise AssertionError("FIFO receiver lost records")

    def close(self):
        self.release.set()
        self.stop.set()
        self.thread.join(2)
