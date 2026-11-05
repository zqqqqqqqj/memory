"""Ordered CPU prefetch with checkpoints taken at the consumer's position."""
import copy
import queue
import threading


class PrefetchedInputs:
    """One producer preserves source RNG/order; queued samples are never skipped.

    ``prepare`` must be deterministic and CPU-only: random model augmentation
    and all CUDA model operations remain in the training thread.
    """
    def __init__(self, source, prepare, capacity=32):
        if capacity < 1:
            raise ValueError("Prefetch capacity must be positive")
        self.source, self.prepare = source, prepare
        self.queue = queue.Queue(maxsize=capacity)
        self.stopping = threading.Event()
        self.committed = copy.deepcopy(source.state_dict())
        self.thread = threading.Thread(target=self._produce, name="qrm-cpu-inputs", daemon=True)
        self.thread.start()

    def _put(self, value):
        while not self.stopping.is_set():
            try:
                self.queue.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def _produce(self):
        try:
            while not self.stopping.is_set():
                episode, timestep, raw, target = next(self.source)
                prepared = self.prepare(raw, target)
                # The source will advance independently after this snapshot.
                state = copy.deepcopy(self.source.state_dict())
                self._put(((episode, timestep, prepared), state, None))
        except BaseException as error:
            self._put((None, None, error))
        finally:
            self.source.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.stopping.is_set():
            raise StopIteration
        while True:
            try:
                item, state, error = self.queue.get(timeout=0.1)
                break
            except queue.Empty:
                if not self.thread.is_alive():
                    raise RuntimeError("CPU prefetch worker exited unexpectedly")
        if error is not None:
            raise error
        self.committed = state
        return item

    def state_dict(self):
        # Only samples delivered to training count, irrespective of lookahead.
        return copy.deepcopy(self.committed)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("CPU prefetch worker did not close within 30 seconds")
        while not self.queue.empty():
            self.queue.get_nowait()
