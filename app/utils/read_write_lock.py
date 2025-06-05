import threading
from contextlib import contextmanager

class ReadWriteLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._writer_waiting = False

    def acquire_read(self):
        with self._read_ready:
            # if a writer is active or waiting, let it go first
            while self._writer or self._writer_waiting:
                self._read_ready.wait()
            # increment number of readers
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                # if no readers are left, notify anyone waiting for a write lock
                self._read_ready.notify_all()

    def acquire_write(self):
        with self._read_ready:
            self._writer_waiting = True
            while self._readers > 0 or self._writer:
                self._read_ready.wait()
            self._writer_waiting = False
            self._writer = True

    def release_write(self):
        with self._read_ready:
            self._writer = False
            self._read_ready.notify_all()  # Wake up both readers and writers

    @contextmanager
    def read_lock(self):
        """Context manager for read operations. Usage: with lock.read_lock(): ..."""
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(self):
        """Context manager for write operations. Usage: with lock.write_lock(): ..."""
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()
