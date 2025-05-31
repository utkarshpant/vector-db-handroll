import threading
import time
import pytest
from app.utils.read_write_lock import ReadWriteLock

def test_multiple_readers():
    lock = ReadWriteLock()
    read_count = 0
    errors = []

    def reader():
        nonlocal read_count
        try:
            lock.acquire_read()
            read_count += 1
            time.sleep(0.1)
            read_count -= 1
        finally:
            lock.release_read()

    threads = [threading.Thread(target=reader) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert read_count == 0


def test_writer_blocks_readers():
    lock = ReadWriteLock()
    events = []

    def writer():
        lock.acquire_write()
        events.append('writer_acquired')
        time.sleep(0.2)
        events.append('writer_releasing')
        lock.release_write()

    def reader():
        time.sleep(0.05)
        lock.acquire_read()
        events.append('reader_acquired')
        lock.release_read()

    t_writer = threading.Thread(target=writer)
    t_reader = threading.Thread(target=reader)
    t_writer.start()
    t_reader.start()
    t_writer.join()
    t_reader.join()
    # Reader should acquire after writer releases
    assert events == ['writer_acquired', 'writer_releasing', 'reader_acquired']


def test_readers_block_writer():
    lock = ReadWriteLock()
    events = []

    def reader():
        lock.acquire_read()
        events.append('reader_acquired')
        time.sleep(0.2)
        events.append('reader_releasing')
        lock.release_read()

    def writer():
        time.sleep(0.05)
        lock.acquire_write()
        events.append('writer_acquired')
        lock.release_write()

    t_reader = threading.Thread(target=reader)
    t_writer = threading.Thread(target=writer)
    t_reader.start()
    t_writer.start()
    t_reader.join()
    t_writer.join()
    # Writer should acquire after reader releases
    assert events == ['reader_acquired', 'reader_releasing', 'writer_acquired']
