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


def test_multiple_readers_concurrent():
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
        except Exception as e:
            errors.append(e)
        finally:
            lock.release_read()

    threads = [threading.Thread(target=reader) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert read_count == 0
    assert not errors


def test_writer_waits_for_readers():
    lock = ReadWriteLock()
    read_started = threading.Event()
    write_acquired = threading.Event()
    order = []

    def reader():
        lock.acquire_read()
        order.append('read')
        read_started.set()
        time.sleep(0.2)
        lock.release_read()

    def writer():
        read_started.wait()
        lock.acquire_write()
        order.append('write')
        write_acquired.set()
        lock.release_write()

    t1 = threading.Thread(target=reader)
    t2 = threading.Thread(target=writer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert order == ['read', 'write']
    assert write_acquired.is_set()


def test_only_one_writer_concurrent():
    lock = ReadWriteLock()
    write_count = 0
    errors = []

    def writer():
        nonlocal write_count
        try:
            lock.acquire_write()
            write_count += 1
            time.sleep(0.1)
        except Exception as e:
            errors.append(e)
        finally:
            lock.release_write()

    threads = [threading.Thread(target=writer) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert write_count == 3
    assert not errors


def test_writer_blocks_new_readers():
    lock = ReadWriteLock()
    order = []
    write_started = threading.Event()
    read_done = threading.Event()

    def writer():
        lock.acquire_write()
        order.append('write')
        write_started.set()
        time.sleep(0.2)
        lock.release_write()

    def reader():
        write_started.wait()
        lock.acquire_read()
        order.append('read')
        lock.release_read()
        read_done.set()

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert order == ['write', 'read']
    assert read_done.is_set()
