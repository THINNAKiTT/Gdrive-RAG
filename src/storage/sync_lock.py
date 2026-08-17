import os
import time

class SyncLock:
    def __init__(self, lock_path: str = "./.sync.lock", stale_after_seconds: float = 120.0):
        self.lock_path = lock_path
        self.stale_after_seconds = stale_after_seconds

    def acquire(self):
        with open(self.lock_path, "w") as f:
            f.write(str(os.getpid()))

    def refresh(self):
        try:
            os.utime(self.lock_path, None)
        except FileNotFoundError:
            pass

    def release(self):
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass

    def is_locked(self) -> bool:
        try:
            age = time.time() - os.path.getmtime(self.lock_path)
        except FileNotFoundError:
            return False
        if age > self.stale_after_seconds:
            # Stale lock (daemon likely crashed mid-sync) -- don't let
            # it block queries indefinitely.
            self.release()
            return False
        return True

    def wait_until_free(self, timeout_seconds: float = 30.0, poll_interval: float = 0.2):
        waited = 0.0
        while self.is_locked():
            if waited >= timeout_seconds:
                return False
            time.sleep(poll_interval)
            waited += poll_interval
        return True
