import logging
import os

from torch.utils.tensorboard import SummaryWriter


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(logging.Logger, metaclass=Singleton):
    def __init__(self, name, log_dir, log_level=logging.DEBUG):
        super(Logger, self).__init__(name, log_level)
        self.writer = SummaryWriter(os.path.join(log_dir, "tb"))

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            "%Y-%m-%dT%T%Z",
        )
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.setLevel(log_level)
        self.addHandler(file_handler)
        self.addHandler(stream_handler)

    @classmethod
    def get(cls):
        return Singleton.__call__(cls)
