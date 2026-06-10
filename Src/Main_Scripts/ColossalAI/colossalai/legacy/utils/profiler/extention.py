# Copyright (c) 2025 MatN23. All rights reserved.
from abc import ABC, abstractmethod


class ProfilerExtension(ABC):
    @abstractmethod
    def prepare_trace(self):
        pass

    @abstractmethod
    def start_trace(self):
        pass

    @abstractmethod
    def stop_trace(self):
        pass

    @abstractmethod
    def extend_chrome_trace(self, trace: dict) -> dict:
        pass