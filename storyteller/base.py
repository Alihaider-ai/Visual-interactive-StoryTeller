from abc import ABC, abstractmethod

class StorytellerBase(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs):
        pass