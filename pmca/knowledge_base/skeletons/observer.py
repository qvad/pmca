from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def update(self, data):
        pass

class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Observer):
        self._observers.remove(observer)

    def notify(self, data):
        for observer in self._observers:
            observer.update(data)
