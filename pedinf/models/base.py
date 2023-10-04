from abc import ABC, abstractmethod
from typing import Tuple
from numpy import ndarray


class ProfileModel(ABC):
    name: str
    parameters: dict
    n_parameters: int

    @abstractmethod
    def prediction(self, radius: ndarray, theta: ndarray) -> ndarray:
        pass

    @abstractmethod
    def gradient(self, radius: ndarray, theta: ndarray) -> ndarray:
        pass

    @abstractmethod
    def jacobian(self, radius: ndarray, theta: ndarray) -> ndarray:
        pass

    @abstractmethod
    def prediction_and_jacobian(self, radius: ndarray, theta: ndarray) -> Tuple[ndarray, ndarray]:
        pass
