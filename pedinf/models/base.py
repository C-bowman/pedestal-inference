from abc import ABC, abstractmethod
from typing import Tuple
from numpy import ndarray


class ProfileModel(ABC):
    name: str
    parameters: dict
    n_parameters: int

    @staticmethod
    @abstractmethod
    def prediction(R: ndarray, theta: ndarray) -> ndarray:
        pass

    @staticmethod
    @abstractmethod
    def gradient(R: ndarray, theta: ndarray) -> ndarray:
        pass

    @staticmethod
    @abstractmethod
    def jacobian(R: ndarray, theta: ndarray) -> ndarray:
        pass

    @staticmethod
    @abstractmethod
    def prediction_and_jacobian(R: ndarray, theta: ndarray) -> Tuple[ndarray, ndarray]:
        pass
