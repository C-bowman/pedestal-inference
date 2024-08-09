from abc import ABC, abstractmethod
from numpy import ndarray


class ProfileModel(ABC):
    name: str
    parameters: dict
    n_parameters: int

    radius: ndarray
    forward_prediction: callable
    forward_gradient: callable
    forward_jacobian: callable
    forward_prediction_and_jacobian: callable

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
    def prediction_and_jacobian(self, radius: ndarray, theta: ndarray) -> tuple[ndarray, ndarray]:
        pass

    @abstractmethod
    def update_radius(self, radius: ndarray):
        pass

    @abstractmethod
    def get_model_configuration(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_configuration(cls, config: dict):
        pass

    def copy(self):
        """
        Build and return a separate copy of the profile model with
        the same configuration.
        """
        return self.from_configuration(self.get_model_configuration())
