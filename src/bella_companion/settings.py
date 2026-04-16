from bella_companion.backend import ReLU, Sigmoid, Softplus, Tanh
from bella_companion.models import BELLASetting, WeightsPrior

BELLA_SETTINGS = {
    "BELLA-16": BELLASetting(
        weights_prior=WeightsPrior.NORMAL,
        hidden_activation=ReLU,
        nodes=[16],
    ),
    "BELLA-3_2": BELLASetting(
        weights_prior=WeightsPrior.NORMAL,
        hidden_activation=ReLU,
        nodes=[3, 2],
    ),
    "BELLA-16_8": BELLASetting(
        weights_prior=WeightsPrior.NORMAL,
        hidden_activation=ReLU,
        nodes=[16, 8],
    ),
    "BELLA-32_16": BELLASetting(
        weights_prior=WeightsPrior.NORMAL,
        hidden_activation=ReLU,
        nodes=[32, 16],
    ),
    "BELLA-16_8_4": BELLASetting(
        weights_prior=WeightsPrior.NORMAL,
        hidden_activation=ReLU,
        nodes=[16, 8, 4],
    ),
    "BELLA-Tanh": BELLASetting(
        weights_prior=WeightsPrior.NORMAL,
        hidden_activation=Tanh,
        nodes=[32, 16],
    ),
    "BELLA-Softplus": BELLASetting(
        weights_prior=WeightsPrior.NORMAL,
        hidden_activation=Softplus,
        nodes=[32, 16],
    ),
    "BELLA-Sigmoid": BELLASetting(
        weights_prior=WeightsPrior.NORMAL,
        hidden_activation=Sigmoid,
        nodes=[32, 16],
    ),
    "BELLA-Laplace": BELLASetting(
        weights_prior=WeightsPrior.LAPLACE,
        hidden_activation=ReLU,
        nodes=[32, 16],
    ),
}
