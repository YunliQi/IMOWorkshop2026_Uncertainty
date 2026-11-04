"""Inference backends. Every backend turns an InferenceProblem into a Posterior."""
from .base import Backend, Posterior
from .abc_smc import ABCBackend
from .mcmc import MCMCBackend

BACKENDS = {"abc": ABCBackend, "mcmc": MCMCBackend}


def get_backend(name) -> Backend:
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(BACKENDS)}") from None
