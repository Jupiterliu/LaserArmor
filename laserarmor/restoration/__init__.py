from .dominant_remove import StripeResDominantRemove
from .residual_remove import RowWiseAffineResidualModel


def build_restorer(*args, **kwargs):
    from .adapters import build_restorer as build

    return build(*args, **kwargs)


__all__ = ["RowWiseAffineResidualModel", "StripeResDominantRemove", "build_restorer"]
