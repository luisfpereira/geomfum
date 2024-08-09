"""Base shape."""

import abc
import logging


class Shape(abc.ABC):
    def __init__(self):
        # TODO: create automated way for computing this?
        # TODO: should this be handled as e.g. laplacian.
        self.basis = None

        # TODO: empty np instead?
        # TODO: add function to add them
        self.landmarks = []

    def equip_with_operator(self, name, Operator, allow_overwrite=True, **kwargs):
        """Equip with operator."""
        name_exists = hasattr(self, name)
        if name_exists:
            if allow_overwrite:
                logging.warning(f"Overriding {name}.")
            else:
                raise ValueError(f"{name} already exists")

        operator = Operator(self, **kwargs)
        setattr(self, name, operator)

        return self
