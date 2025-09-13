from typing import Callable

import pytest


class Helpers:
    @staticmethod
    def get_embedding_function() -> Callable:
        def _func(input):
            assert input == ["What are the requirements for the task?"]
            with open("tests/data/embedding.txt", "r") as file:
                content = file.read()
                numbers = content.split(",")
                return [[float(number) for number in numbers]]

        return _func


@pytest.fixture
def helpers():
    return Helpers
