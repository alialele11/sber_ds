import pytest
import random
import math


def my_sqrt(x: int) -> int:
    if x < 0:
        raise ValueError("Negative number")
    if type(x) is not int:
        raise TypeError("Not integer number")
    div = 1
    result = 0
    while x > 0:
        x -= div
        div += 2
        if x < 0:
            result += 0
        else:
            result += 1
    return result


def test_my_sqrt_negative_number():
    with pytest.raises(ValueError):
        my_sqrt(-2)


def test_my_sqrt_zero():
    assert my_sqrt(0) == 0


def test_my_sqrt_one():
    assert my_sqrt(1) == 1


def test_my_sqrt_big_number():
    assert my_sqrt(1524155677489) == 1234567


def test_my_sqrt_norm_number():
    assert my_sqrt(225) == 15


def test_my_sqrt_double():
    with pytest.raises(TypeError):
        my_sqrt(169.08)


def test_my_sqrt_on_random_numbers():
    random.seed(42)
    n = random.randint(1, pow(2, 10))
    eps = 1e-7
    for i in range(n):
        item = random.randint(1, pow(2, 31))
        assert abs(int(math.sqrt(item)) - my_sqrt(item)) < eps
