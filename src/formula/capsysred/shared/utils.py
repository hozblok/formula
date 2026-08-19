"""Small structure helpers with no domain knowledge."""


def zeros(nx: int, ny: int):
    """ny×nx row-major grid of 0.0."""
    return [[0.0] * nx for _ in range(ny)]


def flat(grid):
    """Row-major grid -> flat list."""
    return [v for row in grid for v in row]
