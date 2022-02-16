from enum import Enum
from typing import List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

Value = int
WALL = 0
CORRIDOR = 1
START = 2
GOAL = 3
Map = List[List[Value]]

Position = Tuple[int, int]

DEFAULT_MAP = [
    [0, 0, 1, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 2],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
]


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @classmethod
    def opposite(cls, direction: "Direction") -> "Direction":
        return Direction((direction.value + 2) % len(cls))


class Maze:
    def __init__(self, map: Optional[Map] = None):
        if map is None:
            map = DEFAULT_MAP
        self._map = np.array(map, dtype=np.uint8)
        self._rows = len(map)
        self._cols = len(map[0])

    @property
    def map(self) -> npt.NDArray[np.float32]:
        return self._map

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def start(self) -> Position:
        rows, cols = np.where(self._map == START)
        assert (
            len(rows) == len(cols) == 1
        ), "There should be exactly one starting position."
        return rows[0], cols[0]

    @property
    def goal(self) -> Position:
        rows, cols = np.where(self._map == GOAL)
        assert len(rows) == len(cols) == 1, "The should be exactly one goal position."
        return rows[0], cols[0]

    def is_intersection(self, position: Position) -> bool:
        adjacent_tiles = self._adjacent_tiles(position)
        if self._is_boundary(position):
            return len(adjacent_tiles) > 1
        return len(adjacent_tiles) > 2

    def is_direction_walkable(self, position: Position, direction: Direction) -> bool:
        tile_position = self._next_position_at_direction(position, direction)
        return self._is_walkable(tile_position)

    def walkable_directions(self, position: Position) -> Set[Direction]:
        return {
            direction
            for direction in Direction
            if self.is_direction_walkable(position, direction)
        }

    def next_position(self, position: Position, direction: Direction) -> Position:
        tile_position = self._next_position_at_direction(position, direction)
        assert self._is_walkable(tile_position)
        return tile_position

    def _adjacent_tiles(self, position: Position) -> List[Position]:
        tiles = []
        for direction in Direction:
            next_position = self._next_position_at_direction(position, direction)
            if self._is_walkable(next_position):
                tiles.append(next_position)
        return tiles

    def _next_position_at_direction(
        self, position: Position, direction: Direction
    ) -> Position:
        x, y = position
        adjacent_indices = {
            Direction.UP: (x - 1, y),
            Direction.RIGHT: (x, y + 1),
            Direction.DOWN: (x + 1, y),
            Direction.LEFT: (x, y - 1),
        }
        return adjacent_indices[direction]

    def _is_walkable(self, position: Position) -> bool:
        try:
            value = int(self._map[position])
        except IndexError:
            return False
        return value in (CORRIDOR, START, GOAL)

    def _is_boundary(self, position: Position) -> bool:
        x, y = position
        return not (0 < x < self.rows - 1 and 0 < y < self.cols - 1)
