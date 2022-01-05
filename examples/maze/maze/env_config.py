from typing import TypedDict

from maze.maze import Map, DEFAULT_MAP


class MazeEnvConfig(TypedDict, total=False):
    map: Map


DEFAULTS: MazeEnvConfig = {"map": DEFAULT_MAP}
