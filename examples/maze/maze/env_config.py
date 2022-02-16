from typing import TypedDict

from maze.maze import DEFAULT_MAP, Map


class MazeEnvConfig(TypedDict, total=False):
    map: Map


DEFAULTS: MazeEnvConfig = {"map": DEFAULT_MAP}
