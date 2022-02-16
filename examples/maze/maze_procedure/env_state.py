from dataclasses import dataclass

from maze.maze import Maze, Position


@dataclass
class MazeEnvState:
    maze: Maze
    position: Position
