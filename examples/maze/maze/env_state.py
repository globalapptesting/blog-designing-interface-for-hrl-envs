from dataclasses import dataclass

from maze.maze import Maze, Position, Direction


@dataclass
class MazeEnvState:
    maze: Maze
    position: Position
    direction: Direction
