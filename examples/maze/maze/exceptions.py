from maze.maze import Direction


class DirectionNonWalkable(Exception):
    def __init__(self, direction: Direction):
        super().__init__(f"The direction `{direction}` is not walkable.")
