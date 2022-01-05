import pytest

from maze.maze import Maze, Position, Direction


@pytest.fixture
def maze() -> Maze:
    return Maze()


def test_maze_start_outputs_valid_starting_position(maze: Maze) -> None:
    assert maze.start == (4, 9)


def test_maze_goal_outputs_valid_goal_position(maze: Maze) -> None:
    assert maze.goal == (0, 4)


@pytest.mark.parametrize(
    "position",
    [
        (1, 2),
        (1, 4),
        (1, 8),
        (2, 2),
        (4, 2),
        (4, 5),
        (4, 8),
        (4, 9),
        (6, 1),
        (6, 4),
        (6, 8),
        (8, 1),
        (8, 4),
        (8, 8),
    ],
)
def test_maze_is_intersection_outputs_true_for_intersections(
    maze: Maze, position: Position
) -> None:
    assert maze.is_intersection(position)


@pytest.mark.parametrize(
    "position",
    [
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 6),
        (1, 9),
    ],
)
def test_maze_is_intersection_outputs_false_for_corridors(
    maze: Maze, position: Position
) -> None:
    assert not maze.is_intersection(position)


@pytest.mark.parametrize(
    "position",
    [
        (0, 0),
        (1, 1),
        (2, 7),
        (9, 9),
    ],
)
def test_maze_is_intersection_outputs_false_for_walls(
    maze: Maze, position: Position
) -> None:
    assert not maze.is_intersection(position)


def test_maze_is_direction_walkable(maze: Maze) -> None:
    assert maze.is_direction_walkable((1, 7), Direction.LEFT)
    assert maze.is_direction_walkable((1, 7), Direction.RIGHT)
    assert not maze.is_direction_walkable((1, 7), Direction.UP)
    assert not maze.is_direction_walkable((1, 7), Direction.DOWN)


def test_maze_walkable_directions(maze: Maze) -> None:
    assert maze.walkable_directions((1, 7)) == {Direction.LEFT, Direction.RIGHT}


def test_maze_next_tile(maze: Maze) -> None:
    assert maze.next_position((1, 3), Direction.LEFT) == (1, 2)
    assert maze.next_position((1, 3), Direction.RIGHT) == (1, 4)


def test_direction_opposite() -> None:
    assert Direction.opposite(Direction.LEFT) == Direction.RIGHT
    assert Direction.opposite(Direction.RIGHT) == Direction.LEFT
    assert Direction.opposite(Direction.UP) == Direction.DOWN
    assert Direction.opposite(Direction.DOWN) == Direction.UP
