#!/bin/env python3

from typing import Any
import click
import networkx as nx
import numpy as np
import stl
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


CIRCLE_PERIMETER_NUM_POINTS = 20
CIRCLE_PERIMETER_DELTA_RADIANS = 2 * np.pi / CIRCLE_PERIMETER_NUM_POINTS


def rotate_z(pos: np.array, rads: float) -> np.array:
    x, y, z = pos
    return np.array(
        [x * np.cos(rads) - y * np.sin(rads), y * np.cos(rads) + x * np.sin(rads), z]
    )


def get_unit_circle():
    return np.array(
        [
            rotate_z([1, 0, 0], CIRCLE_PERIMETER_DELTA_RADIANS * i)
            for i in range(CIRCLE_PERIMETER_NUM_POINTS)
        ]
    )


UNIT_CIRCLE = get_unit_circle()


def iter_looped_pairs(x: list[Any]) -> tuple[Any, Any]:
    for a, b in zip(x, x[1:] + [x[0]]):
        yield a, b


class PendingMesh:
    def __init__(self):
        self._vertices = []
        self._triangles = []

    def add_vertex(self, pos: np.array) -> int:
        """Returns the index of the position after being added."""
        assert len(pos) == 3, f"{pos}"
        self._vertices.append(pos)
        return len(self._vertices) - 1

    def add_verticies(self, poss: np.array) -> list[int]:
        return [self.add_vertex(poss[i, :]) for i in range(len(poss))]

    def add_triangle(self, idx_a: int, idx_b: int, idx_c: int):
        """Adds a triangle of three points to the mesh."""
        assert 0 <= idx_a < len(self._vertices)
        assert 0 <= idx_b < len(self._vertices)
        assert 0 <= idx_c < len(self._vertices)
        self._triangles.append(np.array([idx_a, idx_b, idx_c]))

    def add_rectangle(self, idx_a, idx_b, idx_c, idx_d):
        """Assumes indices are in an order around the parimeter."""
        self.add_triangle(idx_a, idx_b, idx_c)
        self.add_triangle(idx_c, idx_d, idx_a)

    def add_box(self, a, b, c, d, e, f, g, h):
        self.add_rectangle(a, b, c, d)
        self.add_rectangle(e, f, g, h)
        self.add_rectangle(a, b, f, e)
        self.add_rectangle(b, c, g, f)
        self.add_rectangle(c, d, h, g)
        self.add_rectangle(d, a, e, h)

    def add_cylinder(self, lower_center: np.array, radius: float, height: float):
        upper_center = lower_center + np.array([0, 0, height])
        upper_center_idx = self.add_vertex(upper_center)
        lower_center_idx = self.add_vertex(lower_center)
        lower_circle = UNIT_CIRCLE * (radius / 2) + lower_center
        lower_circle_indices = self.add_verticies(lower_circle)
        upper_circle = UNIT_CIRCLE * (radius / 2) + upper_center
        upper_circle_indices = self.add_verticies(upper_circle)

        lowers = iter_looped_pairs(lower_circle_indices)
        uppers = iter_looped_pairs(upper_circle_indices)
        for (lower_idx_a, lower_idx_b), (upper_idx_a, upper_idx_b) in zip(
            lowers, uppers
        ):
            self.add_triangle(lower_idx_a, lower_idx_b, lower_center_idx)
            self.add_triangle(upper_idx_a, upper_idx_b, upper_center_idx)
            self.add_rectangle(lower_idx_a, lower_idx_b, upper_idx_b, upper_idx_a)

    def add_line(
        self, center_a: np.array, center_b: np.array, width: float, height: float
    ):
        delta = center_b - center_a
        normal = rotate_z(delta, np.pi / 2)
        xy_offset = normal / np.linalg.norm(normal) * width / 2
        z_offset = np.array([0, 0, height])

        lower_a = center_a + xy_offset
        lower_b = center_a - xy_offset
        lower_c = center_b - xy_offset
        lower_d = center_b + xy_offset

        upper_a = lower_a + z_offset
        upper_b = lower_b + z_offset
        upper_c = lower_c + z_offset
        upper_d = lower_d + z_offset

        self.add_box(
            self.add_vertex(lower_a),
            self.add_vertex(lower_b),
            self.add_vertex(lower_c),
            self.add_vertex(lower_d),
            self.add_vertex(upper_a),
            self.add_vertex(upper_b),
            self.add_vertex(upper_c),
            self.add_vertex(upper_d),
        )

    def finish(self) -> stl.mesh.Mesh:
        mesh = stl.mesh.Mesh(np.zeros(len(self._triangles), dtype=stl.mesh.Mesh))
        for face_idx, face in enumerate(self._triangles):
            for i, vertex_idx in enumerate(face):
                mesh.vectors[face_idx][i] = self._vertices[vertex_idx]
        return mesh


def render_mesh(mesh: stl.mesh.Mesh):
    # Create a new plot
    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")

    # Load the STL files and add the vectors to the plot
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))

    # Auto scale to the mesh size
    scale = mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    plt.show()


@click.command()
@click.option("--edge-list", type=click.Path(exists=True))
@click.option("--edge-list-comment-char", type=click.STRING)
@click.option("--mesh", type=click.Path(exists=False))
@click.option("--node-width", type=click.FLOAT, default=0.15)
@click.option("--node-height", type=click.FLOAT, default=0.05)
@click.option("--edge-width", type=click.FLOAT, default=0.02)
@click.option("--render", type=click.BOOL)
def main(
    edge_list: click.Path,
    edge_list_comment_char: str,
    mesh: click.Path,
    node_width: float,
    node_height: float,
    edge_width: float,
    render: bool,
):
    pending_mesh = PendingMesh()

    graph = nx.read_edgelist(edge_list, comments=edge_list_comment_char)

    node_positions = nx.kamada_kawai_layout(graph)
    # Add z=0 to all pos.
    for i in node_positions:
        x, y = node_positions[i]
        node_positions[i] = np.array([x, y, 0])

    for node_center in node_positions.values():
        pending_mesh.add_cylinder(node_center, node_width / 2, node_height)

    for node_a, node_b in graph.edges():
        node_center_a = node_positions[node_a]
        node_center_b = node_positions[node_b]
        pending_mesh.add_line(node_center_a, node_center_b, edge_width, node_height)

    finished_mesh = pending_mesh.finish()
    finished_mesh.save(mesh)

    if render:
        render_mesh(finished_mesh)


if __name__ == "__main__":
    main()
