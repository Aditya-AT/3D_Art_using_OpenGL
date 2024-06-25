import numpy as np

from vertex import Vertex


# Function to clip a line using Cohen-Sutherland algorithm
def clipLine(P0, P1, top, bottom, right, left):
    # Define region codes for each edge of the clip window
    INSIDE = 0  # 0000
    LEFT = 1  # 0001
    RIGHT = 2  # 0010
    BOTTOM = 4  # 0100
    TOP = 8  # 1000

    def computeCode(vertex):
        code = INSIDE

        if vertex.x < left:
            code |= LEFT
        elif vertex.x > right:
            code |= RIGHT
        if vertex.y < bottom:
            code |= BOTTOM
        elif vertex.y > top:
            code |= TOP

        return code

    codeP0 = computeCode(P0)
    codeP1 = computeCode(P1)

    clipped_points = []

    while True:
        if codeP0 == 0 and codeP1 == 0:
            # Both endpoints are inside the clip window
            clipped_points.append(P0)
            clipped_points.append(P1)
            break
        elif codeP0 & codeP1 != 0:
            # Both endpoints are outside the same edge, so the line is outside
            break
        else:
            # Calculate the intersection point
            code = codeP0 if codeP0 != 0 else codeP1

            if code & TOP:
                x = P0.x + (P1.x - P0.x) * (top - P0.y) / (P1.y - P0.y)
                y = top
            elif code & BOTTOM:
                x = P0.x + (P1.x - P0.x) * (bottom - P0.y) / (P1.y - P0.y)
                y = bottom
            elif code & RIGHT:
                y = P0.y + (P1.y - P0.y) * (right - P0.x) / (P1.x - P0.x)
                x = right
            elif code & LEFT:
                y = P0.y + (P1.y - P0.y) * (left - P0.x) / (P1.x - P0.x)
                x = left

            if code == codeP0:
                P0 = Vertex(x, y, P0.r, P0.g, P0.b)
                codeP0 = computeCode(P0)
            else:
                P1 = Vertex(x, y, P1.r, P1.g, P1.b)
                codeP1 = computeCode(P1)

    return np.array(clipped_points)


def clipPoly(vertices, top, bottom, right, left):
    if not vertices.any():
        return np.array([])  # Return an empty array if there are no vertices to clip.

    clipped_vertices = vertices

    # Clip against the top edge.
    clipped_vertices = clipEdge(clipped_vertices, top, 'top')

    # Clip against the bottom edge.
    clipped_vertices = clipEdge(clipped_vertices, bottom, 'bottom')

    # Clip against the right edge.
    clipped_vertices = clipEdge(clipped_vertices, right, 'right')

    # Clip against the left edge.
    clipped_vertices = clipEdge(clipped_vertices, left, 'left')

    return np.array(clipped_vertices)


def clipEdge(vertices, edge_value, edge_name):
    clipped_vertices = []
    num_vertices = len(vertices)

    if num_vertices == 0:
        return []

    prev_vertex = vertices[-1]

    for curr_vertex in vertices:
        if isInside(curr_vertex, edge_name, edge_value):
            if isInside(prev_vertex, edge_name, edge_value):
                # Both vertices are inside, add the current vertex to the result.
                clipped_vertices.append(curr_vertex)
            else:
                # Current vertex is inside, but previous vertex is outside.
                # Add the intersection point to the result and the current vertex.
                intersection = computeIntersection(prev_vertex, curr_vertex, edge_name, edge_value)
                clipped_vertices.append(intersection)
                clipped_vertices.append(curr_vertex)
        elif isInside(prev_vertex, edge_name, edge_value):
            # Previous vertex is inside, but current vertex is outside.
            # Add the intersection point to the result.
            intersection = computeIntersection(prev_vertex, curr_vertex, edge_name, edge_value)
            clipped_vertices.append(intersection)

        prev_vertex = curr_vertex

    return clipped_vertices


def isInside(vertex, edge_name, edge_value):
    if edge_name == 'top':
        return vertex.y <= edge_value
    elif edge_name == 'bottom':
        return vertex.y >= edge_value
    elif edge_name == 'right':
        return vertex.x <= edge_value
    elif edge_name == 'left':
        return vertex.x >= edge_value


def computeIntersection(p0, p1, edge_name, edge_value):
    u = 0.0  # Interpolation parameter
    if edge_name == 'top':
        u = (edge_value - p0.y) / (p1.y - p0.y)
    elif edge_name == 'bottom':
        u = (edge_value - p0.y) / (p1.y - p0.y)
    elif edge_name == 'right':
        u = (edge_value - p0.x) / (p1.x - p0.x)
    elif edge_name == 'left':
        u = (edge_value - p0.x) / (p1.x - p0.x)

    new_x = p0.x + u * (p1.x - p0.x)
    new_y = p0.y + u * (p1.y - p0.y)
    new_r = (1 - u) * p0.r + u * p1.r
    new_g = (1 - u) * p0.g + u * p1.g
    new_b = (1 - u) * p0.b + u * p1.b

    return Vertex(new_x, new_y, new_r, new_g, new_b)
