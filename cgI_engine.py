import math

import glm

from vertex import *

import numpy as np


class CGIengine:
    def __init__(self, myWindow, defaction):
        self.w_width = myWindow.width
        self.w_height = myWindow.height
        self.win = myWindow
        self.keypressed = 1
        self.default_action = defaction

        # Initialize transformation matrices as 4x4 identity matrices
        self.model_transform = np.identity(4)
        self.normalization_transform = np.identity(4)
        self.camera_transform = np.identity(4)
        self.default_normal_transform = np.matrix([
            [
                2 / (self.w_width - 1), 0, 0, -1
            ],
            [
                0, 2 / (self.w_height - 1), 0, -1
            ],
            [
                0, 0, -2 / self.w_height, -1
            ],
            [
                0, 0, 0, 1
            ]
        ])
        self.default_view_transform = np.linalg.inv(self.default_normal_transform)
        self.projection_transform = np.identity(4)
        self.stack = [np.identity(4)]
        # Z-buffer initialization
        self.zbuffer = np.full((self.w_width, self.w_height), np.inf)
        self.ambient_color = None
        self.light_color = None
        self.light_position = None
        self.eye = None

    def pushTransform(self):
        if self.stack:
            self.stack.append(self.top() @ self.model_transform)
            self.model_transform = np.identity(4)
        else:
            self.stack.append(self.model_transform)

    def popTransform(self):
        if self.stack:
            self.stack.pop()

    def top(self):
        return self.stack[-1]

    def setAmbient(self, color):
        self.ambient_color = np.array(color)

    def setLight(self, position, color):
        self.light_position = np.array(position)
        self.light_color = np.array(color)


    # Helper methods for Phong shading calculations
    def calculateAmbient(self, color):
        return self.ambient_color * color

    def calculateDiffuse(self, vertex_pos, color, normal):
        vertex_pos = vertex_pos / np.linalg.norm(vertex_pos)
        light_dir = self.light_position - vertex_pos
        light_dir = light_dir / np.linalg.norm(light_dir)
        # Calculate the diffuse component
        return max(np.dot(normal, light_dir), 0) * self.light_color * color

    def calculateSpecular(self, vertex_pos, normal, scolor, exponent):
        normal = normal / np.linalg.norm(normal)
        light_dir = self.light_position - vertex_pos
        light_dir = light_dir / np.linalg.norm(light_dir)
        # View direction vector (assuming the viewer is at the origin)
        view_dir = self.eye - vertex_pos
        view_dir = view_dir / np.linalg.norm(view_dir)

        # Calculate the reflection vector
        reflect_dir = ((2 * np.dot(normal, light_dir)) * normal) - light_dir

        reflect_dir = reflect_dir / np.linalg.norm(reflect_dir)
        # Calculate the specular component
        specular_strength = np.power(max(np.dot(reflect_dir, view_dir), 0), exponent)
        return self.light_color * scolor * specular_strength

    def transformNormals(self, normal):
        # Calculate the normal transformation matrix (transpose of the inverse of the model-view matrix)
        normal_matrix = np.linalg.inv(self.stack[-1] @ self.camera_transform).T
        return normal_matrix @ np.array([normal, 0])

    def setOrtho(self, left, right, bottom, top, near, far):
        ortho = np.array([
            [2 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
            [0, 0, -2 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1]
        ])
        self.projection_transform = ortho

    def setPerspective(self, l, r, b, t, n, f):
        # Create the perspective projection matrix based on the frustum bounds
        M = np.zeros((4, 4))
        M[0, 0] = 2 * n / (r - l)
        M[0, 2] = (r + l) / (r - l)
        M[1, 1] = 2 * n / (t - b)
        M[1, 2] = (t + b) / (t - b)
        M[2, 2] = -(f + n) / (f - n)
        M[2, 3] = -2 * f * n / (f - n)
        M[3, 2] = -1

        self.projection_transform = M

    def setCamera(self, eye, lookat, up):
        self.eye = eye
        eye = np.array(eye)
        lookat = np.array(lookat)
        up = np.array(up)

        n = eye - lookat
        n = n / np.linalg.norm(n)

        u = np.cross(up, n)
        u = u / np.linalg.norm(u)

        v = np.cross(n, u)
        v = v / np.linalg.norm(v)

        # Creating the camera matrix
        self.camera_transform = np.array([
            [u[0], u[1], u[2], -np.dot(u, eye)],
            [v[0], v[1], v[2], -np.dot(v, eye)],
            [-n[0], -n[1], -n[2], np.dot(n, eye)],
            [0, 0, 0, 1]
        ])

    # draw a line from (x0, y0) to (x1, y1) in (r,g,b)
    def rasterizeLine(self, x0, y0, x1, y1, r, g, b):

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while x0 != x1 or y0 != y1:

            self.win.set_pixel(x0, y0, *(r, g, b))
            err2 = 2 * err

            if err2 > -dy:
                err -= dy
                x0 += sx

            if err2 < dx:
                err += dx
                y0 += sy

    def edgeFunction(self, v0, v1, v2):
        return (v0.x - v1.x) * (v2.y - v1.y) - (v2.x - v1.x) * (v0.y - v1.y)

    def rasterizeTriangle(self, p0, p1, p2, doGouraud = True):
        # Determine the bounding box of the triangle
        min_x = min(p0.x, p1.x, p2.x)
        max_x = max(p0.x, p1.x, p2.x)
        min_y = min(p0.y, p1.y, p2.y)
        max_y = max(p0.y, p1.y, p2.y)

        # Iterate through each pixel in the bounding box
        for x in range(int(min_x), int(max_x) + 1):
            for y in range(int(min_y), int(max_y) + 1):
                # Calculate the edge functions for each vertex
                # if 0 <= x < self.w_width and 0 <= y < self.w_height:
                e0 = self.edgeFunction(p1, p2, Vertex(x, y, 0, 0, 0, 0, None))
                e1 = self.edgeFunction(p2, p0, Vertex(x, y, 0, 0, 0, 0, None))
                e2 = self.edgeFunction(p0, p1, Vertex(x, y, 0, 0, 0, 0, None))
                # Check if the pixel is inside the triangle
                if e0 >= 0 and e1 >= 0 and e2 >= 0:
                    # Calculate the total area of the triangle
                    total_area = e0 + e1 + e2
                    # Calculate barycentric coordinates
                    lambda0 = 1
                    lambda1 = 1
                    lambda2 = 1
                    if total_area != 0:
                        lambda0 = e0 / total_area
                        lambda1 = e1 / total_area
                        lambda2 = e2 / total_area

                    # Calculate depth
                    depth = lambda0 * p0.z + lambda1 * p1.z + lambda2 * p2.z

                    # Z-buffer test
                    if depth < self.zbuffer[x, y]:
                        self.zbuffer[x, y] = depth

                        # Interpolate colors for Gouraud shading
                        r = lambda0 * p0.r * 255 + lambda1 * p1.r * 255 + lambda2 * p2.r * 255
                        g = lambda0 * p0.g * 255 + lambda1 * p1.g * 255 + lambda2 * p2.g * 255
                        b = lambda0 * p0.b * 255 + lambda1 * p1.b * 255 + lambda2 * p2.b * 255

                        self.win.set_pixel(x, y, int(r), int(g), int(b))
                        if not doGouraud:
                            # Interpolate normal, light direction, view direction, and reflection direction
                            interpolated_normal = lambda0 * p0.data['Normal'] + lambda1 * p1.data['Normal'] + lambda2 * \
                                                  p2.data['Normal']
                            interpolated_light_dir = lambda0 * p0.data['LightDir'] + lambda1 * p1.data[
                                'LightDir'] + lambda2 * p2.data['LightDir']
                            interpolated_view_dir = lambda0 * p0.data['ViewDir'] + lambda1 * p1.data[
                                'ViewDir'] + lambda2 * p2.data['ViewDir']
                            interpolated_reflect_dir = lambda0 * p0.data['ReflectDir'] + lambda1 * p1.data[
                                'ReflectDir'] + lambda2 * p2.data['ReflectDir']

                            # Normalize the interpolated vectors
                            interpolated_normal = interpolated_normal / np.linalg.norm(interpolated_normal)

                            # Recalculate lighting for the interpolated values
                            ambient = self.calculateAmbient(p0.data['ocolor'])
                            diffuse = self.light_color * p0.data['ocolor'] * max(np.dot(interpolated_normal,
                                                                              interpolated_light_dir), 0)

                            reflect_dir = ((2 * np.dot(interpolated_normal, interpolated_light_dir)) * interpolated_normal) - interpolated_light_dir
                            reflect_dir = reflect_dir / np.linalg.norm(reflect_dir)

                            interpolated_reflect_dir = lambda0 * reflect_dir + lambda1 * reflect_dir + lambda2 * reflect_dir

                            specular = self.light_color * p0.data['scolor'] * (max(np.dot(
                                interpolated_reflect_dir, interpolated_view_dir), 0) ** p0.data['exponent'])

                            # The final color is the sum of the ambient, diffuse, and specular components
                            color = ambient * p0.data['k'][0] + diffuse * p0.data['k'][1] + specular * p0.data['k'][2]
                            color = np.clip(color, 0, 1) * 255

                            # Set the pixel color
                            self.win.set_pixel(x, y, int(color[0]), int(color[1]), int(color[2]))

    def sampleTextureColor(self, texture, u, v, tex_width, tex_height):
        # Map UV coordinates to texture coordinates
        tex_x = int(u * tex_width)
        tex_y = int(v * tex_height)

        # Clamp coordinates to the texture's dimensions
        tex_x = max(0, min(tex_x, tex_width - 1))
        tex_y = max(0, min(tex_y, tex_height - 1))

        # Sample the color from the texture
        color = texture.getpixel((tex_x, tex_y))
        return color

    def rasterizeTexturedTriangle(self, p0, p1, p2, texture, tex_width, tex_height, myTexture = False, max_iter = 0):
        # Determine the bounding box of the triangle
        min_x = min(p0.x, p1.x, p2.x)
        max_x = max(p0.x, p1.x, p2.x)
        min_y = min(p0.y, p1.y, p2.y)
        max_y = max(p0.y, p1.y, p2.y)

        # Iterate through each pixel in the bounding box
        for x in range(int(min_x), int(max_x) + 1):
            for y in range(int(min_y), int(max_y) + 1):
                # Calculate the edge functions for each vertex
                e0 = self.edgeFunction(p1, p2, Vertex(x, y, 0, 0, 0, 0, None))
                e1 = self.edgeFunction(p2, p0, Vertex(x, y, 0, 0, 0, 0, None))
                e2 = self.edgeFunction(p0, p1, Vertex(x, y, 0, 0, 0, 0, None))
                # Check if the pixel is inside the triangle
                if e0 >= 0 and e1 >= 0 and e2 >= 0:
                    # Calculate the total area of the triangle
                    total_area = e0 + e1 + e2
                    # Calculate barycentric coordinates
                    lambda0 = 1
                    lambda1 = 1
                    lambda2 = 1
                    if total_area != 0:
                        lambda0 = e0 / total_area
                        lambda1 = e1 / total_area
                        lambda2 = e2 / total_area

                    # Calculate depth
                    depth = lambda0 * p0.z + lambda1 * p1.z + lambda2 * p2.z
                    u = lambda0 * p0.data['uv0'][0] + lambda1 * p1.data['uv1'][0] + lambda2 * p2.data['uv2'][0]
                    v = lambda0 * p0.data['uv0'][1] + lambda1 * p1.data['uv1'][1] + lambda2 * p2.data['uv2'][1]

                    if myTexture:
                        # c = complex(u * 3 - 2, v * 2 - 1)
                        # mandel_value = self.mandelbrot(c, max_iter)
                        # texture_color = self.mandelbrot_color(mandel_value)
                        pattern_value = self.swirl(u, v)
                        color_value = int(pattern_value * 255)
                        texture_color = (255- color_value, color_value, -color_value)
                    else:
                        # Sample texture color
                        texture_color = self.sampleTextureColor(texture, u, v, tex_width, tex_height)

                    # Z-buffer test
                    if depth < self.zbuffer[x, y]:
                        self.zbuffer[x, y] = depth

                        # Set the pixel color in the framebuffer
                        self.win.set_pixel(x, y, *texture_color)

    def drawTrianglesTextures(self, vertex_pos, indices, uvs, texture):
        tex_width, tex_height = texture.size

        for i in range(0, len(indices), 3):
            # Extract vertices indices and corresponding UV coordinates for each triangle
            i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]
            uv0, uv1, uv2 = uvs[i0 * 2:i0 * 2 + 2], uvs[i1 * 2:i1 * 2 + 2], uvs[i2 * 2:i2 * 2 + 2]
            # Create Vertex objects for each vertex of the triangle
            p0 = Vertex(vertex_pos[i0 * 3], vertex_pos[i0 * 3 + 1], vertex_pos[i0 * 3 + 2], 0, 0, 0, {'uv0': uv0})
            p1 = Vertex(vertex_pos[i1 * 3], vertex_pos[i1 * 3 + 1], vertex_pos[i1 * 3 + 2], 0, 0, 0, {'uv1': uv1})
            p2 = Vertex(vertex_pos[i2 * 3], vertex_pos[i2 * 3 + 1], vertex_pos[i2 * 3 + 2], 0, 0, 0, {'uv2': uv2})

            # Transform vertices
            p0, p1, p2 = self.transform(p0, p1, p2)

            # Rasterize the triangle
            self.rasterizeTexturedTriangle(p0, p1, p2, texture, tex_width, tex_height)

    def mandelbrot(self, c, max_iter):
        z = 0
        for n in range(max_iter):
            if abs(z) > 2:
                return n / max_iter
            z = z * z + c
        return 0

    def mandelbrot_color(self, mandel_value):
        if mandel_value == 1:
            return (0, 0, 0)  # Black for points inside the set
        else:
            # Create a color gradient, e.g., from blue to red
            return (int(255 * -mandel_value), int(255 * (1 - mandel_value)), int(128 * (1 - mandel_value)))

    def swirl(self, u, v, size=25):
        # Convert UV to polar coordinates
        x = (u - 0.5) * 2
        y = (v - 0.5) * 2
        r = math.sqrt(x ** 2 + y ** 2)
        theta = math.atan2(y, x)

        # Create a radial symmetry
        frequency = size * theta + r * 10
        pattern = (math.sin(frequency) + 1) / 2
        return pattern

    def drawTrianglesMyTextures(self, vertex_pos, indices, uvs, max_iter=100):

        for i in range(0, len(indices), 3):
            # Extract vertices indices and corresponding UV coordinates for each triangle
            i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]
            uv0, uv1, uv2 = uvs[i0 * 2:i0 * 2 + 2], uvs[i1 * 2:i1 * 2 + 2], uvs[i2 * 2:i2 * 2 + 2]
            # Create Vertex objects for each vertex of the triangle
            p0 = Vertex(vertex_pos[i0 * 3], vertex_pos[i0 * 3 + 1], vertex_pos[i0 * 3 + 2], 0, 0, 0, {'uv0': uv0})
            p1 = Vertex(vertex_pos[i1 * 3], vertex_pos[i1 * 3 + 1], vertex_pos[i1 * 3 + 2], 0, 0, 0, {'uv1': uv1})
            p2 = Vertex(vertex_pos[i2 * 3], vertex_pos[i2 * 3 + 1], vertex_pos[i2 * 3 + 2], 0, 0, 0, {'uv2': uv2})

            # Transform vertices
            p0, p1, p2 = self.transform(p0, p1, p2)

            # Rasterize the triangle
            self.rasterizeTexturedTriangle(p0, p1, p2, 0, 0, 0, True, max_iter)


    def drawTrianglesPhong(self, vertex_pos, indices, normals, ocolor, scolor, k, exponent, doGouraud):
        """
        Draws triangles using the Phong shading model.
        """

        for i in range(0, len(indices), 3):
            # Extract vertices and normals for each triangle
            i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]


            # Initialize arrays to store vertices and normals
            Vertex_vector = []
            Normal_vector = []
            Data_list = []
            color_list = []

            # Process each vertex of the triangle
            for j in [i0, i1, i2]:
                p = np.array([vertex_pos[j * 3], vertex_pos[j * 3 + 1], vertex_pos[j * 3 + 2]])
                n = np.array([normals[j * 3], normals[j * 3 + 1], normals[j * 3 + 2]])

                ambient = self.calculateAmbient(ocolor)
                diffuse = self.calculateDiffuse(p, ocolor, n)
                specular = self.calculateSpecular(p, n, scolor, exponent)

                pos = p / np.linalg.norm(p)
                light_dir = self.light_position - pos
                light_dir = light_dir / np.linalg.norm(light_dir)

                view_dir = self.eye - p
                view_dir = view_dir / np.linalg.norm(view_dir)

                # Calculate the reflection vector
                reflect_dir = ((2 * np.dot(n, light_dir)) * n) - light_dir
                reflect_dir = reflect_dir / np.linalg.norm(reflect_dir)

                data = {
                    'A': ambient * k[0],
                    'D': diffuse * k[1],
                    'S': specular * k[2],
                    'Normal': n,
                    'LightDir': light_dir,
                    'ViewDir': view_dir,
                    'ReflectDir': reflect_dir,
                    'k': k,
                    'exponent': exponent,
                    'ocolor': ocolor,
                    'scolor': scolor
                }
                r, g, b = ambient * k[0] + diffuse * k[1] + specular * k[2]
                color_list.append(np.array([r, g, b]))
                Vertex_vector.append(p)
                Normal_vector.append(n)
                Data_list.append(data)

            # Create Vertex objects for each vertex of the triangle
            P0 = Vertex(*Vertex_vector[0], *color_list[0], Data_list[0])
            P1 = Vertex(*Vertex_vector[1], *color_list[1], Data_list[1])
            P2 = Vertex(*Vertex_vector[2], *color_list[2], Data_list[2])



            transformed_P0, transformed_P1, transformed_P2 = self.transform(P0, P1, P2)
            if doGouraud:
                # Rasterize the transformed triangle
                self.rasterizeTriangle(transformed_P0, transformed_P1, transformed_P2)
            else:
                self.rasterizeTriangle(transformed_P0, transformed_P1, transformed_P2, doGouraud)

    def transform(self, p0, p1, p2):
        # Transform the vertices
        transformed_p0 = self.applyTransformations(np.array([[p0.x], [p0.y], [p0.z], [1.0]]))
        transformed_p1 = self.applyTransformations(np.array([[p1.x], [p1.y], [p1.z], [1.0]]))
        transformed_p2 = self.applyTransformations(np.array([[p2.x], [p2.y], [p2.z], [1.0]]))

        transformed_P0 = Vertex(int(transformed_p0[0] // transformed_p0[3]),
                                int(transformed_p0[1] // transformed_p0[3]),
                                int(transformed_p0[2] // transformed_p0[3]), p0.r, p0.g, p0.b, p0.data)
        transformed_P1 = Vertex(int(transformed_p1[0] // transformed_p1[3]),
                                int(transformed_p1[1] // transformed_p1[3]),
                                int(transformed_p1[2] // transformed_p1[3]), p1.r, p1.g, p1.b, p1.data)
        transformed_P2 = Vertex(int(transformed_p2[0] // transformed_p2[3]),
                                int(transformed_p2[1] // transformed_p2[3]),
                                int(transformed_p2[2] // transformed_p2[3]), p2.r, p2.g, p2.b, p2.data)

        return transformed_P0, transformed_P1, transformed_P2

    def transformAndRasterize(self, vertex_pos, indices, r, g, b):
        for i in range(0, len(indices), 3):
            i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]

            # Create Vertex objects for each vertex of the triangle
            p0 = Vertex(vertex_pos[i0 * 3], vertex_pos[i0 * 3 + 1], vertex_pos[i0 * 3 + 2], r, g, b)
            p1 = Vertex(vertex_pos[i1 * 3], vertex_pos[i1 * 3 + 1], vertex_pos[i1 * 3 + 2], r, g, b)
            p2 = Vertex(vertex_pos[i2 * 3], vertex_pos[i2 * 3 + 1], vertex_pos[i2 * 3 + 2], r, g, b)

            # Apply the transformations to the vertices
            transformed_p0 = self.applyTransformations(np.array([[p0.x], [p0.y], [p0.z], [1.0]]))
            transformed_p1 = self.applyTransformations(np.array([[p1.x], [p1.y], [p1.z], [1.0]]))
            transformed_p2 = self.applyTransformations(np.array([[p2.x], [p2.y], [p2.z], [1.0]]))

            transformed_P0 = Vertex(int(transformed_p0[0] // transformed_p0[3]),
                                    int(transformed_p0[1] // transformed_p0[3]),
                                    int(transformed_p0[2] // transformed_p0[3]), p0.r, p0.g, p0.b)
            transformed_P1 = Vertex(int(transformed_p1[0] // transformed_p1[3]),
                                    int(transformed_p1[1] // transformed_p1[3]),
                                    int(transformed_p1[2] // transformed_p1[3]), p1.r, p1.g, p1.b)
            transformed_P2 = Vertex(int(transformed_p2[0] // transformed_p2[3]),
                                    int(transformed_p2[1] // transformed_p2[3]),
                                    int(transformed_p2[2] // transformed_p2[3]), p2.r, p2.g, p2.b)

            yield transformed_P0, transformed_P1, transformed_P2

    def drawTrianglesC(self, vertex_pos, indices, r, g, b, r1 = -1, g1 = -1,b1 = -1):
        for transformed_P0, transformed_P1, transformed_P2 in self.transformAndRasterize(vertex_pos, indices, r, g, b):
            self.rasterizeTriangle(transformed_P0, transformed_P1, transformed_P2)
        if r1 != -1:
            self.drawTrianglesWireframe(vertex_pos, indices, r1, g1, b1)


    def drawTrianglesWireframe(self, vertex_pos, indices, r, g, b):
        for transformed_P0, transformed_P1, transformed_P2 in self.transformAndRasterize(vertex_pos, indices, r, g, b):
            p0, p1, p2 = np.array([transformed_P0.x, transformed_P0.y, transformed_P0.z]), np.array(
                            [transformed_P1.x, transformed_P1.y, transformed_P1.z]), np.array(
                            [transformed_P2.x, transformed_P2.y, transformed_P2.z])

            e1, e2 = p1 - p0, p2 - p0
            x, y, z = np.cross(e1, e2)

            if z > 0:
                self.rasterizeLine(transformed_P0.x, transformed_P0.y, transformed_P1.x, transformed_P1.y, r, g, b)
                self.rasterizeLine(transformed_P1.x, transformed_P1.y, transformed_P2.x, transformed_P2.y, r, g, b)
                self.rasterizeLine(transformed_P0.x, transformed_P0.y, transformed_P2.x, transformed_P2.y, r, g, b)

    def clearModelTransform(self):
        # Reset the model transform matrix to the identity matrix
        self.model_transform = np.identity(4)

    def translate(self, x, y, z):
        # Create a translation matrix and multiply it with the model transform
        translation_matrix = np.array([[1, 0, 0, x],
                                       [0, 1, 0, y],
                                       [0, 0, 1, z],
                                       [0, 0, 0, 1]])
        self.model_transform = translation_matrix @ self.model_transform

    def scale(self, x, y, z):
        # Create a scaling matrix and multiply it with the model transform
        scaling_matrix = np.array([[x, 0, 0, 0],
                                   [0, y, 0, 0],
                                   [0, 0, z, 0],
                                   [0, 0, 0, 1]])
        self.model_transform = scaling_matrix @ self.model_transform

    def rotatex(self, angle):
        radians = np.radians(angle)
        cos_angle = np.cos(radians)
        sin_angle = np.sin(radians)

        rotation_matrix = np.array([[1, 0, 0, 0],
                                   [0, cos_angle, -sin_angle, 0],
                                   [0, sin_angle, cos_angle, 0],
                                   [0, 0, 0, 1]])
        self.model_transform = rotation_matrix @ self.model_transform

    def rotatey(self, angle):
        radians = np.radians(angle)
        cos_angle = np.cos(radians)
        sin_angle = np.sin(radians)

        rotation_matrix = np.array([[cos_angle, 0, sin_angle, 0],
                                    [0, 1, 0, 0],
                                    [-sin_angle, 0, cos_angle, 0],
                                    [0, 0, 0, 1]])
        self.model_transform = rotation_matrix @ self.model_transform

    def rotatez(self, angle):
        radians = np.radians(angle)
        cos_angle = np.cos(radians)
        sin_angle = np.sin(radians)

        rotation_matrix = np.array([[cos_angle, -sin_angle, 0, 0],
                                    [sin_angle, cos_angle, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        self.model_transform = rotation_matrix @ self.model_transform


    def defineClipWindow (self, t, b, r, l):
        # Calculate the normalization transform matrix based on the clip window
        self.normalization_transform = np.array([[2 / (r - l), 0, 0, ((-2 * l) / (r - l)) - 1],
                                                [0, 2 / (t - b), 0, ((-2 * b) / (t - b)) - 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]])
        self.model_transform = self.normalization_transform @ self.model_transform

    def defineViewWindow(self, t, b, r, l):
        # Calculate the view transform matrix based on the view window
        x_scale = (r - l) / 2.0
        y_scale = (t - b) / 2.0
        x_translate = (r + l) / 2.0
        y_translate = (t + b) / 2.0
        view_transform = np.array([[x_scale, 0.0, 0.0, x_translate],
                                        [0.0, y_scale, 0.0, y_translate],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]])
        self.model_transform = view_transform @ self.model_transform

    def applyTransformations(self, vertex):
        # Apply model, view, and projection transformations
        return self.default_view_transform  @ self.projection_transform @  self.camera_transform @ self.stack[-1] @ vertex

    def Transform(self, vertex):
        # Apply model, view, and projection transformations
        return self.stack[-1] @ vertex

    def drawTriangles(self, vertex_pos, colors, indices):
        for i in range(0, len(indices), 3):
            i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]

            # Create Vertex objects for each vertex of the triangle
            p0 = Vertex(vertex_pos[i0 * 2], vertex_pos[i0 * 2 + 1], *colors[i0 * 3:i0 * 3 + 3])
            p1 = Vertex(vertex_pos[i1 * 2], vertex_pos[i1 * 2 + 1], *colors[i1 * 3:i1 * 3 + 3])
            p2 = Vertex(vertex_pos[i2 * 2], vertex_pos[i2 * 2 + 1], *colors[i2 * 3:i2 * 3 + 3])
            # Apply the transformations to the vertices
            transformed_p0 = self.applyTransformations(np.array([p0.x, p0.y, 1.0]))
            transformed_p1 = self.applyTransformations(np.array([p1.x, p1.y, 1.0]))
            transformed_p2 = self.applyTransformations(np.array([p2.x, p2.y, 1.0]))

            # Create Vertex objects with transformed coordinates
            transformed_P0 = Vertex(transformed_p0[0] / transformed_p0[2], transformed_p0[1] / transformed_p0[2], p0.r,
                                    p0.g, p0.b)
            transformed_P1 = Vertex(transformed_p1[0] / transformed_p1[2], transformed_p1[1] / transformed_p1[2], p1.r,
                                    p1.g, p1.b)
            transformed_P2 = Vertex(transformed_p2[0] / transformed_p2[2], transformed_p2[1] / transformed_p2[2], p2.r,
                                    p2.g, p2.b)

            self.rasterizeTriangle(transformed_P0, transformed_P1, transformed_P2)

    def drawClippedPoly(self, vertices):
        nverts = vertices.size
        if nverts < 3:
            return

        # chose your pivot vertex to be the first
        p0 = Vertex(round(vertices[0].x), round(vertices[0].y), vertices[0].r, vertices[0].g, vertices[0].b)
        endV = 2;
        while endV < nverts:
            p1 = Vertex(round(vertices[endV - 1].x), round(vertices[endV - 1].y), vertices[endV - 1].r,
                        vertices[endV - 1].g, vertices[endV - 1].b)
            p2 = Vertex(round(vertices[endV].x), round(vertices[endV].y), vertices[endV].r, vertices[endV].g,
                        vertices[endV].b)
            # Apply the transformations to the vertices
            transformed_p0 = self.applyTransformations(np.array([p0.x, p0.y, 1.0]))
            transformed_p1 = self.applyTransformations(np.array([p1.x, p1.y, 1.0]))
            transformed_p2 = self.applyTransformations(np.array([p2.x, p2.y, 1.0]))

            # Create Vertex objects with transformed coordinates
            transformed_P0 = Vertex(transformed_p0[0]/transformed_p0[2], transformed_p0[1]/transformed_p0[2], p0.r, p0.g, p0.b)
            transformed_P1 = Vertex(transformed_p1[0]/transformed_p1[2], transformed_p1[1]/transformed_p1[2], p1.r, p1.g, p1.b)
            transformed_P2 = Vertex(transformed_p2[0]/transformed_p2[2], transformed_p2[1]/transformed_p2[2], p2.r, p2.g, p2.b)
            self.rasterizeTriangle(transformed_P0, transformed_P1, transformed_P2)
            endV = endV + 1

    # go is called on every update of the window display loop
    # have your engine draw stuff in the window.
    def go(self):
        if (self.keypressed == 1):
            # default scene
            self.default_action()

        if (self.keypressed == 2):
            # add you own unique scene here
            self.win.clearFB (0, 0, 0)
            # Draw a yellow circle with violet background
            center_x = self.w_width // 2
            center_y = self.w_height // 2
            radius = 200
            yellow = (255, 255, 0)
            violet = (100, 0, 155)
            # background
            for x in range(self.w_height):
                for y in range(self.w_width):
                    self.win.set_pixel(x, y, *violet)
            # circle
            for x in range(center_x - radius, center_x + radius):
                for y in range(center_y - radius, center_y + radius):
                    if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                        self.win.set_pixel(x, y, *yellow)

        # push the window's framebuffer to the window
        self.win.applyFB()

    def keyboard (self, key) :
        if (key == '1'):
            self.keypressed = 1
            self.go()
        if (key == '2'):
            self.keypressed = 2
            self.go()






