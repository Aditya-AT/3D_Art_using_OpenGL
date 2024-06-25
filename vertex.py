

class Vertex:
    def __init__(self, x, y, z, r, g, b, data):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
        # self.normal = {'nx': 0, 'ny': 0, 'nz': 0}  # Initialize normal vector components
        self.data = data

    # # Method to set the normal for the vertex
    # def set_normal(self, nx, ny, nz):
    #     self.normal['nx'] = nx
    #     self.normal['ny'] = ny
    #     self.normal['nz'] = nz
        

def getTriangleMin(p0, p1, p2):
    V = Vertex(p0.x, p0.y, p0.z, 0, 0, 0)  # Initialize with 3D values
    if p1.x < V.x:
        V.x = p1.x
    if p2.x < V.x:
        V.x = p2.x
    if p1.y < V.y:
        V.y = p1.y
    if p2.y < V.y:
        V.y = p2.y
    if p1.z < V.z:
        V.z = p1.z
    if p2.z < V.z:
        V.z = p2.z

    return V

def getTriangleMax(p0, p1, p2):
    V = Vertex(p0.x, p0.y, p0.z, 0, 0, 0)  # Initialize with 3D values
    if p1.x > V.x:
        V.x = p1.x
    if p2.x > V.x:
        V.x = p2.x
    if p1.y > V.y:
        V.y = p1.y
    if p2.y > V.y:
        V.y = p2.y
    if p1.z > V.z:
        V.z = p1.z
    if p2.z > V.z:
        V.z = p2.z

    return V

