import math
from Polyline import  *
class ClipperAdaptor:
    def __init__(self, digits =7):
        self.f = math.pow(10, digits)
        self.arcTolerance = 0.005
    def toPath(self,poly):
        path = []
        for pt in poly.points:
            path.append((pt.x * self.f, pt.y * self.f, pt.z *self.f))
        return path
    def toPaths(self,polys):
        paths = []
        for poly in polys:
            paths.append(self.toPath(poly))
        return paths
    def toPoly(self,path, z=0,closed= True):
        poly = Polyline()
        for tp in path:
            poly.addPoint(Point3D(tp[0]/self.f, tp[1]/self.f,z))
        if len(path)>0 and closed:
            poly.addPoint(poly.startPoint())
        return poly
    def toPolys(self, paths,z=0,closed = True):
        polys =[]
        for path in paths:
            polys.append(self.toPoly(path, z, closed))
        return polys

