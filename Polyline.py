import copy
from GeomBase import *


class Polyline:
    def __init__(self):
        """

        :rtype: object
        """
        self.points = []
        self.belongtowhichpoly=-1
        self.index_in_splitcurve = None
        self.overlap = []
        self.error_poly = None
        self.y_min = None
        self.y_max = None
        self.angle = None

    def __str__(self):
        if self.count()>0:
            return 'Polyline Point number:%s Start %s End %s'% (self.count(), str(self.startPoint()),str(self.endPoint()))
        else: return 'Polyline Point number:0'
    def clone(self):
        poly = Polyline()
        for pt in self.points:
            poly.addPoint(pt.clone())
        return poly
    def maxy(self):
        maxy = float("-inf")
        for point in self.points:
            if point.y > maxy:
                maxy = point.y
        return maxy

    def miny(self):
        miny = float("inf")
        for point in self.points:
            if point.y < miny:
                miny = point.y
        return miny
    def count(self):
        return len(self.points)

    def addPoint(self, pt):
        self.points.append(pt)

    def raddPoint(self, pt):
        self.points.insert(0, pt)

    def addTuple(self,tuple):
        self.points.append(Point3D(tuple[0], tuple[1], tuple[2]))
    def removePoint(self,index):
        return self.points.pop(index)
    def point(self,index):
        return self.points[index]
    def startPoint(self):
        return self.points[0]
    def endPoint(self):
        return  self.points[len(self.points)-1]
    def isClosed(self):
        if self.count() <=2:return False
        return self.startPoint().isCoincide(self.endPoint())

    def multiply(self,m):
        for pt in self.points:
            pt.multiply(m)##书中原文是multiply,但point类中没有根据意思自己建了一个
    def multiplied(self,m):
        poly=Polyline()
        for pt in self.points:
            poly.addPoint(pt * m)##有改动
        poly.belongtowhichpoly=self.belongtowhichpoly
        return poly

    def Segments(self):
        segments = []  # get all segments of polyline
        a = self.points[0]
        for i in range(1, len(self.points)):
            b = self.points[i]
            segments.append((a, b))
            a = b
        return segments

    def SelfIntersections(self):
        epsilon = 1e-14

        def LineIntersects(p1, p2, q1, q2):  # check if intersection present for segment

            def Clockwise(p0, p1, p2):  # check for intersection of all point combinations
                dx1 = p1.x - p0.x
                dy1 = p1.y - p0.y
                dx2 = p2.x - p0.x
                dy2 = p2.y - p0.y
                d = dx1 * dy2 - dy1 * dx2  # distance
                if d > epsilon: return 1
                if d < epsilon: return -1
                if dx1 * dx2 < -epsilon or dy1 * dy2 < -epsilon: return -1
                if dx1 * dx1 + dy1 * dy1 < (dx2 * dx2 + dy2 * dy2) + epsilon: return 1
                return 0

            return (Clockwise(p1, p2, q1) * Clockwise(p1, p2, q2) <= 0) and (
                    0 >= Clockwise(q1, q2, p1) * Clockwise(q1, q2, p2))

        def line(p1, p2):  # find A,B,C coefficients for line
            A = (p1.y - p2.y)
            B = (p2.x - p1.x)
            C = (p1.x * p2.y - p2.x * p1.y)
            return A, B, -C

        def IntersectionPoint(L1, L2):  # find intersection point for two lines
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return x, y
            else:
                return False

        segments = self.Segments()
        present = []
        for seg1 in segments:  # for all combinations of segments
            for seg2 in segments:
                if (seg1[0] != seg2[1] and seg1[1] != seg2[0]) and \
                        (seg1[1] != seg2[1] and seg1[0] != seg2[0]):  # exclude points of segments connection
                    if LineIntersects(seg1[0], seg1[1], seg2[0], seg2[1]) == True:  # check if current segment intersect
                        L1 = line(seg1[0], seg1[1])  # find line coefficients (A, B, C) for segment parts
                        L2 = line(seg2[0], seg2[1])
                        P = IntersectionPoint(L1, L2)  # get point coordinate of intersection
                        present.append(P)  # add points to list
        answer = sorted(set(present))  # clean list from duplicate points
        return answer  # return list of points

    def __eq__(self, other):#顺序相等，大小也相等才相等
        for i in range(len(self.points)):
            if self.points[i] != other.points[i]:
                return False
        return True
    def __hash__(self):
        return hash(tuple(self.points))

    def area(self):
        area = 0
        pts = self.points
        nPts = len(pts)
        j = nPts - 1
        i = 0
        for point in pts:
            p1 = pts[i]
            p2 = pts[j]
            area += (p1.x * p2.y)
            area -= p1.y * p2.x
            j = i
            i += 1

        area /= 2
        return area

    def centroid(self):
        pts = self.points
        nPts = len(pts)
        x = 0
        y = 0
        j = nPts - 1
        i = 0

        for point in pts:
            p1 = pts[i]
            p2 = pts[j]
            f = p1.x * p2.y - p2.x * p1.y
            x += (p1.x + p2.x) * f
            y += (p1.y + p2.y) * f
            j = i
            i += 1

        f = self.area() * 6
        return Point3D(x / f, y / f)










