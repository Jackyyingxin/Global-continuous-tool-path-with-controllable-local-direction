import math
from GeomBase import *
from Line import *
from Segment import *
from Polyline import *
from Layer import *
from Ray import *


def nearZero(x):
    return True if math.fabs(x) < epsilon else False

def intersectLineLine(line1: Line,line2:Line):
    P1, V1, P2, V2 = line1.P, line1.V, line2.P, line2.V
    P1P2 = P1.pointTo(P2)
    deno= V1.dy * V2.dx -V1.dx * V2.dy
    if deno != 0:
        t1 = -(-P1P2.dy * V2.dx + P1P2.dx * V2.dy)/deno
        t2 = -(-P1P2.dy * V1.dx + P1P2.dx * V1.dy)/deno
        return P1 +V1.amplified(t1), t1, t2
    else:
        deno =V1.dz * V2.dy - V1.dy * V2.dz
        if deno != 0:
            t1 = -(-P1P2.dz * V2.dy + P1P2.dy * V2.dz) / deno
            t2 = -(-P1P2.dz * V1.dy + P1P2.dy * V1.dz) / deno
            return P1 + V1.amplified(t1), t1, t2
    return None, 0, 0

def pointOnRay(p:Point3D, ray:Ray):
    v = ray.P.pointTo(p)
    return True if v.dotProduct(ray.V)>=0and v.crossProduct(ray.V).isZeroVector()else False

def pointInPolygon(p:Point3D, polygon:Polyline):
    passCount = 0
    ray = Ray(p, Vector3D(1, 0, 0))
    segments = []
    for i in range(polygon.count()-1):
        seg = Segment(polygon.point(i), polygon.point(i+1))
        segments.append(seg)
    for seg in segments:
        line1, line2 = Line(ray.P, ray.V), Line(seg.A, seg.direction())
        P, t1, t2 = intersectLineLine(line1, line2)
        if P is not None:
            if nearZero(t1):return -1
            elif seg.A.y != p.y and seg.B.y != p.y and t1 >0 and t2 >0 and t2 < seg.length():
                passCount += 1
    upSegments, downSegments = [], []
    for seg in segments:
        if seg.A.isIdentical(ray.P) or seg.B.isIdentical(ray.P):
            return -1
        elif pointOnRay(seg.A, ray) ^ pointOnRay(seg.B, ray):
            if seg.A.y >= p.y and seg.B >= p.y: upSegments.append(seg)
            elif seg.A.y <= p.y and seg.B <= p.y: downSegments.append(seg)
    passCount += min(len(downSegments), len(upSegments))
    if passCount % 2 == 1:
        return 1
    return 0

def intersect(obj1,obj2):
    line1, line2 = obj1, Line(obj2.A, obj2.direction())
    P, t1, t2 = intersectLineLine(line1, line2)
    return P if P is not None and t2 >=0 and t2 <= obj2.length() else None

def adjustPolygonDirs(polygons):
    for i in range(len(polygons)):
        pt = polygons[i].startPoint()
        insideCount = 0
        for j in range(len(polygons)):
            if j == i:continue
            restPoly = polygons[j]
            if 1 == pointInPolygon(pt, restPoly):
                insideCount += 1 ##可能有问题原文是evencount但是没找到这个变量的初始化
        if insideCount % 2 == 0:
            polygons[i].makeCCW()
        else:
            polygons[i].makeCW()


def rotatePolygons(polygons, angle, center = None):
    dx = 0 if center is None else center.x
    dy = 0 if center is None else center.y
    mt = Matrix3D.createTranslateMatrix(-dx, -dy, 0)
    mr = Matrix3D.createRotateMatrix('Z', angle)
    mb = Matrix3D.createTranslateMatrix(dx, dy, 0)
    m = mt * mr * mb
    newPolys = []
    for poly in polygons:
        newPolys.append(poly.multiplied(m))
    return newPolys
