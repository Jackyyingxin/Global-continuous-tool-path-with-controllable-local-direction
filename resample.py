from GeomBase import *
from Polyline import *
import math
def distance(A:Point3D,B:Point3D):
    return math.sqrt(pow(A.x - B.x, 2) + pow(A.y - B.y, 2))


def pl_arcLength(xycoordinates:list, isOpen:bool):
    length = 0
    for i in range(len(xycoordinates)-1):
        length = length+distance(xycoordinates[i], xycoordinates[i+1])
    if isOpen:
        length += distance(xycoordinates[0], xycoordinates[-1])
    return length

def ResampleCurve(Curve:Polyline,N):
    assert len(Curve.points)>0
    resamplepl = Polyline()
    resamplepl.belongtowhichpoly=Curve.belongtowhichpoly
    resamplepl.points=[]
    #for i in range(N):
    #    resamplepl.points.append([])
    pl = Curve.points
    resamplepl.points.append(pl[0])
    Curvelength = pl_arcLength(pl, 0)
    resample_size = Curvelength/ N
    curr = 0
    dist = 0.0
    i=1
    while(i<N):
        if(curr >= len(pl) - 1):
            print("erro")
        last_dist = distance(pl[curr], pl[curr + 1])
        dist = dist+last_dist
        if (dist >= resample_size):
            _d = last_dist - (dist - resample_size)
            cp = Point3D(pl[curr].x, pl[curr].y, pl[curr].z)
            cp1 = Point3D(pl[curr + 1].x, pl[curr + 1].y, pl[curr+1].z)
            resamplepl.points.append(cp + cp.pointTo(cp1).normalized().amplified(_d))
            i = i+1
            dist = last_dist - _d # #dist 是从当前节点到原曲线第二个节点的距离，如果大于一个resamplesize就跨过一个cuur。
            while (dist - resample_size > 1e-10):
                resamplepl.points.append(resamplepl.points[i - 1] + cp.pointTo(cp1).normalized().amplified(resample_size))
                dist -= resample_size
                i = i+1
        curr = curr+1
    return resamplepl

def Resample_gen(Curve:Polyline,N):
    pass



def insertcurve(Curve:Polyline,N):
    insertcurve=Polyline()
    insertcurve.belongtowhichpoly=Curve.belongtowhichpoly
    endpoints=Curve.points
    insertcurve.points.append(Curve.points[0])
    for i in  range(len(endpoints)-1):
        temp = []
        dir = endpoints[i].pointTo(endpoints[i+1]).normalized()
        if (dir.dx==-1 or dir.dx==1) and dir.dy==0 and dir.dz==0:
            totaldistace = endpoints[i].distance(endpoints[i + 1])
            divdistance = totaldistace / N
            for j in range(1, N):
                temp.append(endpoints[i].translated(dir.amplified(divdistance * j)))
            temp.append(endpoints[i + 1])
            insertcurve.points.extend(temp)
        else:
            insertcurve.points.append(endpoints[i+1])

    return insertcurve
def curve_insertpoints(Curve:Polyline):
    insertcurve = Polyline()
    insertcurve.belongtowhichpoly = Curve.belongtowhichpoly
    endpoints = Curve.points
    insertcurve.points.append(Curve.points[0])
    for i in  range(len(endpoints)-1):
        temp = []
        dir = endpoints[i].pointTo(endpoints[i+1]).normalized()
        if (dir.dx==-1 or dir.dx==1) and dir.dy==0 and dir.dz==0:
            totaldistace = endpoints[i].distance(endpoints[i + 1])
            divdistance = totaldistace / N
            for j in range(1, N):
                temp.append(endpoints[i].translated(dir.amplified(divdistance * j)))
            temp.append(endpoints[i + 1])
            insertcurve.points.extend(temp)
        else:
            insertcurve.points.append(endpoints[i+1])

    return insertcurve



































