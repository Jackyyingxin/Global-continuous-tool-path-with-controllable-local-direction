from GeomBase import *
from Line import *
from GeomAIgo import *
from Polyline import *


class SweepLine:
    def __init__(self):
        self.segs=[]

    def intersect(self, y):
        ips= []
        yLine=Line(Point3D(0, y, self.segs[0].A.z), Vector3D(1, 0, 0))
        for seg in self.segs:
            if seg.A.y == y:

                ips.append(seg.A.clone())
            elif seg.B.y == y:
                ips.append(seg.B.clone())

            else:
                ip = intersect(yLine,seg)
                if ip is not  None:
                    ips.append(ip)
        ips.sort(key=lambda p: p.x) ## sort the intersect point via x coordinate of one sweep line
        i = len(ips)-1
        while i > 0:
            if ips[i].distanceSquare(ips[i-1]) == 0:
                del ips[i]
                del ips[i-1]
                i = i - 2
            else:
                i = i - 1
        return ips


def calcHatchPoints(polygons, ys):

    segs = []
    for poly in polygons:
        for i in range(poly.count()-1):
            seg = Segment(poly.point(i), poly.point(i+1))
            seg.yMin = min(seg.A.y, seg.B.y)
            seg.yMax = max(seg.A.y, seg.B.y)
            segs.append(seg)
    segs.sort(key=lambda seg: seg.yMin)
    k, sweep = 0, SweepLine()
    ipses = []
    ys.sort()
    for y in ys:
        for i in range(len(sweep.segs)-1, -1, - 1):
            if sweep.segs[i].yMax < y:
                del sweep.segs[i]
        for i in range(k, len(segs)):
            if segs[i].yMin < y and segs[i].yMax >= y:
                sweep.segs.append(segs[i])
            elif segs[i].yMin >= y:
                k = i
                break
        resultList = []
        for item in sweep.segs:
            if not item in resultList:
                resultList.append(item)
        sweep.segs = resultList
        if len(sweep.segs) > 0:
            ips = sweep.intersect(y)

            ipses.append(ips)
    return ipses ##二维列表，第一个维度表示扫描高度的序号，第二个维度表示某一扫描高度上所有的交点12

def genSweepHatches(polygons, interval, angle):
    mt = Matrix3D.createRotateMatrix('Z', -angle)
    mb = Matrix3D.createRotateMatrix('Z', angle)
    rotPolys = []
    for poly in polygons:
        rotPolys.append(poly.multiplied(mt))
    yMin, yMax = float('inf'), float('-inf')
    for poly in rotPolys:
        for pt in poly.points:
            yMin, yMax = min(yMin, pt.y),max(yMax, pt.y)
    ys = []
    y = yMin + interval
    while y < yMax:
        y += interval
    segs = genHatches(rotPolys, ys)
    for seg in segs:
        seg.multiply(mb)
    return segs

def genHatches(polygons, ys):
    segs = []
    ipses = calcHatchPoints(polygons, ys)
    for ips in ipses:
        for i in range(0, len(ips), 2):
            seg = Segment(ips[i], ips[i+1])
            segs.append(seg)
    return segs


##############################################################################################
#                               第一种情况 两列表余数为0

def genHatchesR0(polygons, ys,interval,angle):
    segs = []
    ipses = calcHatchPoints(polygons, ys)
    for j, ips in enumerate(ipses):
            for i in range(0, len(ips), 2):
                ips[i].x = ips[i].x # 连续zigzag改动，非连续路径删除这条
                seg = Segment(ips[i], ips[i + 1])
                segs.append(seg)
    return segs
def genHatchesL0(polygons, ys,interval,angle):  # 第一种情况 L列表余数为0
    segs = []
    ipses = calcHatchPoints(polygons, ys)
    for j, ips in enumerate(ipses):
        for i in range(0, len(ips), 2):
            if (j != 0 and j != len(ipses) - 1):
                ips[i+1].x = ips[i+1].x  # 连续zigzag改动，非连续路径删除这条
            seg = Segment(ips[i], ips[i + 1])
            segs.append(seg)
    return segs
################################################################################################
################################################################################################
#                               第二种情况 L列表余数为1 ,第六种情况  L列表余数为3，R列表余数为2
def genHatchesR1(polygons, ys,interval):
    segs = []
    ipses = calcHatchPoints(polygons, ys)
    for j, ips in enumerate(ipses):
            for i in range(0, len(ips), 2):
                ips[i].x = ips[i].x + interval  # 连续zigzag改动，非连续路径删除这条
                seg = Segment(ips[i], ips[i + 1])
                segs.append(seg)
    return segs
def genHatchesL1(polygons, ys,interval):
    segs = []
    ipses = calcHatchPoints(polygons, ys)
    for j, ips in enumerate(ipses):
        for i in range(0, len(ips), 2):
            if (j != 0 and j != len(ipses) - 2):
                ips[i + 1].x = ips[i + 1].x - interval  # 连续zigzag改动，非连续路径删除这条
            seg = Segment(ips[i], ips[i + 1])
            segs.append(seg)
    segs.pop()
    return segs
##################################################################################################
##################################################################################################
#                              第三种情况 L列表余数为1，R列表余数为1 ,第七种情况  L列表余数为3，R列表余数为3
def genHatchesR2(polygons, ys,interval,angle):
    segs = []
    ipses = calcHatchPoints(polygons, ys)
    for j, ips in enumerate(ipses):
        for i in range(0, len(ips), 2):
            if (j != len(ipses) - 1):
                ips[i].x = ips[i].x  # 连续zigzag改动，非连续路径删除这条
            seg = Segment(ips[i], ips[i + 1])
            segs.append(seg)
    return segs
def genHatchesL2(polygons, ys,interval,angle):
    segs = []
    ipses = calcHatchPoints(polygons, ys)
    for j, ips in enumerate(ipses):
        for i in range(0, len(ips), 2):
            if (j != 0):
                ips[i + 1].x = ips[i + 1].x # 连续zigzag改动，非连续路径删除这条
            seg = Segment(ips[i], ips[i + 1])
            segs.append(seg)
    return segs
#####################################################################################################
#####################################################################################################
#                            第四种情况 L列表余数为1，R列表余数为2，第八种情况  L列表余数为3，R列表余数为4
def genHatchesR3(polygons, ys,interval):
    segs = []
    ipses = calcHatchPoints(polygons, ys)
    for j, ips in enumerate(ipses):
        for i in range(0, len(ips), 2):
            if (j != len(ipses) - 2):
                ips[i].x = ips[i].x + interval  # 连续zigzag改动，非连续路径删除这条
            seg = Segment(ips[i], ips[i + 1])
            segs.append(seg)
    segs.pop()
    return segs
def genHatchesL3(polygons, ys,interval):
    segs = []
    ipses = calcHatchPoints(polygons, ys)
    for j, ips in enumerate(ipses):
        for i in range(0, len(ips), 2):
            if (j != 0):
                ips[i + 1].x = ips[i + 1].x - interval  # 连续zigzag改动，非连续路径删除这条
            seg = Segment(ips[i], ips[i + 1])
            segs.append(seg)
    return segs
#######################################################################################################
#######################################################################################################
#                                第五种情况 L列表余数为2，R列表余数为2
def genHatchesR4(polygons, ys,interval):
    segs = []
    ipses = calcHatchPoints(polygons, ys)
    for j, ips in enumerate(ipses):
            for i in range(0, len(ips), 2):
                ips[i].x = ips[i].x + interval  # 连续zigzag改动，非连续路径删除这条
                seg = Segment(ips[i], ips[i + 1])
                segs.append(seg)
    return segs
def genHatchesL4(polygons, ys,interval):
    segs = []
    ipses = calcHatchPoints(polygons, ys)
    for j, ips in enumerate(ipses):
        for i in range(0, len(ips), 2):
            if (j != 0 and j != len(ipses) - 1):
                ips[i + 1].x = ips[i + 1].x -interval  # 连续zigzag改动，非连续路径删除这条
            seg = Segment(ips[i], ips[i + 1])
            segs.append(seg)
    return segs
###########################################################################################################

















