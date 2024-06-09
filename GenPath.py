import numpy
import pyclipper

import Polyline
import utils
import GeomAIgo
from GeoError import Small
from Utility import *
from GenHatch import *


class GenDpPath:
    def __init__(self, polygons, interval, angle, ys):
        self.polygons, self.interval, self.angle = polygons, interval, angle
        self.splitPolys = []
        self.interval = interval
        self.ys1list = []
        self.startpoint = Point3D(float('-inf'), float('-inf'))
        self.endpoint = Point3D(float('-inf'), float('-inf'))
        self.ys = ys

    def generate2(self):
        paths = []
        a = -1
        for i in range(len(self.polygons)):
            poly = self.polygons[i]
            if poly.error_poly == 1:
                path = copy.deepcopy(poly)
                path.points.pop()
                paths.append(path)
                continue
            ys1 = utils.get_ys(poly, self.ys)
            ys1.sort()
            self.get_y_min_and_y_max(poly)
            L = self.get_left_right_offset_segments(poly, 0)
            R = self.get_left_right_offset_segments(poly, 1)
            L = self.line_offset(L)
            R = self.line_offset(R)
            L = self.reduce_repeated_points(L)
            R = self.reduce_repeated_points(R)
            remainder = len(ys1) % 8
            splitymax = float('-inf')
            spacing_increment = 0.0
            for point in poly.points:
                splitymax = max(splitymax, point.y)
            if len(ys1) >= 1 and (remainder == 1 or remainder == 3 or remainder == 5 or remainder == 7.):
                X_distance = {}
                for i in (0, -1):
                    ipses = calcHatchPoints([poly], [ys1[i]])
                    X_distance[i] = abs(ipses[0][1].x - ipses[0][0].x)
                X_distance = sorted(X_distance.items(), key=lambda x : x[1])
                ys1.pop(X_distance[0][0])
            ysleft = []
            ysright = []
            n = int(len(ys1) / 8)
            self.interval = self.interval + spacing_increment
            for i in range(0, n + 1):
                if 0 + 8 * i <= len(ys1) - 1:
                    ysleft.append(ys1[0 + 8 * i])
                if 3 + 8 * i <= len(ys1) - 1:
                    ysleft.append(ys1[3 + 8 * i])
                if 4 + 8 * i <= len(ys1) - 1:
                    ysleft.append(ys1[4 + 8 * i])
                if 7 + 8 * i <= len(ys1) - 1:
                    ysleft.append(ys1[7 + 8 * i])
                if 1 + 8 * i <= len(ys1) - 1:
                    ysright.append(ys1[1 + 8 * i])
                if 2 + 8 * i <= len(ys1) - 1:
                    ysright.append(ys1[2 + 8 * i])
                if 5 + 8 * i <= len(ys1) - 1:
                    ysright.append(ys1[5 + 8 * i])
                if 6 + 8 * i <= len(ys1) - 1:
                    ysright.append(ys1[6 + 8 * i])

            remainderL = (len(ysleft)) % 4
            remainderR = (len(ysright)) % 4

            if ((remainderL == 0 and remainderR == 0) or (remainderL == 2 and remainderR == 2)):  # 第一种情况
                # reverse list
                segsL = genHatchesL0([poly], ysleft, self.interval, self.angle)
                segsR = genHatchesR0([poly], ysright, self.interval, self.angle)
            if ((remainderL == 1 and remainderR == 1) or (remainderL == 3 and remainderR == 3)):
                segsL = genHatchesL2([poly], ysleft, self.interval, self.angle)
                segsR = genHatchesR2([poly], ysright, self.interval, self.angle)

            if len(segsL) > 0:
                pathL = self.linkLocalHatches(segsL, poly, L, R, ys1)# 得到两条路径再合在一起！！！
                #va = VtkAdaptor()
                #va.drawPolyline(poly).GetProperty().SetColor(1, 0, 1)
                #va.display()
                pathR = self.linkLocalHatches(segsR, poly, L, R, ys1)

                if (abs(pathL.points[len(pathL.points) - 1].x - segsL[len(segsL) - 1].B.x) <= 1e-12 and abs(
                        pathL.points[len(pathL.points) - 1].y - segsL[len(segsL) - 1].B.y) <= 1e-12):
                    PeekPoints = self.IsHavePeekR(pathR.points[len(pathR.points) - 1],
                                                  pathL.points[len(pathL.points) - 1], poly)
                    if PeekPoints != None:
                        for i in range(len(PeekPoints) - 1, -1, -1):
                            pathL.addPoint(PeekPoints[i])
                if (abs(pathL.points[len(pathL.points) - 1].x - segsL[len(segsL) - 1].A.x) <= 1e-12 and abs(
                        pathL.points[len(pathL.points) - 1].y - segsL[len(segsL) - 1].A.y) <= 1e-12):
                    PeekPoints = self.IsHavePeekL(pathL.points[len(pathL.points) - 1],
                                                  pathR.points[len(pathR.points) - 1], poly)
                    if PeekPoints != None:
                        for PeekPoint in PeekPoints:
                            pathL.addPoint(PeekPoint)
                PeekPoints1 = self.IsHavePeekR(pathL.points[0], pathR.points[0], poly)
                path = Polyline()

                for i in range(pathL.count()):
                    path.points.append(pathL.points[i])
                for i in range(pathR.count() - 1, -1, -1):
                    path.points.append(pathR.points[i])
                if PeekPoints1 != None:
                    for i in range(len(PeekPoints1) - 1, -1, -1):
                        path.addPoint(PeekPoints1[i])
                if (self.startpoint.y < pathL.points[-1].y):
                    self.startpoint = pathL.points[-1]
                    self.endpoint = pathR.points[-1]
                else:
                   pass
                path.belongtowhichpoly = a
                path.points.reverse()
                paths.append(path)
                for y in ys1:
                    if y not in self.ys1list:
                        self.ys1list.append(y)
                self.ys1list.sort()

            self.interval = self.interval - spacing_increment
        return GeomAIgo.rotatePolygons(paths, self.angle)

    def reduce_repeated_points(self,polygon):
        "这个函数改变了输入"
        temp = []
        for point in polygon.points:
            if point not in temp:
                temp.append(point)
        polygon.points = temp
        polygon.points.append(polygon.points[0])
        return polygon

    def genScanYs(self, polygons):
        yuzhi = 1e-9
        ys, yMin, yMax =[], float('inf'), float('-inf')
        for poly in polygons:
            for pt in poly.points:
             yMin, yMax = min(yMin, pt.y), max(yMax, pt.y)
        y = yMin + self.interval
        while yMax - y >= 1e-10:
            ys.append(y)
            y += self.interval
        return ys

    def get_y_min_and_y_max(self, poly):
        poly.y_min = float("inf")
        poly.y_max = float("-inf")
        for point in poly.points:
            if point.y <= poly.y_min:
                poly.y_min = point.y
            if point.y >= poly.y_max:
                poly.y_max = point.y


    def get_left_right_offset_segments(self, poly,L_R_flag = 0):
        y_min = []
        y_max = []
        for point in poly.points:
            if point.y == poly.y_min:
                y_min.append(point)
            if point.y == poly.y_max:
                y_max.append(point)
        y_min.sort(key=lambda x: x.x)
        y_max.sort(key=lambda x: x.x)
        temp = Polyline()
        original = copy.deepcopy(poly)
        if L_R_flag == 0:
            original.points.pop()
            original.points = utils.replaceL(utils.findindex(y_max[0], original), original.points)
            for i in range(len(original.points)):
                if original.points[i] != y_min[0]:
                    temp.points.append(original.points[i])
                else:
                    temp.points.append(original.points[i])
                    break
        else:
            original.points.pop()
            original.points = utils.replaceL(utils.findindex(y_min[-1], original), original.points)
            for i in range(len(original.points)):
                if original.points[i] != y_max[-1]:
                    temp.points.append(original.points[i])
                else:
                    temp.points.append(original.points[i])
                    break

        return temp


    def caculatetheta(self,P2:Point3D,P1:Point3D):#P2 默认在P1上面
        if(abs(P2.x - P1.x) >= 1e-6):
            theta = math.atan((P2.y - P1.y) / (P2.x - P1.x))
        else:
            theta=math.pi * 0.5

        return theta
    def peekpointsoffsetR(self,PeekPoint,theta1,theta2):
        deg1 = radToDeg(theta1)
        deg2 = radToDeg(theta2)
        if(theta1<0 and theta2>0):
            theta1=abs(theta1)

        elif(theta1<0 and theta2<0):
            theta1=theta1+math.pi
            theta2=abs(theta2)
            deg1=radToDeg(theta1)
            deg2=radToDeg(theta2)
        elif(theta1>0 and theta2>0):
            theta1=math.pi-theta1
            deg1 = radToDeg(theta1)
            deg2 = radToDeg(theta2)
        elif(theta1>0 and theta2<0):
            theta2=abs(theta2)
            deg1 = radToDeg(theta1)
            deg2 = radToDeg(theta2)

        degtotal=radToDeg((theta1 + theta2) * 0.5)
        PeekPoint.x = PeekPoint.x - (
                self.interval / abs(math.sin((theta1 + theta2) * 0.5)))

        return PeekPoint.x
    def peekpointsoffsetL(self,PeekPoint,theta1,theta2):
        deg1 = radToDeg(theta1)
        deg2 = radToDeg(theta2)
        if(theta1<0 and theta2>0):
            theta1=abs(theta1)

        elif(theta1<0 and theta2<0):
            theta1=theta1+math.pi
            theta2=abs(theta2)
            deg1=radToDeg(theta1)
            deg2=radToDeg(theta2)
        elif(theta1>0 and theta2>0):
            theta1=math.pi-theta1
            deg1 = radToDeg(theta1)
            deg2 = radToDeg(theta2)
        elif(theta1>0 and theta2<0):
            theta2=abs(theta2)
            deg1 = radToDeg(theta1)
            deg2 = radToDeg(theta2)

        degtotal = radToDeg((theta1 + theta2) * 0.5)
        PeekPoint.x = PeekPoint.x + (
                self.interval / abs(math.sin((theta1 + theta2) * 0.5)))

        return PeekPoint.x

    def linternaloffset(self, theta1, P1):
        if (theta1 != math.pi * 0.5):
            P1.x = P1.x + (self.interval / abs(math.sin(theta1)))
        else:
            P1.x = P1.x + self.interval
        return P1.x
    def rinternaloffset(self,theta1,P1):
        if (theta1 != math.pi * 0.5):
            P1.x = P1.x - (self.interval / abs(math.sin(theta1)))
        else:
            P1.x = P1.x - self.interval
        return P1.x

    def peektheta(self,PeekPoints2):
        thetai2 = []
        if (len(PeekPoints2) == 1):
            return thetai2
        else:
            for i in range(len(PeekPoints2)):
                if (i != len(PeekPoints2) - 1):
                    thetai = self.caculatetheta(PeekPoints2[i + 1], PeekPoints2[i])
                    thetai2.append(thetai)
            return thetai2


    def linkLocalHatches(self, segs, polys, L,R,ys1): ###判断最后一条线是左线还是右线
        poly = Polyline()

        for i, seg in enumerate(segs):
            poly.addPoint(seg.B if (i % 2 == 0)else seg.A)
            poly.addPoint(seg.A if(i % 2 == 0)else seg.B)#LINK LOCAK

            if i != len(segs) - 1:
                if i % 2 == 0:
                    if abs(segs[i + 1].A.y - poly.points[poly.count() - 1].y - self.interval) >= 1e-5:
                        PeekPoints = self.IsHavePeekL(poly.points[poly.count() - 1], segs[i + 1].A, polys)
                    else:
                        segs[i + 1].A.x = segs[i + 1].A.x
                        poly.points[poly.count() - 1].x = poly.points[poly.count() - 1].x

                        ips1 = calcHatchPoints([L], [poly.points[poly.count() - 1].y])
                        ips2 = calcHatchPoints([L], [segs[i+1].A.y])
                        ips2[0].sort(key=lambda x: x.x)
                        ips1[0].sort(key=lambda x: x.x)
                        segs[i + 1].A.x = ips2[0][-1].x
                        segs[i + 1].A.y = ips2[0][-1].y
                        poly.points[poly.count() - 1].x = ips1[0][-1].x
                        poly.points[poly.count() - 1].y = ips1[0][-1].y
                        PeekPoints = self.IsHavePeekR(poly.points[poly.count() - 1], segs[i + 1].A, L)

                else:
                    if abs(segs[i + 1].B.y - poly.points[poly.count() - 1].y - self.interval) >= 1e-5:
                        PeekPoints = self.IsHavePeekR(poly.points[poly.count() - 1], segs[i + 1].B, polys)
                    else:
                        segs[i + 1].B.x = segs[i + 1].B.x
                        poly.points[poly.count() - 1].x = poly.points[poly.count() - 1].x
                        ips3 = calcHatchPoints([R], [poly.points[poly.count() - 1].y])
                        ips4 = calcHatchPoints([R], [segs[i + 1].B.y])
                        ips4[0].sort(key=lambda x: x.x)
                        ips3[0].sort(key=lambda x: x.x)
                        segs[i + 1].B.x = ips4[0][0].x
                        segs[i + 1].B.y = ips4[0][0].y
                        poly.points[poly.count() - 1].x = ips3[0][0].x
                        poly.points[poly.count() - 1].y = ips3[0][0].y
                        PeekPoints = self.IsHavePeekL(poly.points[poly.count() - 1], segs[i + 1].B, R)
                if PeekPoints != None and len(PeekPoints)!=0:
                    for PeekPoint in PeekPoints:
                        poly.addPoint(PeekPoint)
        return poly

    def check_remove(self, poly_list:list):
        pass

    def line_offset(self, lines, dig=7):
        cofficient = math.pow(10, dig)
        contourtuple = []
        for point in lines.points:
            var_tuple = (round(point.x * cofficient), round(point.y * cofficient))
            contourtuple.append(var_tuple)
        contourtuple = tuple(contourtuple)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(contourtuple, pyclipper.JT_SQUARE, pyclipper.ET_OPENBUTT)
        solution = pco.Execute(self.interval * cofficient)
        # if len(solution)==0:
        #   return solution
        polys1 = Polyline()
        for i in range(len(solution)):
            solution[i] = numpy.array(solution[i]) / cofficient
            solution[i] = solution[i].tolist()

        if len(solution) == 0:

            raise Small("the lines error")
        for j in range(len(solution[0])):
            point = Point3D(solution[0][j][0], solution[0][j][1], lines.points[0].z)
            polys1.points.append(point)
        polys1.points.append(polys1.points[0])

        return polys1

    def Re_big(self, poly, interval, dig = 7):
        cofficient = math.pow(10, 7)
        contourtuple = []
        for point in poly.points:
            var_tuple = (round(point.x * cofficient), round(point.y * cofficient))
            contourtuple.append(var_tuple)
        contourtuple = tuple(contourtuple)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(contourtuple, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)
        solution = pco.Execute(interval * cofficient)
        polys1 = Polyline()
        for i in range(len(solution)):
            solution[i] = numpy.array(solution[i]) / cofficient
            solution[i] = solution[i].tolist()
        if len(solution) == 0:
            raise Small("big wrong")
        for j in range(len(solution[0])):
            point = Point3D(solution[0][j][0], solution[0][j][1], poly.points[0].z)
            polys1.points.append(point)
        polys1.points.append(polys1.points[0])

        return polys1

    def contouroffset(self, splitpoly:Polyline):
        cofficient = 10000
        contourtuple = []
        for point in splitpoly.points:
            var_tuple = (round(point.x * cofficient), round(point.y * cofficient))
            contourtuple.append(var_tuple)
        contourtuple = tuple(contourtuple)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(contourtuple, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solution = pco.Execute(-self.interval * cofficient)
        polys1 = Polyline()
        polylist = []
        for i in range(len(solution)):
            solution[i] = numpy.array(solution[i])/cofficient
            solution[i] = solution[i].tolist()
        if len(solution) == 0:
            raise Small("the single poly too narrow can not offset one time")
        for i in range(len(solution)):
            polys1 = Polyline()
            for j in range(len(solution[i])):
                point = Point3D(solution[i][j][0], solution[i][j][1],  splitpoly.points[0].z)
                polys1.points.append(point)
            polys1.points.append(polys1.points[0])
            polylist.append(polys1)
        disdir = {}
        if len(solution) > 1:

                for poly in polylist:
                    disdir[poly.centroid().distance(splitpoly.centroid())] = poly
                tmp = sorted(disdir.items(), key=lambda item: item[0])
                disdir = dict(tmp)
                return disdir[list(disdir.keys())[0]]

        return polys1

    def IsHavePeekL(self, prep, nextp, polys: Polyline,flag = 0):
        temp_poly = copy.deepcopy(polys)
        peekpointL = []
        temp_poly.points.reverse()
        x3 = prep.x
        y3 = prep.y
        x4 = nextp.x
        y4 = nextp.y
        min_sp = float("inf")
        min_sn = float("inf")
        start_point = Point3D(float("inf"), float("inf"))
        end_point = None
        for i in range(len(temp_poly.points)-1):
            x1 = temp_poly.points[i].x
            y1 = temp_poly.points[i].y
            x2 = temp_poly.points[i + 1].x
            y2 = temp_poly.points[i + 1].y
            if (x1 == x3 and y1 == y3) or (x2 == x3 and y2 == y3):
                print("error pre")
            if (x1 == x4 and y1 == y4) or (x2 == x4 and y2 == y4):
                print("error nex")
            sprep = 0.5 * (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2)
            snextp = 0.5 * (x1 * y2 + x2 * y4 + x4 * y1 - x1 * y4 - x2 * y1 - x4 * y2)

            if abs(sprep) <= 1e-3:
                if ((min(x1, x2) < x3 or abs(min(x1, x2) - x3) <= 1e-6) and (
                        max(x1, x2) > x3 or abs(max(x1, x2) - x3) <= 1e-6)) and (
                        (min(y1, y2) < y3 or abs(min(y1, y2) - y3) <= 1e-6) and (
                        max(y1, y2) > y3 or abs(max(y1, y2) - y3) <= 1e-6)):
                    if min_sp > abs(sprep):
                        min_sp = abs(sprep)
                        start_point = temp_poly.points[i] if temp_poly.points[i].y < temp_poly.points[i + 1].y else \
                            temp_poly.points[i + 1]

            if abs(snextp) <= 1e-3:
                if ((min(x1, x2) < x4 or abs(min(x1, x2) - x4) <= 1e-6) and (
                        max(x1, x2) > x4 or abs(max(x1, x2) - x4) <= 1e-6)) and (
                        (min(y1, y2) < y4 or abs(min(y1, y2) - y4) <= 1e-6) and (
                        max(y1, y2) > y4 or abs(max(y1, y2) - y4) <= 1e-6)):
                    if min_sn > abs(snextp):
                        min_sn = abs(snextp)
                        end_point = temp_poly.points[i] if temp_poly.points[i].y > temp_poly.points[i + 1].y else \
                        temp_poly.points[i + 1]
        temp_poly.points.pop()
        temp_poly.points = utils.replaceL(utils.findindex(start_point, temp_poly), temp_poly.points)
        end_index = utils.findindex(end_point, temp_poly)
        if flag == 0:
            for i in range(1, end_index):
                peekpointL.append(temp_poly.points[i])

        else:
            for i in range(0, end_index+1):
                peekpointL.append(temp_poly.points[i])

        return peekpointL

    def IsHavePeekR(self, prep, nextp, polys, flag=0):
        temp_poly = copy.deepcopy(polys)
        peekpointR = []
        x3 = prep.x
        y3 = prep.y
        x4 = nextp.x
        y4 = nextp.y
        min_sp = float("inf")
        min_sn = float("inf")
        start_point = None
        end_point = None
        for i in range(len(temp_poly.points) - 1):
            x1 = temp_poly.points[i].x
            y1 = temp_poly.points[i].y
            x2 = temp_poly.points[i + 1].x
            y2 = temp_poly.points[i + 1].y
            if (x1 == x3 and y1 == y3) or (x2 == x3 and y2 == y3):
                print("error pre")
            if (x1 == x4 and y1 == y4) or (x2 == x4 and y2 == y4):
                print("error nex")
            sprep = 0.5 * (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2)
            snextp = 0.5 * (x1 * y2 + x2 * y4 + x4 * y1 - x1 * y4 - x2 * y1 - x4 * y2)

            if abs(sprep) <= 1e-3:
                if ((min(x1, x2) < x3 or abs(min(x1, x2) - x3) <= 1e-6) and (
                        max(x1, x2) > x3 or abs(max(x1, x2) - x3) <= 1e-6)) and (
                        (min(y1, y2) < y3 or abs(min(y1, y2) - y3) <= 1e-6) and (
                        max(y1, y2) > y3 or abs(max(y1, y2) - y3) <= 1e-6)):
                    if min_sp > abs(sprep):
                        min_sp = abs(sprep)
                        start_point = temp_poly.points[i] if temp_poly.points[i].y < temp_poly.points[i + 1].y else \
                            temp_poly.points[i + 1]
            if abs(snextp) <= 1e-3:
                if ((min(x1, x2) < x4 or abs(min(x1, x2) - x4) <= 1e-6) and (
                        max(x1, x2) > x4 or abs(max(x1, x2) - x4) <= 1e-6)) and (
                        (min(y1, y2) < y4 or abs(min(y1, y2) - y4) <= 1e-6) and (
                        max(y1, y2) > y4 or abs(max(y1, y2) - y4) <= 1e-6)):
                    if min_sn > abs(snextp):
                        min_sn = abs(snextp)
                        end_point = temp_poly.points[i] if temp_poly.points[i].y > temp_poly.points[i + 1].y else \
                        temp_poly.points[i + 1]
        temp_poly.points.pop()
        temp_poly.points = utils.replaceL(utils.findindex(start_point, temp_poly),
                                                       temp_poly.points)
        end_index = utils.findindex(end_point, temp_poly)
        if flag == 0:
            for i in range(1, end_index):
                peekpointR.append(temp_poly.points[i])
        else:
            for i in range(0, end_index+1):
                peekpointR.append(temp_poly.points[i])

        return peekpointR


def caculate_offset_point_l(point, PeekPoints, interval, flag):
    if flag == 1:
        vector_f = PeekPoints[0].pointTo(PeekPoints[1])
    else:
        vector_f = PeekPoints[-2].pointTo(PeekPoints[-1])

    normal_v = Vector3D(vector_f.dy, -vector_f.dx)
    if vector_f.crossProduct(normal_v).normalized().dz > 0:
        normal_v = Vector3D(-vector_f.dy, vector_f.dx)
    theta = vector_f.getAngle2D()
    offset = 1 / (math.sin(theta) / interval)
    if normal_v.dotProduct(Vector3D(1,0,0))>0:
        point.x = point.x + offset
    else:
        point.x = point.x - offset


def caculate_offset_peek(PeekPoints, interval):
    offset_peek = []
    for i in range(len(PeekPoints)-2):
        pre = PeekPoints[i].pointTo(PeekPoints[i+1])
        next = PeekPoints[i+1].pointTo(PeekPoints[i+2])
        angle = pre.getAngle(next)
        angle = math.pi - angle
        length = 1/(math.sin((angle/2))/interval)
        vector1 = PeekPoints[i].pointTo(PeekPoints[i+1]).normalized()
        vector2 = PeekPoints[i+2].pointTo(PeekPoints[i+1]).normalized()
        direction = vector1 + vector2
        direction = direction.normalized()
        offset_peek.append(PeekPoints[i+1] + direction.amplified(length))
    return offset_peek


def caculate_offset_point_r(point, PeekPoints, interval, flag):

    if flag == 1:
        vector_f = PeekPoints[0].pointTo(PeekPoints[1])
    else:
        vector_f = PeekPoints[-2].pointTo(PeekPoints[-1])
    normal_v = Vector3D(vector_f.dy, -vector_f.dx)
    if vector_f.crossProduct(normal_v).normalized().dz < 0:
        normal_v = Vector3D(-vector_f.dy, vector_f.dx)
    theta = vector_f.getAngle2D()
    offset = 1 / (math.sin(theta) / interval)
    if normal_v.dotProduct(Vector3D(1, 0, 0)) > 0:
        point.x = point.x + offset
    else:
        point.x = point.x - offset


def genDpPath2(polygons, interval, angle, ys):

    return GenDpPath(polygons, interval, angle, ys).generate2()