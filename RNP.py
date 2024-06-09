import math
import GenDpPath5
import build_connect_tree
from Utility import *

class DNP:

    def __init__(self, polyline, interval, filter_angle, flag=0):
        self.polyline = polyline
        self.angle = 0
        self.DNP = GenDpPath5.Polyline()
        self.sp = polyline.points[0]
        self.k = 0
        self.interval = interval
        self.filter_angle = filter_angle
        self.flag = flag

    def get_angle(self, vector1, vector2):
        dot = vector1.dotProduct(vector2)
        v1l = vector1.length()
        v2l = vector2.length()
        temp = dot/(v1l * v2l)
        if temp > 1 or temp < -1:
            if temp < -1:
                temp = -1
            if temp > 1:
                temp = 1
        angle = math.acos(temp)  # angle is arc
        return angle

    def dnp(self):
        self.DNP = GenDpPath5.Polyline()
        self.DNP.points.append(self.polyline.points[0])
        pre = 1
        index = 1
        while index <= len(self.polyline.points)-2:
            vector1 = self.polyline.points[pre-1].pointTo(self.polyline.points[pre])
            vector2 = self.polyline.points[index].pointTo(self.polyline.points[index+1])
            angle = self.get_angle(vector1, vector2)
            radToDeg(angle)
            if self.flag == 0:
                if radToDeg(angle) >= self.angle:
                    self.DNP.points.append(self.polyline.points[index])

                    if self.DNP.points[-1].distance(
                            self.DNP.points[-2]) < self.interval and self.angle < self.filter_angle:
                        self.DNP.points.pop()
                        self.angle = self.angle + 1
                    else:
                        pre = index + 1
                        index = index + 1
                        self.angle = 0

                else:
                    index = index + 1

            elif self.flag == 1:
                if radToDeg(angle) >= self.filter_angle:
                    self.DNP.points.append(self.polyline.points[index])
                    pre = index + 1
                    index = index + 1
                else:
                    index = index + 1

            elif self.flag == 2:
                if vector2.dy == 0:
                    self.DNP.points.append(self.polyline.points[index])
                    pre = index + 1
                    index = index + 1
                    build_connect_tree.path_show([self.DNP])
                else:
                    if radToDeg(angle) >= self.angle:
                        self.DNP.points.append(self.polyline.points[index])

                        if self.DNP.points[-1].distance(
                                self.DNP.points[-2]) < self.interval and self.angle < self.filter_angle:
                            self.DNP.points.pop()
                            self.angle = self.angle + 1
                        else:
                            pre = index + 1
                            index = index + 1
                            self.angle = 0

                    else:
                        index = index + 1
        if self.polyline.points[-1].distance(self.DNP.points[-1]) > self.interval:
                self.DNP.points.append(self.polyline.points[-1])
        return self.DNP














