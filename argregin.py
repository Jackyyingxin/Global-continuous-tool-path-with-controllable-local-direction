import itertools
import numpy as np
import pyclipper
from pyclipper import Pyclipper
import VtkAdaptor
import utils
from GenHatch import *
from ClipperAdaptor import *
import GeoError


class ArgRegion:
    def __init__(self, polygons, interval, coefficient, delta, adjustPolyDirs=False):

        self.interval = interval
        self.polygons = polygons
        self.delta = delta
        self.ys = self.To_Intersect_ys()
        self.C_regions = []
        self.re_check_pt = []
        self.coefficient = coefficient
        if adjustPolyDirs:
            adjustPolygonDirs(self.polygons)

    def To_Intersect_ys(self):  # Get the scan_lines of the polygon
        ys, yMin, yMax = [], float('inf'), float('-inf')
        for poly in self.polygons:
            for pt in poly.points:
                yMin, yMax = min(yMin, pt.y), max(yMax, pt.y)
        y = yMin + self.interval
        while yMax - y >= 1e-10:
            ys.append(y)
            y += self.interval

        ys.sort(reverse=True)
        return ys

    def findTurnPoints(self):  # Get the turnVertexes of the polygon
        vx = Vector3D(1, 0, 0)
        turnPts = []
        for poly in self.polygons:
            for i in range(poly.count() - 1):
                pts = poly.points
                v1 = pts[-2 if (i == 0) else (i - 1)].pointTo(pts[i])
                v2 = pts[i].pointTo(pts[i + 1])
                if v1.crossProduct(vx).dz * v2.crossProduct(vx).dz <= 0:
                    if v1.crossProduct(v2).dz < 0:
                        turnPts.append(pts[i])
        return turnPts

    def check_inner(self, region):  # 检测分区后是否还存在孔洞
        for poly in region:
            if utils.direnction(poly) == 1:
                return False
        return True

    def check_inner_count(self, region):
        count = 0
        for poly in region:
            if utils.direnction(poly) == 1:
                count = count + 1
        return count

    def polygon_decomposition(self, select_turnPts):
        ys = []
        for pt in select_turnPts:
            ys.append(pt.y)
        ys.sort()
        select_turnPts.sort(key=lambda x: x.y)
        hatchPtses = calcHatchPoints(self.polygons, ys)
        result_clipper = []
        C_polygons = self.decomposition(self.polygons, select_turnPts, hatchPtses, result_clipper)
        # utils.path_show(C_polygons[0])
        for polygon in C_polygons:
            self.C_regions.append(polygon[0])
        # utils.path_show(self.C_regions)
        error, error_list, modify = self.unprintable(self.C_regions)
        for j in modify:
            self.C_regions[j].error_poly = 1
        return self.C_regions

    def check_self_intersection(self, pt, polygon, offset_regions: list):
        "return false means self-intersection exist"
        V = VtkAdaptor.VtkAdaptor()
        r_offset_regions = []
        for polygon in offset_regions:
            r_offset_regions.append(polygon[0])
        # Get two scan_lines
        y2 = sorted(self.ys,  key = lambda x: abs(x - pt.y))[:2]
        ipses = calcHatchPoints(r_offset_regions, y2)
        original_ipses = calcHatchPoints(polygon, y2)
        for i in range(len(ipses)):
            if len(ipses[i]) > 2 and len(ipses[i]) > len(original_ipses[i]):
                for ip in ipses[i]:
                    V.drawPoint(ip).GetProperty().SetColor(1, 0, 0)
                for polygon in offset_regions:
                    V.drawPolyline(polygon[0]).GetProperty().SetColor(0, 0, 1)
                V.display()
                return False
        return True

    def arg_split(self, turnPts):

        child_regions = self.polygon_decomposition(turnPts)

        return child_regions

    def findLRPoints(self, pt, ptses):
        for pts in ptses:
            if len(pts) > 0 and pts[0].y == pt.y:
                for i in range(len(pts) - 1):
                    if pt.x > pts[i].x and pt.x < pts[i + 1].x:
                        return pts[i], pts[i + 1]
        return None, None

    def createSplitter(self, p1, p2):  #
        vx, vy = Vector3D(1, 0, 0), Vector3D(0, 1, 0)
        splitter = Polyline()
        splitter.addPoint(p1 - vx.amplified(self.delta) - vy.amplified(self.delta))
        splitter.addPoint(p2 + vx.amplified(self.delta) - vy.amplified(self.delta))
        splitter.addPoint(p2 + vx.amplified(self.delta) + vy.amplified(self.delta))
        splitter.addPoint(p1 - vx.amplified(self.delta) + vy.amplified(self.delta))
        splitter.addPoint(splitter.startPoint())
        return splitter

    def split(self, turnPts):
        if len(turnPts) != 0:
            ys = []
            for pt in turnPts:
                ys.append(pt.y)
            ys.sort()
            hatchPtses = calcHatchPoints(self.polygons, ys)
            splitters = []
            for i in range(len(turnPts)):
                lPt, rPt = self.findLRPoints(turnPts[i], hatchPtses)
                if lPt is not None and rPt is not None:
                    splitter = self.createSplitter(lPt, rPt)
                    splitters.append(splitter)
            if len(splitters) != 0:
                clipper, ca = Pyclipper(), ClipperAdaptor()
                clipper.AddPaths(ca.toPaths(self.polygons), pyclipper.PT_SUBJECT)  ##有改动
                clipper.AddPaths(ca.toPaths(splitters), pyclipper.PT_CLIP)
                sln = clipper.Execute(pyclipper.CT_DIFFERENCE)
                a = ca.toPolys(sln, turnPts[0].z)
                return a
        return self.polygons

    def get_leafNode(self, head, leaf):
        if len(head.children) == 0:
            leaf.append(int(head.name))
            return leaf

        for child in head.children:
            self.get_leafNode(child, leaf)
        return leaf

    @staticmethod
    def combine(arr, result, index, n, arr_len, total: list, start=0):
        ct = 0
        # i = 0
        for ct in range(start, arr_len - index + 1, 1):
            result[index - 1] = ct
            if index - 1 == 0:
                temp = []
                for j in range(n - 1, -1, -1):
                    temp.append(arr[result[j]])
                total.append(temp)
            else:
                # print(i+1)
                ArgRegion.combine(arr, result, index - 1, n, arr_len, total, ct + 1)
        return total

    def Split(self):
        turnpts = self.findTurnPoints()
        print(len(turnpts))
        if len(self.ys) < 2:
            raise GeoError.Small("{}".format("fill_area is too small to print"))
        if len(turnpts) == 0 and len(self.ys) >= 2:
            print("no turn vertexes")
            return self.polygons

        a = self.arg_split(turnpts)
        return a

    def printable(self):
        Strange_ips = {}
        for i in range(len(self.polygons)):  # self.polygons的顺序不可以调整
            Strange_ips[i] = []
            ipses = calcHatchPoints(self.polygons[i], self.ys)
            for j in range(len(ipses)):
                if len(ipses[j]) != 0 and len(ipses[j]) != 2:
                    if len(ipses[j]) % 2 == 1:  # 出现奇数交点可能遇到特殊情况了
                        Strange_ips[i].append(ipses[j][0].y)
                        print("occur strange point")
                        for y in self.ys:
                            if y == ipses[j][0].y:
                                y = y + 0.01
                                if calcHatchPoints(self.polygons[i], [y]) != 2:
                                    y = y - 0.02
                    else:
                        print("error")
                        return False
            return True

    def unprintable(self, polygons):
        Strange_ips = {}
        error = []
        error_list = []
        modify = []
        for i in range(len(polygons)):  # the order of self.polygons can not change
            Strange_ips[i] = []
            # calculate the intersection point between child_region and scan_lines
            ipses = calcHatchPoints([polygons[i]], self.ys)
            # the situation that the child_region has only one or zero scan_line
            if len(ipses) == 0 or len(ipses) == 1:
                modify.append(i)
                continue
            for j in range(len(ipses)):
                if len(ipses[j]) != 0 and len(ipses[j]) != 2:
                    if len(ipses[j]) % 2 == 1:
                        Strange_ips[i].append(ipses[j][0].y)
                        print("one of the scan lines passes through a point of the child_region")
                        error.append(polygons[i])
                        error_list.append(i)
                        break
                    else:
                        error.append(polygons[i])
                        error_list.append(i)
                        break
        return error, error_list, modify

    def get_father(self, poly):
        ipses = calcHatchPoints([poly], self.ys)
        if len(ipses) == 0 or len(ipses) == 1:
            return False
        for j in range(len(ipses)):
            if len(ipses[j]) != 2:
                return False
        return True

    def add(self, x, y):  # x为初始二进制数，y为每次更新加几，实现左右不同方向分区
        p0c0 = 0, {}
        p1c0 = 1, {}
        p0c1 = 0, {}
        p1c1 = 1, {}

        # 各进程状态之间的转换
        p0c0[1].update({(0, 0): p0c0, (1, 0): p1c0,
                        (0, 1): p1c0, (1, 1): p0c1})
        p1c0[1].update({(0, 0): p0c0, (1, 0): p1c0,
                        (0, 1): p1c0, (1, 1): p0c1})
        p0c1[1].update({(0, 0): p1c0, (1, 0): p0c1,
                        (0, 1): p0c1, (1, 1): p1c1})
        p1c1[1].update({(0, 0): p1c0, (1, 0): p0c1,
                        (0, 1): p0c1, (1, 1): p1c1})

        x = map(int, reversed(x))
        y = map(int, reversed(y))
        z = []
        # 模拟自动机
        value, transition = p0c0
        for r, s in itertools.zip_longest(x, y, fillvalue=0):
            value, transition = transition[r, s]
            z.append(value)

        z.append(transition[0, 0][0])
        z.reverse()
        return z

    def checkminpoly(self, polygons):
        test = {}
        yMin, yMax = float('inf'), float('-inf')
        for poly in polygons:
            for pt in poly.points:
                yMin, yMax = min(yMin, pt.y), max(yMax, pt.y)
            dis = yMax - yMin
            test[poly] = dis
        y = sorted(test.items(), key=lambda p: p[1])
        return y

    def list_to_Str(self, z: list):
        result = [str(int(i)) for i in z]

        return result

    def Check_split(self, splitters):  # 检测分区块是否重叠
        temp_split = copy.deepcopy(splitters)
        clipper, ca = Pyclipper(), ClipperAdaptor()  ##有改动
        for i in range(len(splitters) - 1):
            clipper.AddPath(ca.toPath(temp_split[0]), pyclipper.PT_SUBJECT)
            temp_split.remove(temp_split[0])
            clipper.AddPaths(ca.toPaths(temp_split), pyclipper.PT_CLIP)
            solution = clipper.Execute(pyclipper.CT_INTERSECTION)
            if len(solution) != 0:
                print("this spliter is wrong")
                return False
            clipper, ca = Pyclipper(), ClipperAdaptor()

    def deletable(self, min_Poly):
        pass

    def connect_matrix(self, Curve):
        distancematrix = np.ndarray((len(Curve), len(Curve)))
        for i in range(len(Curve)):
            for j in range(len(Curve)):
                if i != j:
                    distancematrix[i][j] = self.Adj(Curve[i], Curve[j])
                else:
                    distancematrix[i][j] = 0
        return distancematrix


    def correct_poly(self, polys):  # 找到正确的轮廓作为父节点
        for i in range(len(polys)):
            if self.get_father(polys[i]) == True:
                return i

        return "false"



    def get_ys(self, poly):  # 在总ys中抽取每个轮廓中的扫描线
        yMin, yMax = float('inf'), float('-inf')
        for pt in poly.points:
            yMin, yMax = min(yMin, pt.y), max(yMax, pt.y)
        front = -1
        behind = -1
        for i in range(len(self.ys)):
            if yMax - self.ys[i] >= 1e-10:
                front = i
                break
        if front != -1:
            for j in range(len(self.ys) - 1, front, -1):  # 从len(total_ys)-1开始到front为止，不包括front的整数。
                if self.ys[j] - yMin >= 1e-10:
                    behind = j
                    break
            if behind == -1:
                local_ys = self.ys[front]
                return local_ys
        else:
            local_ys = []
            return local_ys
        local_ys = self.ys[front:behind + 1]
        return local_ys

    def calcHatchPoints(self, polygons, ys):
        segs = []
        for poly in polygons:
            for i in range(poly.count() - 1):
                seg = Segment(poly.point(i), poly.point(i + 1))
                seg.yMin = min(seg.A.y, seg.B.y)
                seg.yMax = max(seg.A.y, seg.B.y)
                segs.append(seg)
        segs.sort(key=lambda seg: seg.yMin)
        k, sweep = 0, SweepLine()
        ipses = {}
        # ipi = {}
        ys.sort()
        for y in ys:
            for i in range(len(sweep.segs) - 1, -1, - 1):
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
                # if len(ips)>2:
                #    ips.sort(key=lambda x:x.x)
                #    ips=[ips[0],ips[-1]]
                # if len(ips)<2:
                #    ips.sort(key=lambda x:x.x)
                #    ips=[ips[0],ips[-1]]

                if y not in ipses.keys():
                    ipses[y] = []
                ipses[y].append(ips)
        flag = 0
        return ipses

    def polygons_decomposition(self, turnpts):
        ca = ClipperAdaptor()
        pco = pyclipper.PyclipperOffset()
        pco.AddPaths(ca.toPaths(self.polygons), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solution = pco.Execute(0)
        polys = ca.toPolys(solution)
        turnpts = self.find_Single_TurnPoints(polys)
        result_clipper = []
        ys = []
        for pt in turnpts:
            ys.append(pt.y)
        ys.sort()
        turnpts.sort(key=lambda x: x.y)
        "对拐点从小到大排序"
        hatchPtses = calcHatchPoints(polys, ys)
        # self.C_tree.add(polygons[0])
        return self.decomposition(polys, turnpts, hatchPtses, result_clipper)

    def decomposition(self, polygon, turnpts, hatchPtses, result_clipper):

        queue = [polygon]
        result = []

        while 1:
            temp = copy.deepcopy(queue)
            # va = VtkAdaptor()
            # va1 = VtkAdaptor()
            # for polygon in queue:
            #    for poly in polygon:
            #        tem = va.drawPolyline(poly)
            #        tem.GetProperty().SetColor(0, 0, 1)
            #        tem.GetProperty().SetLineWidth(2)
            # if len(turnpts) > 0:
            #    va.drawPoint(turnpts[0]).GetProperty().SetColor(1, 0, 0)
            # va.display()
            for polygon in temp:
                if self.get_regions(polygon) == 1:
                    if polygon not in result:
                        result.append(polygon)
                    queue.remove(polygon)
            # TEMP = []
            # if len(result) != 0:
            #    for polygon in result:
            #        if polygon not in TEMP:
            #            for poly in polygon:
            #                temp = va1.drawPolyline(poly)
            #                temp.GetProperty().SetColor(1, 0, 0)
            #                temp.GetProperty().SetLineWidth(3)
            #            TEMP.append(polygon)
            #    va1.display()

            if len(queue) == 0:
                break
                # if len(turnpts) == 0:
                # va = VtkAdaptor()
                # for polygon in queue:
                #        for poly in polygon:
                #            temp = va.drawPolyline(poly)
                #            temp.GetProperty().SetColor(0, 0, 1)
                #            temp.GetProperty().SetLineWidth(1)
                # va.display()

            #    temp = copy.deepcopy(queue)
            #    for polygon in temp:
            #        if self.get_regions(polygon) == 1:
            #            if polygon not in result:
            #                result.append(polygon)

            #            queue.remove(polygon)
            #    if len(queue)!=0:
            #            turnpts = self.find_Single_TurnPoints(queue[0])
            #            turnpts.sort(key=lambda x: x.y)

            #    else:
            #        break

            if len(turnpts) == 0:
                raise GeoError.Decomposition("误差分解出错，跳过轮廓")
            pt = turnpts[0]
            polygon = self.polygon_turnpts(queue, pt)

            if polygon is None:
                turnpts.remove(pt)
                continue
            try:
                L_Polys, L_splitter = self.get_L_R_poly(pt, hatchPtses, 0, polygon)
                R_Polys, R_splitter = self.get_L_R_poly(pt, hatchPtses, 1, polygon)
            except GeoError.Turnpt_E:
                print("has only one lpt or rpt")
                turnpts.remove(pt)
                queue.insert(0, polygon)
                continue

            # build_connect_tree.path_show(L_Polys)
            # build_connect_tree.path_show(R_Polys)
            # TODO get polygons may makes the turnpt can not find child_region
            L_polygons = self.get_polygons(L_Polys)
            R_polygons = self.get_polygons(R_Polys)
            l_min_s = float("inf")
            r_min_s = float("inf")
            if len(L_polygons) == 1 and len(R_polygons) == 1:
                if len(L_polygons[0]) == len(polygon) and len(R_polygons[0]) == len(polygon):
                    turnpts.remove(pt)
                    queue.insert(0, polygon)
                    print("LR无用")

                if len(L_polygons[0]) == len(polygon) and len(R_polygons[0]) != len(polygon):
                    queue.insert(0, R_polygons[0])
                    turnpts.remove(pt)
                if len(L_polygons[0]) != len(polygon) and len(R_polygons[0]) == len(polygon):
                    turnpts.remove(pt)
                    queue.insert(0, L_polygons[0])
                if len(L_polygons[0]) != len(polygon) and len(R_polygons[0]) != len(polygon):
                    turnpts.remove(pt)
                    queue.insert(0, L_polygons[0])

            elif len(L_polygons) == 1 and len(R_polygons) != 1:

                if len(L_polygons[0]) == len(polygon):
                    turnpts.remove(pt)
                    for rpolygon in R_polygons:
                        queue.insert(0, rpolygon)
                    print("L无用")
                else:
                    turnpts.remove(pt)
                    queue.insert(0, L_polygons[0])

            elif len(L_polygons) != 1 and len(R_polygons) == 1:
                if len(R_polygons[0]) == len(polygon):
                    turnpts.remove(pt)
                    for lpolygon in L_polygons:
                        queue.insert(0, lpolygon)
                    print("R无用")
                else:
                    turnpts.remove(pt)
                    queue.insert(0, R_polygons[0])

            else:
                for lpolygon in L_polygons:
                    ip_Sweep = self.calcHatchPoints(lpolygon, self.ys)
                    if len(ip_Sweep) <= l_min_s:
                        l_min_s = len(ip_Sweep)

                for rpolygon in R_polygons:
                    ip_Sweep = self.calcHatchPoints(rpolygon, self.ys)
                    if len(ip_Sweep) <= r_min_s:
                        r_min_s = len(ip_Sweep)

                if r_min_s == l_min_s and r_min_s == 0:
                    try:
                        child_offset = utils.contour_offset(polygon, self.interval, self.coefficient)

                    except GeoError.Small:
                        print("child region is too small to offset")
                        # child_region can not offset will  be the fill path itself
                        turnpts.remove(pt)
                        queue.insert(0, polygon)
                        continue
                        # self.C_regions.append(indeterminacy.pop(i))
                    except GeoError.topology as t:
                        child_offset = t.polygons
                    "The child region after offsetting may become unprintable region"
                    # TODO Fix this bug
                    if self.check_self_intersection(pt, polygon,  offset_regions=child_offset):
                        # self.C_regions.append(indeterminacy.pop(i))

                        turnpts.remove(pt)
                        queue.insert(0, polygon)

                    else:
                        print("the child_region is unprintable, continue decomposition")
                        # random
                        result_clipper.append(R_splitter)
                        turnpts.remove(pt)
                        for rpolygon in R_polygons:
                             queue.insert(0, rpolygon)
                        # utils.path_show(child_offset[0])
                        # utils.path_show(R_polygons[1])
                        # utils.path_show(L_polygons[1])
                        # turnpts.remove(pt)
                        # queue.insert(0, polygon)
                if r_min_s > l_min_s:
                    result_clipper.append(R_splitter)
                    turnpts.remove(pt)
                    for rpolygon in R_polygons:
                        queue.insert(0, rpolygon)

                if r_min_s < l_min_s:
                    result_clipper.append(L_splitter)
                    turnpts.remove(pt)
                    for lpolygon in L_polygons:
                        queue.insert(0, lpolygon)

                if r_min_s == l_min_s and r_min_s != 0:
                    result_clipper.append(R_splitter)
                    turnpts.remove(pt)
                    for lpolygon in L_polygons:
                        queue.insert(0, lpolygon)
        return result

    def check_turnpts(self, pt, result_clipper):
        for clipper in result_clipper:
            X_MAX = float("-inf")
            Y_MAX = float("-inf")
            X_MIN = float("inf")
            Y_MIN = float("inf")
            for point in clipper:
                if point.x >= X_MAX:
                    X_MAX = point.x
                if point.x <= X_MIN:
                    X_MIN = point.x
                if point.y >= Y_MAX:
                    Y_MAX = point.y
                if point.y <= Y_MIN:
                    Y_MIN = point.y
            if X_MIN <= pt.x <= X_MAX and Y_MIN <= pt.y <= Y_MAX:
                return True
        return False

    def check_decomposition_useful(self, region_list, orginal_ipses):
        polygon_ipses = []

        for i in range(len(region_list)):

            temp_ipses = self.calcHatchPoints(region_list[i], self.ys)
            "判断分解后的子区域与原来的连通区域之间，同一条扫描线的交点的变化"
            list_key = list(set(temp_ipses.keys()) & set(orginal_ipses.keys()))
            score = 0
            for key in list_key:
                if len(temp_ipses[key]) < len(orginal_ipses[key]):
                    score = score + 1
            polygon_ipses.append(temp_ipses)

    def get_wrong_Sweep(self, ip_Sweep):
        pass

    def get_polygons(self, polys):
        polygons = []
        coefficient = math.pow(10, 7)
        poly_inner_list = []
        poly_out_list = []
        for poly in polys:
            if utils.direnction(poly) == 1:
                poly_inner_list.append(poly)
            else:
                poly_out_list.append(poly)
        for out_poly in poly_out_list:
            polygon = [out_poly]
            for inner_poly in poly_inner_list:
                tuple_Point = (
                    round(inner_poly.points[0].x * coefficient), round(inner_poly.points[0].y * coefficient),
                    round(inner_poly.points[0].z * coefficient))
                tuple_Poly = utils.PolylineToTuple(out_poly, coefficient)
                if pyclipper.PointInPolygon(tuple_Point, tuple_Poly) == 1:
                    polygon.append(inner_poly)
            polygons.append(polygon)
        return polygons

    def get_L_R_poly(self, pt, hatchPtses, Orient, polygon):

        lPt, rPt = self.findLRPoints(pt, hatchPtses)

        if lPt is None or rPt is None:
            # va = VtkAdaptor()
            # for poly in polygon:
            #    temp = va.drawPolyline(poly)
            #    temp.GetProperty().SetColor(0, 0, 1)
            #    temp.GetProperty().SetLineWidth(1)

            # va.drawPoint(pt).GetProperty().SetColor(1, 0, 0)
            # va.drawPoint(hatchPtses[0][0]).GetProperty().SetColor(0, 0, 1)
            # va.drawPoint(hatchPtses[0][1]).GetProperty().SetColor(0, 0, 1)
            # va.display()
            raise GeoError.Turnpt_E

        if Orient == 0:

            left_splitter = self.createSplitter(lPt, pt)
            splitter = left_splitter
            clipper, ca = Pyclipper(), ClipperAdaptor()
            clipper.AddPaths(ca.toPaths(polygon), pyclipper.PT_SUBJECT)
            clipper.AddPath(ca.toPath(left_splitter), pyclipper.PT_CLIP)
            sln = clipper.Execute(pyclipper.CT_DIFFERENCE)
        else:
            right_splitter = self.createSplitter(pt, rPt)
            splitter = right_splitter
            clipper, ca = Pyclipper(), ClipperAdaptor()
            clipper.AddPaths(ca.toPaths(polygon), pyclipper.PT_SUBJECT)
            clipper.AddPath(ca.toPath(right_splitter), pyclipper.PT_CLIP)
            sln = clipper.Execute(pyclipper.CT_DIFFERENCE)
        sln = ca.toPolys(sln, pt.z)

        return sln, splitter

    def get_regions(self, polygon):
        if len(polygon) > 1:
            "the polygon has inner contours"
            return False
        else:
            ipses = calcHatchPoints(polygon, self.ys)
            for ips in ipses:
                if len(ips) > 2:
                    return False

            return True

    def polygon_turnpts(self, queue, pt):
        index = None
        temp_min = float("inf")
        for i in range(len(queue)):
            for poly in queue[i]:
                for point in poly.points:
                    dis = point.distance(pt)
                    if dis <= temp_min:
                        temp_min = dis
                        index = i
        if temp_min >= 1e-6:
            return None
        polygon = queue.pop(index)
        return polygon

    def Same_precision(self, polygon, dig=7):
        cofficient = math.pow(10, dig)
        result = []
        for poly in polygon:
            temp_poly = Polyline()
            for point in poly.points:
                temp_poly.points.append(
                    Point3D(round(point.x * cofficient) / cofficient, round(point.y * cofficient) / cofficient,
                            round(point.z * cofficient) / cofficient))
            result.append(temp_poly)
        return result

    def select_turnpt(self, pt, turpts):
        dis = float("inf")
        index = None
        for i in range(len(turpts)):
            temp = turpts[i].distance(pt)
            if temp <= dis:
                dis = temp
                index = i
        return index
