import copy
import glob
import math
import os

import numpy as np
import pyclipper
import torch
from PIL import Image
import GeomBase
import Polyline
import GeoError
import RNP
import resample
import CSS
from GenPath import GenDpPath
from VtkAdaptor import VtkAdaptor
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

def get_polygon(points):
    P = Polyline.Polyline()
    for point in points:
        if point[0] != 1348083461:
            P.points.append(GeomBase.Point3D(point[0], point[1], point[2]))
        else:
            yield P
            P = Polyline.Polyline()


def PolylineToTuple(Poly: Polyline, coefficient):
    temp = []
    for point in Poly.points:
        var_tuple = (round(point.x * coefficient), round(point.y * coefficient), round(point.z * coefficient))
        temp.append(var_tuple)
    tuple_P = tuple(temp)
    return tuple_P


def dic_slice(_dic, start, end):
    keys = list(_dic.keys())
    _dic_slice = {}
    for key in keys[start: end + 1]:
        _dic_slice[key] = _dic[key]
    return _dic_slice


def resample_path(points, num_samples, distance=None):
    """
    对三维点序列进行等间距重采样。

    参数：
    - points (torch.Tensor): 原始点序列，形状为 (N, 3)
    - num_samples (int): 重采样后的点数

    返回：
    - resampled_points (torch.Tensor): 重采样后的点序列，形状为 (num_samples, 3)
    """
    # 计算每段的距离
    deltas = points[1:] - points[:-1]  # (N-1, 3)
    segment_lengths = torch.norm(deltas, dim=1)  # (N-1,)
    cumulative_lengths = torch.cat(
        [torch.tensor([0.0], device=points.device), torch.cumsum(segment_lengths, dim=0)])  # (N,)

    total_length = cumulative_lengths[-1]
    # 生成等间距的目标距离
    if distance is not None:
        num_samples = int(total_length / distance)
    target_lengths = torch.linspace(0, total_length, steps=num_samples, device=points.device)  # (num_samples,)

    # 查找每个目标距离所在的区间
    indices = torch.searchsorted(cumulative_lengths, target_lengths, right=True)
    indices = torch.clamp(indices, 1, len(cumulative_lengths) - 1)

    # 获取区间的起始点
    idx_lower = indices - 1
    idx_upper = indices

    # 计算在区间内的比例
    length_lower = cumulative_lengths[idx_lower]
    length_upper = cumulative_lengths[idx_upper]
    segment_length = length_upper - length_lower
    # 防止除以零
    segment_length = torch.where(segment_length == 0, torch.ones_like(segment_length), segment_length)
    t = (target_lengths - length_lower) / segment_length

    # 线性插值计算新点坐标
    resampled_points = points[idx_lower] + (points[idx_upper] - points[idx_lower]) * t.unsqueeze(1)

    return resampled_points


def load_2d_contours_new(load_path, slicer_start, slice_end):
    layers = {}
    #打开文件夹
    for i in range(len(os.listdir(load_path))):
        # 遍历 z_layer
        z_path = os.path.join(load_path, "{}".format(i))
        for j in range(len(os.listdir(z_path))):
            # 读取domain
            out_boundary = glob.glob(os.path.join(z_path + "\{}".format(int(j)), "out-contour*.txt"))
            inner_boundary = glob.glob(os.path.join(z_path + "\{}".format(int(j)), "inner-contour*.txt"))

            polygon = []
            "scale"
            for out in out_boundary:
                boundary = np.loadtxt(out)[:, :]
                poly = Polyline.Polyline()
                for row in boundary:

                        poly.points.append(GeomBase.Point3D(row[0], row[1], row[2]))

                polygon.append(poly)
            for inner in inner_boundary:
                inner = np.loadtxt(inner)
                poly = Polyline.Polyline()
                for row in inner:

                    poly.points.append(GeomBase.Point3D(row[0], row[1], row[2]))
                polygon.append(poly)
            z = polygon[0].points[0].z
            if z not in layers.keys():
                layers[z] = []
            layers[z].append(polygon)

    layers = dic_slice(layers, slicer_start, slice_end)
    return layers



def load_2d_contours(inner_path, out_path, slicer_start, coefficient, slice_end):
    inner_points = np.loadtxt(inner_path)
    out_points = np.loadtxt(out_path)
    a = get_polygon(out_points)
    layers = {}
    try:
        while True:
            out_poly = next(a)
            z = out_poly.points[-1].z
            if z not in layers.keys():
                layers[z] = []
            layers[z].append([out_poly])

    except StopIteration:
        print('StopIteration')
    b = get_polygon(inner_points)

    try:
        while True:
            inner_poly = next(b)
            z = inner_poly.points[0].z
            if z in layers.keys():
                for polygons in layers[z]:
                    tuple_Point = (
                        round(inner_poly.points[0].x * coefficient), round(inner_poly.points[0].y * coefficient),
                        round(inner_poly.points[0].z * coefficient))
                    tuple_Poly = PolylineToTuple(polygons[0], coefficient)
                    if pyclipper.PointInPolygon(tuple_Point, tuple_Poly) == 1:
                        polygons.append(inner_poly)

    except StopIteration:
        print('StopIteration')

    layers = dic_slice(layers, slicer_start, slice_end)
    return layers


def direnction(contour: Polyline):
    points = contour.points
    d = 0
    for i in range(len(points) - 1):
        d += -0.5 * (points[i + 1].y + points[i].y) * (points[i + 1].x - points[i].x)
    if d > 0:
        return 0
    else:
        return 1


def resmaple_about_interval(interval, polyline, isOpen):
    length = resample.pl_arcLength(polyline.points, isOpen)
    if length < interval:
        raise GeoError.lengtherror("Too short can not resample")
    N = int(length / interval)
    pathrp = resample.ResampleCurve(polyline, N)
    return pathrp


def Show_loss(x, y):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 创建一个带有上下标的标题和标签的折线图
    plt.plot(x, y, label=":Loss")

    # 设置标题，带有上下标
    plt.title(r'迭代次数与Loss值', fontsize=14)

    # 设置X轴和Y轴标签，带有上下标
    plt.xlabel(r'迭代次数', fontsize=12)
    plt.ylabel(r'Loss', fontsize=12)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()





def Filter(polygon, interval):
    result = []
    for poly in polygon:
        # 判断当前多边形是否为内轮廓
        if direnction(poly) == 1:
            try:
                poly = resmaple_about_interval(0.4, poly, 1)
            except GeoError.lengtherror:
                print("Can not resmaple this short inner_contour")
                # remove the inner_poly
                continue
        else:
            poly = resmaple_about_interval(0.4, poly, 1)

        #if len(poly.points) >= 31:
        #    c = CSS.polygon_To_Array(poly)
        #    kappa, smooth = CSS.compute_curve_css(c, 3)
        #    poly = CSS.Array_To_polygon(smooth, poly.points[0].z)
        #else:
        #    print("Too few points to CSS")
        # DNP处理开口轮廓
        a = RNP.DNP(poly, 0.4, 5)
        temp = a.dnp()
        temp.points.append(temp.points[0])
        poly = temp
        result.append(poly)
    # path_show(polygon)
    return result


def Adjust(polygon, interval):
    result = []
    for poly in polygon:
        # 判断当前多边形是否为内轮廓
        if direnction(poly) == 1:
            try:
                poly = resmaple_about_interval(0.4, poly, 1)
            except GeoError.lengtherror:
                print("Can not resmaple this short inner_contour")
                # remove the inner_poly
                continue
        else:
            poly = resmaple_about_interval(0.4, poly, 1)
        # DNP处理开口轮廓
        result.append(poly)
    # path_show(polygon)
    return result


def Get_path_area(path):
    pass


def Get_fill_area(fill_area):
    pass


def path_show(paths):
    if len(paths) == 0:
        return
    va = VtkAdaptor()
    for i in range(len(paths)):
        va.drawPolyline(paths[i]).GetProperty().SetColor(0, 0, 1)
    va.display()


def path_conformal_show(paths, polys):
    va = VtkAdaptor()
    for i in range(len(paths)):
        va.drawPolyline(paths[i]).GetProperty().SetColor(0, 0, 1)

    for i in range(len(polys)):
        va.drawPolyline(polys[i]).GetProperty().SetColor(1, 0, 0)

    va.display()


def PolyToTensor(Poly: Polyline.Polyline):
    Tensor = torch.zeros(Poly.count(), 3)
    for i in range(len(Poly.points)):
        Tensor[i][0] = Poly.points[i].x
        Tensor[i][1] = Poly.points[i].y
        Tensor[i][2] = Poly.points[i].z
    return Tensor


def show_path_and_polygon(paths, polygon, write_flag, name):
    va = VtkAdaptor()
    # for i in range(path.shape[0]):
    for p in paths:
        c = va.showPolyline(p)
        c.GetProperty().SetColor(1, 0, 0)
        c.GetProperty().SetLineWidth(2)

    for i in range(len(polygon)):
        c1 = va.showPolyline(polygon[i])
        c1.GetProperty().SetColor(0, 0, 1)
        c1.GetProperty().SetLineWidth(2)
    if write_flag:
        va.write_image(name, va.window)
    else:
        va.display()


def get_path_orientation(path):
    """
    计算使用 Nx3 tensor 表示的多边形的方向。
    假设 path 的前两列为二维坐标 (x, y)。

    返回值:
        area: 有向面积，正值代表逆时针，负值代表顺时针，0代表退化情况。
    """
    # 提取 x, y 坐标
    x = path[:, 0]
    y = path[:, 1]

    # 对应下一个点（循环）
    x_next = torch.roll(x, shifts=-1)
    y_next = torch.roll(y, shifts=-1)

    # 计算叉乘和有向面积（不用除以2，符号判断即可）
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0

    return area


def Calculate_distance(Tensor1, Tensor2):
    x = Tensor1.unsqueeze(1).expand(-1, Tensor2.shape[0], -1)
    y = Tensor2.unsqueeze(0).expand(Tensor1.shape[0],-1, -1)
    distance = torch.norm(x - y, dim=-1)
    return distance


def Calculate_SI(a, b, c, d):
    " a  b"
    " c  d"
    ac = c - a
    ab = b - a
    ad = d - a
    if torch.cross(ac, ad)[-1] * torch.cross(ab, ad)[-1]<0:
        return True
    else:
        return False


def Show_offset_and_org(offset, original):
    va = VtkAdaptor()
    for polygon in offset:
        for i in range(len(polygon)):
            c = va.drawPolyline(polygon[i])
            c.GetProperty().SetColor(0, 0, 1)
            c.GetProperty().SetLineWidth(2)
    for polygon in original:
        for i in range(len(polygon)):
            c = va.drawPolyline(polygon[i])
            c.GetProperty().SetColor(1, 0, 0)
            c.GetProperty().SetLineWidth(2)
    va.display()


def Pre_check(polygon, interval, coefficient):
    # remove repeat points
    polygon = [reduce_repeated_points(poly) for poly in polygon]
    isOpen = True
    length = resample.pl_arcLength(polygon[0].points, isOpen)
    area = pyclipper.Area(PolylineToTuple(polygon[0], coefficient))
    area = area / (coefficient * coefficient)
    if area < pow(interval, 2) or length < interval * 2:
        raise GeoError.outpoly_too_small("too small out contour")


def reduce_repeated_points(poly):
    temp = []
    for point in poly.points:
        if point not in temp:
            temp.append(point)
    poly.points = temp
    return poly


def contour_offset(polygon: list, interval, coefficient, flag=0):
    "offset the polygon, generate filling area O = [o_1, o_2...o_n]"
    length_inner = 0
    length_out = 0
    for poly in polygon:
        if direnction(poly) == 1:
            length_inner = length_inner + 1
        else:
            length_out = length_out + 1
    if length_out == 0:
        length_out = length_inner
        length_inner = 0
    pco = pyclipper.PyclipperOffset()
    for poly in polygon:
        contour_tuple = []
        for point in poly.points:
            var_tuple = (round(point.x * coefficient), round(point.y * coefficient))
            contour_tuple.append(var_tuple)
        contour_tuple = tuple(contour_tuple)
        pco.AddPath(contour_tuple, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(-interval * coefficient)

    poly_out_list = []
    poly_inner_list = []
    total_polygons = []

    for i in range(len(solution)):
        solution[i] = np.array(solution[i]) / coefficient
        solution[i] = solution[i].tolist()

    if len(solution) == 0:
        raise GeoError.Small("The R is too small")
    if flag == 0:
        for i in range(len(solution)):
            polys1 = Polyline.Polyline()
            for j in range(len(solution[i])):
                point = GeomBase.Point3D(solution[i][j][0], solution[i][j][1], polygon[0].points[0].z)
                polys1.points.append(point)
            polys1.points.append(polys1.points[0])
            if direnction(polys1) == 1:
                poly_inner_list.append(polys1)
            else:
                poly_out_list.append(polys1)

        if length_inner != len(poly_inner_list) or len(
                poly_out_list) != length_out:
            for out_poly in poly_out_list:
                polygons = [out_poly]
                for inner_poly in poly_inner_list:
                    tuple_Point = (
                        round(inner_poly.points[0].x * coefficient), round(inner_poly.points[0].y * coefficient),
                        round(inner_poly.points[0].z * coefficient))
                    tuple_Poly = PolylineToTuple(out_poly, coefficient)
                    if pyclipper.PointInPolygon(tuple_Point, tuple_Poly) == 1:
                        polygons.append(inner_poly)
                total_polygons.append(polygons)
            raise GeoError.topology(msg="topology will be change after one time offset", polygons=total_polygons,
                                    len_inner=len(poly_inner_list))
        else:
            poly_out_list.extend(poly_inner_list)
            total_polygons.append(poly_out_list)
            return total_polygons
    else:
        return None


def To_Intersect_ys(polygon, interval):  # Get the scan_lines of the polygon
    ys, yMin, yMax = [], float('inf'), float('-inf')
    for poly in polygon:
        for pt in poly.points:
            yMin, yMax = min(yMin, pt.y), max(yMax, pt.y)
    y = yMin + interval
    while yMax - y >= 1e-10:
        ys.append(y)
        y += interval

    ys.sort(reverse=True)
    return ys


def get_ys(poly, ys):  # extract some scan_lines of the poly from total scan_lines
    yMin, yMax = float('inf'), float('-inf')
    for pt in poly.points:
        yMin, yMax = min(yMin, pt.y), max(yMax, pt.y)
    front = -1
    behind = -1
    for i in range(len(ys)):
        if yMax - ys[i] >= 1e-10:
            front = i
            break
    if front != -1:
        for j in range(len(ys) - 1, front, -1):
            if ys[j] - yMin >= 1e-10:
                behind = j
                break
        if behind == -1:
            local_ys = ys[front]
            return local_ys
    else:
        local_ys = []
        return local_ys
    local_ys = ys[front:behind + 1]
    return local_ys


def Generate_local_Z(child_regions, interval, ys):
    return GenDpPath(child_regions, interval, 0, ys).generate2()


def replaceL(index, A):

    return A[index:] + A[:index]


def findindex(point: GeomBase.Point3D, endpath: Polyline):
    for i in range(len(endpath.points)):
        if abs(endpath.points[i].x - point.x) <= 1e-8 and abs(endpath.points[i].y - point.y) <= 1e-8:
            break
    return i


def insert(fatherpath, childpath, interval, freroute, creroute):

    if ((fatherpath.points[freroute[1]].x <= childpath.points[creroute[1]].x) and (
            fatherpath.points[freroute[0]].x <= childpath.points[creroute[0]].x)):
        insertpointx = (fatherpath.points[freroute[0]].x + childpath.points[creroute[1]].x) * 0.5
    if (
            (fatherpath.points[freroute[1]].x >= childpath.points[creroute[1]].x) and (
            fatherpath.points[freroute[0]].x >= childpath.points[creroute[0]].x)):
        insertpointx = (fatherpath.points[freroute[1]].x + childpath.points[creroute[0]].x) * 0.5
    if ((fatherpath.points[freroute[0]].x <= childpath.points[creroute[0]].x) and (
            fatherpath.points[freroute[1]].x >= childpath.points[creroute[1]].x)):
        insertpointx = (fatherpath.points[freroute[0]].x + fatherpath.points[freroute[1]].x) * 0.5

    if ((fatherpath.points[freroute[0]].x >= childpath.points[creroute[0]].x) and (
            fatherpath.points[freroute[1]].x <= childpath.points[creroute[1]].x)):
        insertpointx = (childpath.points[creroute[0]].x + childpath.points[creroute[1]].x) * 0.5

    if (freroute[0] < freroute[1]):
        for i in range(freroute[0], freroute[1] + 1):
            if (i != len(fatherpath.points) - 1):
                if insertpointx < fatherpath.points[i].x and insertpointx > fatherpath.points[i + 1].x:
                    fatherpath.points.insert(i + 1,
                                             GeomBase.Point3D(insertpointx - interval, fatherpath.points[i].y,
                                                              fatherpath.points[freroute[0]].z))
                    fatherpath.points.insert(i + 1,
                                             GeomBase.Point3D(insertpointx, fatherpath.points[i].y,
                                                              fatherpath.points[freroute[0]].z))
                    break
            else:
                if insertpointx < fatherpath.points[i].x and insertpointx > fatherpath.points[0].x:
                    fatherpath.points.insert(i + 1,
                                             GeomBase.Point3D(insertpointx - interval, fatherpath.points[i].y,
                                                              fatherpath.points[freroute[0]].z))
                    fatherpath.points.insert(i + 1,
                                             GeomBase.Point3D(insertpointx, fatherpath.points[i].y,
                                                              fatherpath.points[freroute[0]].z))
                    break



    else:
        for i in range(freroute[1], freroute[0] + 1):
            if (i != len(fatherpath.points) - 1):
                if insertpointx > fatherpath.points[i].x and insertpointx < fatherpath.points[i + 1].x:
                    fatherpath.points.insert(i + 1,
                                             GeomBase.Point3D(insertpointx, fatherpath.points[i].y,
                                                              fatherpath.points[freroute[0]].z))
                    fatherpath.points.insert(i + 1,
                                             GeomBase.Point3D(insertpointx - interval, fatherpath.points[i].y,
                                                              fatherpath.points[freroute[0]].z))
                    break
            else:
                if insertpointx > fatherpath.points[i].x and insertpointx < fatherpath.points[0].x:
                    fatherpath.points.insert(i + 1,
                                             GeomBase.Point3D(insertpointx, fatherpath.points[i].y,
                                                              fatherpath.points[freroute[0]].z))
                    fatherpath.points.insert(i + 1,
                                             GeomBase.Point3D(insertpointx - interval, fatherpath.points[i].y,
                                                              fatherpath.points[freroute[0]].z))
                    break

    if (creroute[0] < creroute[1]):
        for i in range(creroute[0], creroute[1] + 1):
            if (i != len(childpath.points) - 1):
                if insertpointx < childpath.points[i].x and insertpointx > childpath.points[i + 1].x:
                    childpath.points.insert(i + 1,
                                            GeomBase.Point3D(insertpointx - interval, childpath.points[i].y,
                                                             fatherpath.points[freroute[0]].z))
                    childpath.points.insert(i + 1,
                                            GeomBase.Point3D(insertpointx, childpath.points[i].y,
                                                             fatherpath.points[freroute[0]].z))
                    break
            else:
                if insertpointx < childpath.points[i].x and insertpointx > childpath.points[0].x:
                    childpath.points.insert(i + 1,
                                            GeomBase.Point3D(insertpointx - interval, childpath.points[i].y,
                                                             fatherpath.points[freroute[0]].z))
                    childpath.points.insert(i + 1,
                                            GeomBase.Point3D(insertpointx, childpath.points[i].y,
                                                             fatherpath.points[freroute[0]].z))
                    break

    else:
        for i in range(creroute[1], creroute[0] + 1):
            if (i != len(childpath.points) - 1):
                if insertpointx > childpath.points[i].x and insertpointx < childpath.points[i + 1].x:
                    childpath.points.insert(i + 1,
                                            GeomBase.Point3D(insertpointx, childpath.points[i].y,
                                                             fatherpath.points[freroute[0]].z))
                    childpath.points.insert(i + 1,
                                            GeomBase.Point3D(insertpointx - interval, childpath.points[i].y,
                                                             fatherpath.points[freroute[0]].z))
                    break
            else:
                if insertpointx > childpath.points[i].x and insertpointx < childpath.points[0].x:
                    childpath.points.insert(i + 1,
                                            GeomBase.Point3D(insertpointx, childpath.points[i].y,
                                                             fatherpath.points[freroute[0]].z))
                    childpath.points.insert(i + 1,
                                            GeomBase.Point3D(insertpointx - interval, childpath.points[i].y,
                                                             fatherpath.points[freroute[0]].z))
                    break

    fatherxL = GeomBase.Point3D(insertpointx - interval, fatherpath.points[freroute[0]].y, fatherpath.points[freroute[0]].z)
    fatherxR = GeomBase.Point3D(insertpointx, fatherpath.points[freroute[0]].y, fatherpath.points[freroute[0]].z)
    childxL = GeomBase.Point3D(insertpointx - interval, childpath.points[creroute[0]].y, fatherpath.points[freroute[0]].z)
    childxR = GeomBase.Point3D(insertpointx, childpath.points[creroute[0]].y, fatherpath.points[freroute[0]].z)
    return fatherxL, fatherxR, childxL, childxR

def contourconnect(pathhigher: Polyline, path: Polyline):
    pathhighervar= Polyline.Polyline()
    pathhighervar.points =copy.deepcopy(pathhigher.points)
    pathhighervar.points.extend(path.points)
    pathhighervar.points.append(pathhighervar.points[0])
    return pathhighervar