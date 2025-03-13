
import build_connect_tree

import copy
import Polyline
from Erroe import *




def laod_path_boundary(path_file, boundary_file):
    pass



def poly_processing(polyline:Polyline.Polyline, interval, filter_angle):
    rnp = build_connect_tree.RNP.DNP(polyline, interval, filter_angle)
    outcontour = rnp.dnp()
    outcontour = build_connect_tree.insertpathes(outcontour, interval)
    outcontour_tensor = build_connect_tree.pathtotensor(outcontour)
    return outcontour, outcontour_tensor


def connect(path, r_out_poly, interval):
    path1 = r_out_poly[0]
    "以下步骤，要保证外轮廓不失真"
    out = copy.deepcopy(r_out_poly[0])
    out_r_out_poly_tensor = build_connect_tree.pathtotensor(out)
    path_list = [path]
    for path in path_list:
        out_r_out_poly = path1
        path, path_tensor = poly_processing(path, interval, 10)
        eps = interval * 0.05
        temp_interval = copy.deepcopy(interval)
        while True:
            try:
                out, path, index1, index2 = build_connect_tree.chose_reroute_points(
                    out, path, out_r_out_poly_tensor, path_tensor, temp_interval, 3)
            except interval_error as i:
                print(i)
                temp_interval = temp_interval + eps
                continue
            break
        index1 = build_connect_tree.findindex(out.points[index1], out_r_out_poly)
        out_r_out_poly.points = build_connect_tree.replaceL(index1, out_r_out_poly.points)
        path.points = build_connect_tree.replaceL(index2, path.points)
        path1 = build_connect_tree.out_to_in_connect(out_r_out_poly, path)
    "下一步，路径与内轮廓连接"
    for inner in r_out_poly[1:]:
        "为什么这里不对内轮廓进行轮廓处理，因为之前已经处理过了"
        inner_tensor = build_connect_tree.pathtotensor(inner)
        path1.points.reverse()
        path1_tensor = build_connect_tree.pathtotensor(path1)
        temp_interval = copy.deepcopy(interval)
        while True:
            try:
                path1, inner, index1, index2 = build_connect_tree.chose_reroute_points(path1,
                                                                                       inner,
                                                                                       path1_tensor,
                                                                                       inner_tensor,
                                                                                       temp_interval,
                                                                                       3)
            except interval_error as i:
                print(i)
                temp_interval = temp_interval + eps
                continue
            break
        path1.points = build_connect_tree.replaceL(index1, path1.points)
        inner.points = build_connect_tree.replaceL(index2, inner.points)
        path1 = build_connect_tree.out_to_in_connect(inner, path1)

if __name__ == "__main__":
    path_file = r"D/data/sm.txt"


