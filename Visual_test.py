import math

import shapely
import torch
import VtkAdaptor
import pyclipper


def get_polygon_3D(feature: torch.tensor) -> list:
    """

    :param feature: shape(3, ) single feature
    :return: list of vertices
    """
    vertices = []
    radius = 0.5 * math.pow(2, 0.5) * feature[2]
    for angle in range(45, 405, 90):
        x = feature[0] + radius * math.cos(math.radians(angle))
        y = feature[1] + radius * math.sin(math.radians(angle))
        z = 0
        vertices.append((x, y, z))
    vertices.append(vertices[0])
    return vertices


def Trans(poly) -> list:
    """
    :param poly: list of polyline, which is composed of 2D tensor N X 3
    :return: new_polygon: the type which is used for shapely lib with zero z coord
    """
    new_poly = []
    for i in range(poly.shape[0]):
        temp_coord = poly[i].tolist()
        new_poly.append([temp_coord[0], temp_coord[1], 0])
    return new_poly


def Trans_cliip(poly, coefficient) -> list:
    """
    :param poly: list of polyline, which is composed of 2D tensor N X 3
    :return: new_polygon: the type which is used for shapely lib with zero z coord
    """
    new_poly = []
    for i in range(poly.shape[0]):
        temp_coord = poly[i].tolist()
        new_poly.append([temp_coord[0]*coefficient, temp_coord[1]*coefficient, 0])
    return new_poly


def Vis_cell(cell_dict, point_star, point_hat, open_path, offset_polygon, error_index):
    for i in range(-1, open_path.shape[0] - 1):
        va = VtkAdaptor.VtkAdaptor()
        # 路径上的一个点 与 i-1 site 和i site有关
        # 先画出i有关的线段
        for k in range(len(offset_polygon)):
            c1 = va.showPolyline(offset_polygon[k])
            c1.GetProperty().SetColor(0, 0, 1)
            c1.GetProperty().SetLineWidth(2)

        c = va.showPolyline(open_path)
        c.GetProperty().SetColor(1, 0, 1)
        c.GetProperty().SetLineWidth(1)

        c = va.showPolyline(open_path[[i - 1, i]])
        c.GetProperty().SetColor(1, 0, 0)
        c.GetProperty().SetLineWidth(2)

        c = va.showPolyline(open_path[[i, i + 1]])
        c.GetProperty().SetColor(1, 0, 0)
        c.GetProperty().SetLineWidth(2)

        # 把当前点画出来
        va.showPoint(open_path[i], radius=0.05)
        # 画出有关的cell
        pre = i - 1
        next = i
        if pre == -1:
            pre = open_path.shape[0] - 1
        if next == -1:
            next = open_path.shape[0] - 1
            pre = open_path.shape[0] - 2
        if next not in error_index:
            for kk in cell_dict[next].keys():
                if kk != "Point_E":
                    for j in range(len(cell_dict[next][kk])):
                        c1 = va.showPolyline(cell_dict[next][kk][j])
                        c1.GetProperty().SetColor(0, 0, 1)
                        c1.GetProperty().SetLineWidth(2)

            for item in point_star[next]:
                c = va.showPoint(item, radius=0.05)
                c.GetProperty().SetColor(0, 0, 1)
            for item in point_hat[next]:
                c = va.showPoint(item, radius=0.05)
                c.GetProperty().SetColor(1, 0, 0)

        if pre not in error_index:
            for kk in cell_dict[pre].keys():
                if kk != "Point_S":
                    for j in range(len(cell_dict[pre][kk])):
                        c1 = va.showPolyline(cell_dict[pre][kk][j])
                        c1.GetProperty().SetColor(0, 0, 1)
                        c1.GetProperty().SetLineWidth(2)
        va.write_image("V/{}.png".format(i))


def _2d_coordTo3d(tensor, device="cuda"):

    return torch.cat((tensor[:, :2], torch.zeros(tensor.shape[0], 1).to(tensor.device)), dim=-1)

def Padding(tensor):
    return torch.cat((tensor[:, :2], torch.ones(tensor.shape[0], 1).to(tensor.device)), dim=-1)



def visual_polygons(selected: torch.tensor, features: torch.tensor, selected_features, flag = 0) -> None:
    """

    :param features: grid_size and coordinate shape (256*3)
    :param selected: shape (valid_length)
    :return:
    """

    va = VtkAdaptor.VtkAdaptor()

    for i in range(features.shape[0]):
        JUDGE = features[i] == 0
        if JUDGE.all().item() == 1:
            break
        else:
            grid = get_polygon_3D(features[i])
            va.showgenPolyline(grid).GetProperty().SetColor(0, 0, 1)
    if selected_features is not None:
        if flag == 0:

            selected_v = selected[selected != 511]
            selected_features = features[selected_v][:, :-1]
            # features.cpu().numpy()
            selected_features = _2d_coordTo3d(selected_features)
            va.showgenPolyline(selected_features).GetProperty().SetColor(1, 0, 0)

        else:
            va.showgenPolyline(selected_features).GetProperty().SetColor(1, 0, 0)

    va.display()



def Visual_part(parts, part_problems, polygon, grid_flag, polygon_flag):
    va  = VtkAdaptor.VtkAdaptor()
    for i, features in enumerate(part_problems):
        for j in range(features.shape[0]):
            JUDGE = features[j] == 0
            if JUDGE.all().item() == 1:
                break
            else:
                if grid_flag:
                    grid = get_polygon_3D(features[j])
                    T = va.showgenPolyline(grid)
                    T.GetProperty().SetColor(0, 0, 1)
                    T.GetProperty().SetLineWidth(2)
        selected_features = parts[i]
        T2 = va.showgenPolyline(selected_features)
        T2.GetProperty().SetColor(1, 0, 0)
        T2.GetProperty().SetLineWidth(2)
        va.showPoint(selected_features[0]).GetProperty().SetColor(1, 0, 0)
        va.showPoint(selected_features[-1]).GetProperty().SetColor(0, 0, 1)
    if polygon_flag:
        for i in range(len(polygon)):
            T1 = va.showPolyline(polygon[i])
            T1.GetProperty().SetColor(0, 0, 1)
            T1.GetProperty().SetLineWidth(2)

    va.display()


def Visual_stress_part(parts, part_problems, polygon):
    va = VtkAdaptor.VtkAdaptor()
    #for features in part_problems:
    #    for j in range(len(features)):
    #        grid = get_polygon_3D(features[j])
    #        T = va.showgenPolyline(grid)
    #        T.GetProperty().SetColor(0, 0, 1)
    #        T.GetProperty().SetLineWidth(2)

    for path in parts:
        if path.shape[0] >= 2:
            T2 = va.showgenPolyline(path)
            T2.GetProperty().SetColor(1, 0, 0)
            T2.GetProperty().SetLineWidth(2)
        else:
            va.showPoint(path[0], 0.1).GetProperty().SetColor(1, 0, 0)

    #for i in range(len(polygon)):
    #    T1 = va.showPolyline(polygon[i])
    #    T1.GetProperty().SetColor(0, 0, 1)
    #    T1.GetProperty().SetLineWidth(2)
    va.display()


def Visual_part_colors(parts, part_problems, polygon, patterns, grid_flag, polygon_flag, write_flag, name):
    va = VtkAdaptor.VtkAdaptor()
    for i, features in enumerate(part_problems):
        for j in range(features.shape[0]):
            JUDGE = features[j] == 0
            if JUDGE.all().item() == 1:
                break
            else:
                if grid_flag:
                    grid = get_polygon_3D(features[j])
                    T = va.showgenPolyline(grid)
                    T.GetProperty().SetColor(0, 0, 1)
                    T.GetProperty().SetLineWidth(2)
        selected_features = parts[i]
        if patterns[i] == 0:
            T2 = va.showgenPolyline(selected_features)
            T2.GetProperty().SetColor(1, 0, 0)
            T2.GetProperty().SetLineWidth(2)

        elif patterns[i] == 1:
            T2 = va.showgenPolyline(selected_features)
            T2.GetProperty().SetColor(1, 0, 1)
            T2.GetProperty().SetLineWidth(2)
        elif patterns[i] == 2:
            T2 = va.showgenPolyline(selected_features)
            T2.GetProperty().SetColor(0, 0, 1)
            T2.GetProperty().SetLineWidth(2)
        elif patterns[i] == 3:
            T2 = va.showgenPolyline(selected_features)
            T2.GetProperty().SetColor(1, 0.4, 0)
            T2.GetProperty().SetLineWidth(2)

        va.showPoint(selected_features[0]).GetProperty().SetColor(1, 0, 0)
        va.showPoint(selected_features[-1]).GetProperty().SetColor(0, 0, 1)
    if polygon_flag:
        for i in range(len(polygon)):
            T1 = va.showPolyline(polygon[i])
            T1.GetProperty().SetColor(0, 0, 1)
            T1.GetProperty().SetLineWidth(2)
    if write_flag:
        va.write_image("./Total_random/{}.png".format(name), va.window)
    else:
        va.display()


def visual_grid_polygons(selected: torch.tensor, features: torch.tensor, selected_features, polygon, flag = 0, mark_end = 0) -> None:
    """

    :param features: grid_size and coordinate shape (256*3)
    :param selected: shape (valid_length)
    :return:
    """

    va = VtkAdaptor.VtkAdaptor()

    for i in range(features.shape[0]):
        JUDGE = features[i] == 0
        if JUDGE.all().item() == 1:
            break
        else:
            grid = get_polygon_3D(features[i])
            T = va.showgenPolyline(grid)
            T.GetProperty().SetColor(0, 0, 1)
            T.GetProperty().SetLineWidth(2)

    if polygon is not None:
        for i in range(len(polygon)):
            T1 = va.showPolyline(polygon[i])
            T1.GetProperty().SetColor(0, 0, 1)
            T1.GetProperty().SetLineWidth(2)

    if selected_features is not None:
        if flag == 0:
            selected_v = selected[selected != 511]
            selected_features = features[selected_v][:, :-1]
            # features.cpu().numpy()
            selected_features = _2d_coordTo3d(selected_features)
            T2 = va.showgenPolyline(selected_features)
            T2.GetProperty().SetColor(0, 0, 1)
            T2.GetProperty().SetLineWidth(2)

        else:
            T2 = va.showgenPolyline(selected_features)
            T2.GetProperty().SetColor(1, 0, 0)
            T2.GetProperty().SetLineWidth(2)
            if mark_end == 1:
                va.showPoint(selected_features[0]).GetProperty().SetColor(1, 0, 0)
                va.showPoint(selected_features[-1]).GetProperty().SetColor(0, 0, 1)
                # 红起蓝终

    # va.write_image("1", va.window)
    va.display()


def visual_gird_and_points(grids, points):
    va = VtkAdaptor.VtkAdaptor()
    for i in range(grids.shape[0]):
        grid = get_polygon_3D(grids[i])
        va.showgenPolyline(grid).GetProperty().SetColor(0, 0, 1)
    for j in range(points.shape[0]):
        va.showPoint(points[j], radius=0.1).GetProperty().SetColor(1, 0, 0)
    va.display()


def visual_grid_polygons_mark_start(selected: torch.tensor, features: torch.tensor, selected_features, polygon, start, end,  flag = 0, mark_end = 0) -> None:
    """

    :param features: grid_size and coordinate shape (256*3)
    :param selected: shape (valid_length)
    :return:
    """

    va = VtkAdaptor.VtkAdaptor()

    for i in range(features.shape[0]):
        JUDGE = features[i] == 0
        if JUDGE.all().item() == 1:
            break
        else:
            grid = get_polygon_3D(features[i])
            va.showgenPolyline(grid).GetProperty().SetColor(0, 0, 1)
    if polygon is not None:
        for i in range(len(polygon)):
            va.showPolyline(polygon[i]).GetProperty().SetColor(0, 0, 1)

    if selected_features is not None:
        if flag == 0:
            selected_v = selected[selected != 255]
            selected_features = features[selected_v][:, :-1]
            # features.cpu().numpy()
            selected_features = _2d_coordTo3d(selected_features)
            va.showgenPolyline(selected_features).GetProperty().SetColor(1, 0, 0)

        else:
            va.showgenPolyline(selected_features).GetProperty().SetColor(1, 0, 0)
            if mark_end == 1:
                va.showPoint(selected_features[0]).GetProperty().SetColor(1, 0, 0)
                va.showPoint(selected_features[-1]).GetProperty().SetColor(0, 0, 1)
        #va.showPoint(features[start]).GetProperty().SetColor(1, 0, 0)
        #va.showPoint(features[end]).GetProperty().SetColor(0, 0, 1)

    va.display()


def reconnect(trajectory: torch.tensor, grids: torch.tensor):
    """

    :param trajectory: ordered point
    :param grids: map(grids to present polygon)
    :return: trajectory reconnected
    """


def show_edge_of_cell(Point: torch.tensor, cell, va, edge):


    c = va.showPoint(Point, radius=0.05)
    c.GetProperty().SetColor(1, 0, 0)
    c = va.showPolyline(open_path)
    c.GetProperty().SetColor(0, 0, 1)
    c.GetProperty().SetLineWidth(1)
    c = va.showPolyline(array0 / coff)
    c.GetProperty().SetColor(1, 0, 0)
    c.GetProperty().SetLineWidth(2)
    return c


def show_path_and_polygon(path, polygon, write_flag, name):
    va = VtkAdaptor.VtkAdaptor()
    # for i in range(path.shape[0]):
    c = va.showPolyline(path)
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


def tensor_to_polygon(tensor_list: list):

    polygon = [Trans(poly) for poly in tensor_list]
    out = polygon.pop(0)
    multi_polygon = shapely.Polygon(out, polygon)
    return multi_polygon, tensor_list


def Get_A_L(tensor_list, interval = 0.5):

    Polygon, _ = tensor_to_polygon(tensor_list)
    A = Polygon.area
    L = A/0.5
    return A, L

def Show_contour_off(Features,Point_set, Next, Pre, Point_idx, Contours):
    for i in range(Features.shape[0]):
       #if i < 363:
        #    continue
        va = VtkAdaptor.VtkAdaptor()

        for polygon in Contours:
            for j in range(len(polygon)):
                va.showPolyline(polygon[j]).GetProperty().SetColor(0, 0, 1)
        grid = get_polygon_3D(Features[i], 1)
        T = va.showgenPolyline(grid)
        T.GetProperty().SetColor(0, 0, 1)
        T.GetProperty().SetLineWidth(2)
        T = va.showPoint(Point_set[Point_idx[i].item()],  0.1)
        T.GetProperty().SetColor(0, 0, 1)
        T = va.showPoint(Point_set[Pre[i][0].item()], 0.1)
        T.GetProperty().SetColor(1, 0, 0)
        T = va.showPoint(Point_set[Next[i][0].item()], 0.1)
        T.GetProperty().SetColor(0.5, 1, 0)
        va.display()


def Show_line_field(Features, start, end, ID):
    va = VtkAdaptor.VtkAdaptor()
    for i in range(Features.shape[0]):
        grid = get_polygon_3D(Features[i])
        T = va.showgenPolyline(grid)
        T.GetProperty().SetColor(0, 0, 1)
        T.GetProperty().SetLineWidth(1)
        T = va.showgenPolyline([start[i], end[i]])
        T.GetProperty().SetColor(0, 0, 1)
        T.GetProperty().SetLineWidth(3)
        T.GetProperty().SetColor(1, 0, 0)
    # va.write_image("Field/Pendant/{}".format(ID), va.window)
    va.display()

def tensor_to_pyclipper(tensor_list, coefficient):

    polygon = [Trans_cliip(poly, coefficient) for poly in tensor_list]
    return polygon


def Visual_graph():

    import networkx as nx
    import matplotlib.pyplot as plt

    # 创建带有权重的图
    G = nx.Graph()
    G.add_edge('A', 'B', weight=3)
    G.add_edge('B', 'C', weight=5)

    # 绘制图形
    pos = nx.spring_layout(G)  # 定义节点的布局
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.show()




def Polygon_to_tensor(polygon: shapely.Polygon):

    tensor_list = []
    exterior_ring = list(polygon.exterior.coords)
    interior_rings = [list(interior.coords) for interior in polygon.interiors]

    tensor_list.append(torch.tensor(exterior_ring))

    for i in range(len(interior_rings)):
        tensor_list.append(torch.tensor(interior_rings[i]))

    return tensor_list


def list_to_polygon(polygon_list, coefficent):

    polygon = [torch.tensor(poly)/coefficent for poly in polygon_list]
    polygon = _2dto3d(polygon)

    return polygon



def Polygon_offset(polygon: list, cofficient, interval = 0.6):
    polygon = tensor_to_pyclipper(polygon, cofficient)
    co = pyclipper.PyclipperOffset()
    co.AddPaths(polygon, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    "minus means inward offset"
    solution = co.Execute(interval*cofficient)
    polygon = list_to_polygon(solution, cofficient)
    return polygon


def path_offset(path, cofficient):
    path_c = Trans_cliip(path, cofficient)
    co = pyclipper.PyclipperOffset()
    co.AddPath(path_c, pyclipper.JT_MITER, pyclipper.ET_OPENROUND)
    solution = co.Execute(0.25 * cofficient)
    path = list_to_polygon(solution, cofficient)
    return path[0]



def _2dto3d(polygon):

    return [ torch.cat((poly, torch.zeros((poly.shape[0], 1), device = poly.device)), dim=-1) for poly in polygon]















