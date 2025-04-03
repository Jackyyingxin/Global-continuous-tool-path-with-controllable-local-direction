import matplotlib
import numpy
import pyclipper
import torch
import networkx as nx
import GeomAIgo
import copy
import argregin
import utils
from ClipperAdaptor import ClipperAdaptor
import GeoError
from VtkAdaptor import VtkAdaptor
from global_connect import Global_Connect
import matplotlib.pyplot as plt


def generate_2D_continuous_path(load_path, start, coefficient, end, interval, angles, iter_num,
                                learning_rate, W1, W2, R, flag):
    paths = []
    layers = utils.load_2d_contours_new(load_path, start, end)
    c_layers = copy.deepcopy(layers)
    c_layer_key = list(c_layers.keys())
    for z in range(len(c_layer_key)):
        key = c_layer_key[z]
        for g in range(len(c_layers[key])):
            # LOAD the polygon in the same layer
            polys = c_layers[key][g]

            # polys = utils.contour_offset(polys, 1.13032, coefficient)[0]

            try:
                utils.Pre_check(polys, interval, coefficient)
                # If the out-contour is too small or too short， we will use the out-contour as the fill-path
            except GeoError.outpoly_too_small:
                paths.append(polys[0])
                continue
            polys = utils.Filter(polys, interval)
            # Is_open == 0
            try:
                filling_areas = utils.contour_offset(polys, interval, coefficient)
                "If a topology_error occurs indicating the appearance of a pocket or a modification of the original \
                topology, then filling_area should be set to t.polygons."
            except GeoError.topology as t:
                filling_areas = t.polygons
            except GeoError.Small as s:
                print(s.msg)
                paths.append(polys[0])
                continue
            utils.Show_offset_and_org(offset=[polys], original=filling_areas)
            for fill_area in filling_areas:
                try:
                    path = Generate_path(fill_area, interval, angles[z], coefficient)
                    paths.append(path)
                except GeoError.Small as s:
                    print(s.msg)
                    # remove the fill area that is too small to fill
                    paths.append(polys[0])
                    continue
                except GeoError.Decomposition as d:
                    print(d.msg)
                    continue
                except GeoError.Connect_Error:
                    continue
            import PathOptimizer_cuda_test
            total_boundary = [utils.PolyToTensor(poly)[:-1, :].cuda() for poly in polys]

            opt_path = []
            # utils.path_conformal_show(paths, polys)
            paths = [utils.resmaple_about_interval(0.4, path, 0) for path in paths]
            # utils.path_show(paths)
            _2paths = []
            boundaries = []
            for i in range(len(paths)):
                path = utils.PolyToTensor(paths[i])[:, :].cuda()
                _2paths.append(torch.clone(path))
                boundary = utils.contour_offset(filling_areas[i], -interval, coefficient)[0]
                boundary = [utils.resmaple_about_interval(R, poly, 0) for poly in boundary]
                boundary = [utils.PolyToTensor(poly)[:, :].cuda() for poly in boundary]
                boundaries.append(copy.deepcopy(boundary))
                utils.show_path_and_polygon([path], boundary, 0, 0)
                A = PathOptimizer_cuda_test.optimizer(path, boundary, interval, 4 * interval, W1, W2)
                A.start(iter_num, learning_rate)
                # utils.show_path_and_polygon([A.path], boundary, 0, 0)
                opt_path.append(A.path)
            x = numpy.arange(1, iter_num + 1)
            utils.Show_loss(x, numpy.array(A.loss_log))
            va = VtkAdaptor()
            # for i in range(path.shape[0]):
            for p in opt_path:
                c = va.showPolyline(p)
                c.GetProperty().SetColor(1, 0, 0)
                c.GetProperty().SetLineWidth(2)
            for polygon in boundaries:
                for i in range(len(polygon)):
                    c1 = va.showPolyline(polygon[i])
                    c1.GetProperty().SetColor(0, 0, 1)
                    c1.GetProperty().SetLineWidth(2)
            va.display()
            for i in range(len(filling_areas)):
                for j in range(len(filling_areas[i])):
                    filling_areas[i][j] = (utils.PolyToTensor(filling_areas[i][j])[:-1, :].cuda())

            # resample all polyline

            for i in range(len(filling_areas)):
                for j in range(len(filling_areas)):
                    filling_areas[i][j] = utils.resample_path(filling_areas[i][j], 4, interval)

            for i in range(len(total_boundary)):
                total_boundary[i] = utils.resample_path(total_boundary[i], 4, interval)
            MST = Build_Graph(filling_areas, total_boundary, interval)
            path = Connect_conformal_path(opt_path, total_boundary, MST, interval)
            utils.show_path_and_polygon([path], [], 0, 0)

            # utils.show_path_and_polygon(opt_path, total_boundary, 0, 0)


def Connect_conformal_path(paths, polys, MST, interval):
    polys.extend(paths)
    total_path = [polys[0]]
    Connect(0, MST, polys, total_path, interval)
    return total_path[0]


def Connect(root, MST, paths, total_path, interval):
    for node in list(MST.neighbors(root)):
        total_path[0] = Get_Connect_points(MST, node, total_path[0], paths[node], interval)
        Connect(node, MST, paths, total_path, interval)


def Exist_edge(Node1, Node2, interval):
    if type(Node1["data"]) == list and type(Node2["data"]) == torch.Tensor:
        flag = False
        for contour in Node1["data"]:
            distance = utils.Calculate_distance(contour, Node2["data"])
            D = Estimate_distance(distance)
            mask = (0.9 * interval <= D) & (D <= 1.1 * interval)
            mask = torch.all(mask, dim=-1)
            if torch.any(mask):
                flag = True
                break
        return flag
    if type(Node2["data"]) == list and type(Node1["data"]) == torch.Tensor:
        flag = False
        for contour in Node2["data"]:
            distance = utils.Calculate_distance(contour, Node1["data"])
            D = Estimate_distance(distance)
            # D.cpu().numpy()
            mask = (0.7 * interval <= D) & (D <= 1.4 * interval)
            # mask.cpu().numpy()
            mask = torch.all(mask, dim=-1)
            if torch.any(mask):
                flag = True
                break
        return flag
    if type(Node2["data"]) == list and type(Node1["data"]) == list:
        return False

    if type(Node2["data"]) == torch.Tensor and type(Node1["data"]) == torch.Tensor:
        distance = utils.Calculate_distance(Node1["data"], Node2["data"])
        D = Estimate_distance(distance)
        mask = (0.9 * interval <= D) & (D <= 1.1 * interval)
        mask = torch.all(mask, dim=-1)
        return torch.any(mask)


def Get_Connect_points(MST, node, path1, path2, interval):
    """
    path1: 保证path1永远是逆时针,root一定是外轮廓
    # path2 如果是内轮廓
        path2:保证path2永远是顺时针
    # 如果path1 是 外轮廓， path2 如果是另一个填充区域的路径，path2是逆时针
    # 如果path1是一个区域的填充路径，path2是另一个区域的填充路径， path2 roll的时候顺多滚动一位

    """

    if type(MST.nodes[node]["data"]) == torch.Tensor:
        # path2是内轮廓
        path2 = torch.flip(path2, dims=[0])

    if type(MST.nodes[node]["data"]) == list:
        pass

        # path2是逆时针，且是另外一个区域的连续路径
        # if utils.get_path_orientation(path1) <= 0:
        #        path1 = torch.flip(path1, dims=[0])
        # if utils.get_path_orientation(path2) > 0:
        #    path2 = torch.flip(path2, dims=[0])

    # 计算两个路径各点间的欧式距离矩阵
    distance_matrix = utils.Calculate_distance(path1, path2)

    # 计算从 path1 到 path2 的最小距离及对应索引
    d1, idx1 = distance_matrix.min(dim=-1)  # d1[i] = path1中第 i 个点到 path2 中最近点的距离，idx1[i] 为对应 path2 的索引

    # 计算从 path2 到 path1 的最小距离及对应索引
    d2, idx2 = distance_matrix.min(dim=0)  # d2[j] = path2中第 j 个点到 path1 中最近点的距离，idx2[j] 为对应 path1 的索引

    # 分别找出在 d1 和 d2 中最小的值及对应索引
    min_d1, i1 = d1.min(0)  # 在 path1 方向上最接近的点，其在 path1 中的索引 i1，且对应距离 min_d1
    min_d2, j1 = d2.min(0)  # 在 path2 方向上最接近的点，其在 path2 中的索引 j1，且对应距离 min_d2

    # 根据较小的候选距离决定采用哪一侧作为连接点
    if min_d1 <= min_d2:
        _1index = i1
        _2index = idx1[i1]
    else:
        _2index = j1
        _1index = idx2[j1]

    # 循环平移各路径，使得选定的连接点移到首位

    path2 = torch.roll(path2, dims=0, shifts=-_2index.item())
    path2 = torch.flip(path2, dims=[0])

    path1 = torch.roll(path1, dims=0, shifts=-_1index.item())
    # path2 = torch.roll(path2, dims=0, shifts=-_2index.item())

    # 根据某种判断条件决定是否翻转 path2（这里用 Calculate_SI 判断两路径端点的相对关系）
    # if utils.Calculate_SI(path1[0], path1[-1], path2[0], path2[-1]):
    #    path2 = torch.flip(path2, dims=[0])

    # 拼接两个路径
    path1 = torch.cat((path1, path2), dim=0)
    return path1


def Estimate_distance(distance_matrix):
    Min_distance, index = torch.topk(distance_matrix, k=2, dim=-1, largest=False)
    Min_distance = torch.mean(Min_distance, dim=-1, keepdim=True)
    Next_distance = torch.roll(Min_distance, dims=0, shifts=-1)
    D = torch.cat((Min_distance, Next_distance), dim=-1)
    return D


def Build_Graph(file_region, polys, interval):
    total = polys + file_region
    "total的第一个节点是总的外轮廓"
    G = nx.Graph()
    for i in range(len(total)):
        G.add_node(i, data=total[i])
    for i in range(G.number_of_nodes()):
        for j in range(i + 1, G.number_of_nodes()):
            if Exist_edge(G.nodes[i], G.nodes[j], interval):
                G.add_edge(i, j)
    pos = nx.spring_layout(G)  # 使用spring布局
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
    # 显示图
    plt.title("Graph Structure")
    # plt.show()
    MST = nx.dfs_tree(G, source=0)
    for node in MST.nodes():
        MST.nodes[node].update(G.nodes[node])
    pos = nx.spectral_layout(MST)  # 使用spring布局
    nx.draw(MST, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
    # 显示图
    plt.title("Graph Structure")
    # plt.show()
    return MST


def Generate_path(polygon, interval, angle, coefficient):
    z = polygon[0].points[0].z
    area = pyclipper.Area(utils.PolylineToTuple(polygon[0], coefficient))
    area = area / (10000 * 10000)
    if area < pow(0.8, 2):
        raise GeoError.Small("too small filling_area")
    for _ in range(3):
        polygon = utils.Adjust(polygon, 0.4)

    try:
        ot = utils.contour_offset(polygon, interval, coefficient)
    except GeoError.topology as t:
        if t.len_inner < len(polygon) - 1:
            pass
        else:
            pass
            # raise GeoError.topology(t.msg, t.polygons)
    except GeoError.Small:
        raise GeoError.Small(msg="filling_area too small to offset one time")
    "rotate -angle"
    rotPolys = GeomAIgo.rotatePolygons(polygon, -angle)

    ca = ClipperAdaptor()
    pco = pyclipper.PyclipperOffset()
    pco.AddPaths(ca.toPaths(rotPolys), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(0)
    Polys = ca.toPolys(solution)
    # utils.path_show(Polys)
    total_ys = utils.To_Intersect_ys(Polys, interval)
    a = argregin.ArgRegion(Polys, interval, coefficient, 1e-6)

    try:
        child_regions = a.Split()
    except GeoError.Small as s:
        print(s.msg)
        pass

    pathsop = utils.Generate_local_Z(child_regions, interval, total_ys)

    if len(child_regions) == 1:
        # child_regions = GeomAIgo.rotatePolygons(child_regions, angle)
        for path in pathsop:
            for point in path.points:
                point.z = z
        return pathsop[0]  # pathsop is closed
    else:
        GC = Global_Connect(pathsop, interval)
        LC = GC.connectable()
        for point in LC.points:
            point.z = z
        LC = GeomAIgo.rotatePolygons([LC], angle)[0]
        # utils.path_show([LC])
        return LC
