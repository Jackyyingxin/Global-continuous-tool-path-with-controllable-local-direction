import GeomBase
import Polyline
import pathengine


class CFS:
    def __init__(self, polygon, interval):
        self.polygon = polygon
        self.interval = interval

    def init_spiral(self):
        pe = pathengine.pathEngine()
        spiral, zigzag = pe.fill_spiral_in_connected_region(polys, -interval, len_of_t)

        spiral_path = Polyline.Polyline()
        for array in spiral:
            spiral_path.points.append(GeomBase.Point3D(array[0], array[1], key))
        if spiral_path.points[0] == spiral_path.points[-1]:
            spiral_path.points.pop()
        # path_show([spiral_path])
        spiral_path.points.reverse()
        spiral_path = build_connect_tree.resmaple_about_interval(interval, spiral_path, 0)
        spiral_path, spiral_path_tensor = poly_processing(spiral_path, interval, 10)
        total_path = copy.deepcopy(spiral_path)
        total_path_tensor = copy.deepcopy(spiral_path_tensor)


