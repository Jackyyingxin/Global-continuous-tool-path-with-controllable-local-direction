import Erroe
import GeomBase
from VtkAdaptor import VtkAdaptor
import utils


class Global_Connect:

    def __init__(self, paths, interval):
        self.T = None
        self.paths = paths
        self.interval = interval

    def connectable(self):
        layer_local_connected_path = self.paths[0]

        self.paths.pop(0)
        while len(self.paths) > 0:
            horizon_i = self.get_overlap_Y(layer_local_connected_path, 1)
            index = {}
            horizon_js = {}
            for j in range(len(self.paths)):
                # if j==len(self.paths)-1:
                # va = VtkAdaptor()
                # va.drawPolyline(layer_local_connected_path).GetProperty().SetColor(1, 0, 0)
                # va.drawPolyline(self.paths[j]).GetProperty().SetColor(1, 0, 0)
                # va.display()
                # print("a")

                horizon_j = self.get_overlap_Y(self.paths[j])
                index_j = self.get_MIN_y_distance(horizon_i, horizon_j)
                horizon_js[j] = horizon_j
                if len(index_j) != 0:
                    index[j] = index_j
            if len(index) == 0:
                raise Erroe.Connect_Error("error", self.paths[j])
            index = sorted(index.items(), key=lambda x: abs(x[1][0]))
            index = index[0]
            j = index[0]
            horizon_j = horizon_js[j]
            index = index[1]
            Y_I = index[1][0][0]
            Y_J = index[1][0][1]
            X_MAX_I = horizon_i[Y_I][index[1][1][0]][0]
            X_MAX_J = horizon_j[Y_J][index[1][1][1]][0]
            X_MIN_I = horizon_i[Y_I][index[1][1][0]][1]
            X_MIN_J = horizon_j[Y_J][index[1][1][1]][1]
            I_INDEX_MAX = utils.findindex(GeomBase.Point3D(X_MAX_I, Y_I), layer_local_connected_path)
            J_INDEX_MAX = utils.findindex(GeomBase.Point3D(X_MAX_J, Y_J), self.paths[j])
            I_INDEX_MIX = utils.findindex(GeomBase.Point3D(X_MIN_I, Y_I), layer_local_connected_path)
            J_INDEX_MIX = utils.findindex(GeomBase.Point3D(X_MIN_J, Y_J), self.paths[j])
            # print(j)
            # va = VtkAdaptor()
            # temp1 = va.drawPolyline(layer_local_connected_path)
            # temp1.GetProperty().SetColor(1, 0, 0)
            # temp1.GetProperty().SetLineWidth(2)
            # temp2 = va.drawPolyline(self.paths[j])
            # temp2.GetProperty().SetColor(0, 0, 1)
            # temp2.GetProperty().SetLineWidth(2)
            # va.drawPoint(layer_local_connected_path.points[I_INDEX_MAX], 0.1).GetProperty().SetColor(1, 0, 0)
            # va.drawPoint(layer_local_connected_path.points[I_INDEX_MIX], 0.1).GetProperty().SetColor(1, 0, 0)
            # va.drawPoint(self.paths[j].points[J_INDEX_MAX], 0.1).GetProperty().SetColor(0, 0, 1)
            # va.drawPoint(self.paths[j].points[J_INDEX_MIX], 0.1).GetProperty().SetColor(0, 0, 1)
            # va.display()
            if abs(Y_I - Y_J) < 0.01 * self.interval:
                "如果改成0.5很有可能会出现自交"

                L_index_I = I_INDEX_MIX
                R_index_I = I_INDEX_MAX
                L_index_J = J_INDEX_MIX
                R_index_J = J_INDEX_MAX



            else:
                path_I_insert_L, path_I_insert_R, path_J_insert_L, path_J_insert_R = utils.insert(
                    layer_local_connected_path, self.paths[j], self.interval, [I_INDEX_MAX, I_INDEX_MIX],
                    [J_INDEX_MAX, J_INDEX_MIX])
                L_index_I = utils.findindex(path_I_insert_L, layer_local_connected_path)
                R_index_I = utils.findindex(path_I_insert_R, layer_local_connected_path)
                L_index_J = utils.findindex(path_J_insert_L, self.paths[j])
                R_index_J = utils.findindex(path_J_insert_R, self.paths[j])

            if Y_I - Y_J > 0:
                layer_local_connected_path.points = utils.replaceL(R_index_I,
                                                                                layer_local_connected_path.points)

                self.paths[j].points = utils.replaceL(L_index_J, self.paths[j].points)

                layer_local_connected_path = utils.contourconnect(layer_local_connected_path,
                                                                               self.paths[j])
            else:
                layer_local_connected_path.points = utils.replaceL(L_index_I,
                                                                                layer_local_connected_path.points)

                self.paths[j].points = utils.replaceL(R_index_J, self.paths[j].points)

                layer_local_connected_path = utils.contourconnect(self.paths[j],
                                                                               layer_local_connected_path)
            self.paths.pop(j)

        return layer_local_connected_path

    def insert_strange_path(self, path1, paht2):

        pass

    def get_overlap_Y(self, path, flag=0):
        # if flag==1:
        # va = VtkAdaptor()
        # for tpath in self.paths:
        #    va.drawPolyline(tpath).GetProperty().SetColor(1, 0, 0)
        # va.drawPolyline(path).GetProperty().SetColor(0, 0, 1)
        # va.drawPolyline(layer_local_connected_path).GetProperty().SetColor(1, 0, 0)
        # va.drawPolyline(self.paths[j]).GetProperty().SetColor(1, 0, 0)
        # va.drawPoint(layer_local_connected_path.points[I_INDEX_MAX]).GetProperty().SetColor(1, 0, 0)
        # va.drawPoint(layer_local_connected_path.points[I_INDEX_MIX]).GetProperty().SetColor(1, 0, 0)
        # va.drawPoint(self.paths[j].points[J_INDEX_MAX]).GetProperty().SetColor(1, 0, 0)
        # va.drawPoint(self.paths[j].points[J_INDEX_MIX]).GetProperty().SetColor(1, 0, 0)
        # va.display()
        horizon_path1 = {}
        for i in range(len(path.points) - 1):
            if path.points[i].y == path.points[i + 1].y:
                # if flag==1:
                # va.drawPoint(path.points[i]).GetProperty().SetColor(1, 0, 0)
                # va.drawPoint(path.points[i + 1]).GetProperty().SetColor(1, 0, 0)
                y = path.points[i].y
                MAX_X = max(path.points[i].x, path.points[i + 1].x)
                MIN_X = min(path.points[i].x, path.points[i + 1].x)
                if y not in horizon_path1.keys():
                    horizon_path1[y] = []
                    horizon_path1[y].append([MAX_X, MIN_X])

                else:
                    horizon_path1[y].append([MAX_X, MIN_X])
        # if flag == 1:
        #    va.display()
        return horizon_path1

    def get_MIN_y_distance(self, horizon, horizon_j):
        total_dif = []

        for i in horizon.keys():
            dif = {}
            for j in horizon_j.keys():
                for k in range(len(horizon[i])):
                    for l in range(len(horizon_j[j])):
                        if not (horizon[i][k][1] >= horizon_j[j][l][0] or horizon[i][k][0] <= horizon_j[j][l][1]):
                            "have overlap"
                            dif[i - j] = [[i, j], [k, l]]

            if len(dif) != 0:
                dif = sorted(dif.items(), key=lambda x: abs(x[0]))
                total_dif.append(dif[0])
        if len(total_dif) != 0:

            total_dif.sort(key=lambda x: abs(x[0]))
            return total_dif[0]
        else:
            return []

    def insert(self):
        pass
