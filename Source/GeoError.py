class BeforDnpC(Exception):
    def __init__(self):
        pass
class Turnpt_E(Exception):
    def __init__(self):
        pass


class lengtherror(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg


class Decomposition(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class outpoly_too_small(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

class inner_poly_small(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class Small(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg


class Pocket(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class topology(Exception):
    def __init__(self, msg, polygons, len_inner = None):
        self.msg = msg
        self.polygons = polygons
        self.len_inner = len_inner

    def __str__(self):
        return self.msg

class Divide(Exception):
    def __init__(self, msg, polygons):
        self.msg = msg
        self.polygons = polygons


    def __str__(self):
        return self.msg

class interval_error(Exception):

    def __init__(self, msg):
        self.msg = msg


    def __str__(self):
        return self.msg


class splitter(Exception):


    def __init__(self, msg):
        self.msg = msg


    def __str__(self):
        return self.msg

class Connect_Error(Exception):

    def __init__(self, msg, path):
        self.msg = msg
        self.path = path

    def __str__(self):
        return self.msg

class Unprintable(Exception):
    def __init__(self, msg, path):
        self.msg = msg
        self.path = path

    def __str__(self):
        return self.msg




