import math
epsilon = 1e-7
epsilonSquare = epsilon*epsilon


class Point3D:

    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = x, y, z, w

    def __str__(self):
        return "Point3D:%s,%s,%s" % (self.x, self.y, self.z)

    def clone(self):
        return Point3D(self.x, self.y, self.z, self.w)

    def pointTo(self,other):
        return Vector3D(other.x-self.x, other.y-self.y, other.z-self.z)

    def translate(self,vec):
        self.x, self.y, self.z =self.x+vec.dx, self.y+vec.dy, self.z+vec.dz

    def translated(self,vec):
        return Point3D(self.x+vec.dx, self.y+vec.dy, self.z+vec.dz)

    def multiplied(self,m):
        x = self.x * m.a[0][0] + self.y * m.a[1][0] + self.z * m.a[2][0] + self.w * m.a[3][0]
        y = self.x * m.a[0][1] + self.y * m.a[1][1] + self.z * m.a[2][1] + self.w * m.a[3][1]
        z = self.x * m.a[0][2] + self.y * m.a[1][2] + self.z * m.a[2][2] + self.w * m.a[3][2]
        return Point3D(x, y, z)
    def multiply(self,m):
        self.x = self.x * m.a[0][0] + self.y * m.a[1][0] + self.z * m.a[2][0] + self.w * m.a[3][0]
        self.y = self.x * m.a[0][1] + self.y * m.a[1][1] + self.z * m.a[2][1] + self.w * m.a[3][1]
        self.z = self.x * m.a[0][2] + self.y * m.a[1][2] + self.z * m.a[2][2] + self.w * m.a[3][2]


    def distance(self,other):
        return self.pointTo(other).length()

    def distanceSquare(self,other):
        return self.pointTo(other).lengthSquare()

    def middle(self,other):
        return Point3D((self.x+other.x)/2, (self.y+other.y)/2, (self.z+other.z)/2)

    def isCoincide(self,other,dis2=epsilonSquare):
        return True if self.pointTo(other).lengthSquare()< dis2 else False

    def isIdentical(self,other):
        return True if self.x==other.x and self.y==other.y and self.z==other.z else False

    def __add__(self, vec):
        return self.translated(vec)

    def __sub__(self, other):
        return other.pointTo(self)if isinstance(other,Point3D)else self.translated(other.reversed())

    def __mul__(self, m):
        return self.multiplied(m)
    "可能关于z有bug"
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        hash_tuple = (self.x, self.y, self.z)

        return hash(hash_tuple)




class Vector3D:
    def __init__(self, dx=0.0, dy=0.0, dz=0.0, dw=0.0):
        self.dx, self.dy, self.dz, self.dw =dx, dy, dz, dw

    def __str__(self):
        return 'Vector3D:%s,%s,%s' % (self.dx, self.dy, self.dz)

    def clone(self):
        return Vector3D(self.dx, self.dy, self.dz, self.dw)

    def reverse(self):
        self.dx, self.dy, self.dz = -self.dx, -self.dy, -self.dz

    def reversed(self):
        return Vector3D(-self.dx, -self.dy, -self.dz)

    def dotProduct(self, vec):
        return self.dx * vec.dx +self.dy * vec.dy +self.dz *vec.dz

    def crossProduct(self, vec):
        dx=self.dy * vec.dz - self.dz * vec.dy
        dy=self.dz * vec.dx - self.dx * vec.dz
        dz=self.dx * vec.dy - self.dy * vec.dx
        return Vector3D(dx, dy, dz)
    def amplify(self,f):
        self.dx, self.dy, self.dz = self.dx * f, self.dy * f, self.dz * f

    def amplified(self,f):
        return Vector3D(self.dx * f, self.dy * f, self.dz * f)

    def lengthSquare(self):
        return self.dx * self.dx + self.dy * self.dy + self.dz * self.dz

    def length(self):
        return math.sqrt(self.lengthSquare())

    def normalize(self):
        len = self.length()
        self.dx, self.dy, self.dz = self.dx/len ,self.dy/len , self.dz/len

    def normalized(self):
        len = self.length()
        return Vector3D(self.dx/len, self.dy/len, self.dz/len)

    def isZeroVector(self):
        return self.lengthSquare() < 1.0e-8

    def multiplied(self,m):
        dx = self.dx * m.a[0][0] + self.dy * m.a[1][0] + self.dz * m.a[2][0] + self.dw * m.a[3][0]
        dy = self.dx * m.a[0][1] + self.dy * m.a[1][1] + self.dz * m.a[2][1] + self.dw * m.a[3][1]
        dz = self.dx * m.a[0][2] + self.dy * m.a[1][2] + self.dz * m.a[2][2] + self.dw * m.a[3][2]
        return Vector3D(dx, dy, dz)

    def isParallel(self, other):
        return self.crossProduct(other).isZeroVector()

    def getAngle(self, vec):
        v1, v2 = self.normalized(), vec.normalized()
        dotPro = v1.dotProduct(v2)
        if dotPro > 1: dotPro = 1
        elif dotPro < -1: dotPro = -1
        return math.acos(dotPro)

    def getAngle2D(self):
        rad = self.getAngle(Vector3D(1, 0, 0))
        if self.dy < 0: rad = math.pi * 2.0 - rad
        return rad

    def __add__(self, other):
        return Vector3D(self.dx + other.dx, self.dy + other.dy, self.dz + other.dz)

    def __sub__(self,other):
        return self + other.reversed()
    def __mul__(self, other):
        return self.multiplied(other)


class Matrix3D:
    def __init__(self):
         self.a = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    def __str__(self):
         return 'Matrix3D:\n%s\n%s\n%s\n%s' % (self.a[0], self.a[1], self.a[2],self.a[3])

    def makeIdentical(self):
         self.a = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def multiplied(self, other):
         m = Matrix3D()
         for i in range(4):
             for j in range(4):
                 m.a[i][j] = self.a[i][0] * other.a[0][j] + self.a[i][1] * other.a[1][j] + m.a[i][2] * other.a[2][j]
                 + self.a[i][3] * other.a[3][j]
         return m

    def getDeterminant(self):pass

    def getReverseMatrix(self):pass
    @staticmethod
    def createTranslateMatrix(dx,dy,dz):
         m=Matrix3D()
         m.a[3][0], m.a[3][1], m.a[3][2] = dx, dy,dz
         return m
    @staticmethod
    def createScalMatrix(sx, sy,sz):
         m=Matrix3D()
         m.a[0][0], m.a[1][1], m.a[2][2] = sx,sy,sz
         return m
    @staticmethod
    def createRotateMatrix(axis, angle):
         m= Matrix3D()
         sin, cos = math.sin(angle), math.cos(angle)
         if axis == "X" or axis =="x":
             m.a[1][1], m.a[1][2], m.a[2][1], m.a[2][2] = cos,sin,-sin,cos
         elif axis == "Y" or axis =='y':
             m.a[0][0], m.a[0][2], m.a[2][0], m.a[2][2] = cos,-sin,sin,cos
         elif axis == "Z" or axis =='z':
             m.a[0][0], m.a[0][1], m.a[1][0], m.a[1][1] = cos ,sin, -sin ,cos
         return m
    @staticmethod
    def createMirrorMatrix(point, normal):pass

    def __mul__(self, other):
         return self.multiplied(other)
    def __add__(self, other):
         pass
    def __sub__(self, other):
         pass






















