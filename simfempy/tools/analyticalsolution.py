# -*- coding: utf-8  -*-
"""

"""

import numpy as np
import sympy


#=================================================================#
class AnalyticalSolution():
    """
    computes numpy vectorized functions for the function and its dericatives up to two
    for a given expression, derivatives computed with sympy
    """
    def __repr__(self):
        return f"dim={self.dim} expr={str(self.expr)}"
    def __call__(self, x):
        return self.fct(*x)
    def __init__(self, dim, expr):
        if expr.find('y') != -1 or expr.find('z') != -1:
            expr = expr.replace('x', 'x0')
            expr = expr.replace('y', 'x1')
            expr = expr.replace('z', 'x2')
        if dim==1 and expr.find('x0') == -1:
            expr = expr.replace('x', 'x0')
        print("expr",expr)
        self.dim, self.expr = dim, expr
        symbc = ""
        for i in range(dim):
            symbc += f"x{i},"
        symbc = symbc[:-1]
        print("symbc", symbc, "self", self)
        s = sympy.symbols(symbc)
        print("s", s)
        self.fct = np.vectorize(sympy.lambdify(symbc,expr))
        self.fct_x = []
        for i in range(dim):
            if dim==1: fx = sympy.diff(expr, s)
            else: fx = sympy.diff(expr, s[i])
            self.fct_x.append(np.vectorize(sympy.lambdify(symbc, fx),otypes=[float]))
    def d(self, i, x):
        return self.fct_x[i](*x)
    def x(self, x):
        return self.fct_x[0](*x)
    def y(self, x):
        return self.fct_x[1](*x)
    def z(self, x):
        return self.fct_x[2](*x)


#=================================================================#
class AnalyticalSolution3d():
    """
    computes numpy vectorized functions for the function and its dericatives up to two
    for a given expression, derivatives computed with sympy
    """
    def __init__(self, expr):
        (x, y, z) = sympy.symbols('x,y,z')
        self.expr = expr
        self.fct = np.vectorize(sympy.lambdify('x,y,z',expr))
        fx = sympy.diff(expr, x)
        fy = sympy.diff(expr, y)
        fz = sympy.diff(expr, z)
        fxx = sympy.diff(fx, x)
        fxy = sympy.diff(fx, y)
        fxz = sympy.diff(fx, z)
        fyy = sympy.diff(fy, y)
        fyz = sympy.diff(fy, z)
        fzz = sympy.diff(fz, z)
        self.fct_x = np.vectorize(sympy.lambdify('x,y,z', fx),otypes=[float])
        self.fct_y = np.vectorize(sympy.lambdify('x,y,z', fy),otypes=[float])
        self.fct_z = np.vectorize(sympy.lambdify('x,y,z', fz),otypes=[float])
        self.fct_xx = np.vectorize(sympy.lambdify('x,y,z', fxx),otypes=[float])
        self.fct_xy = np.vectorize(sympy.lambdify('x,y,z', fxy),otypes=[float])
        self.fct_xz = np.vectorize(sympy.lambdify('x,y,z', fxz),otypes=[float])
        self.fct_yy = np.vectorize(sympy.lambdify('x,y,z', fyy),otypes=[float])
        self.fct_yz = np.vectorize(sympy.lambdify('x,y,z', fyz),otypes=[float])
        self.fct_zz = np.vectorize(sympy.lambdify('x,y,z', fzz),otypes=[float])
    def __repr__(self):
        return str(self.expr)
    def __call__(self, x, y=0, z=0):
        return self.fct(x,y, z)
    def x(self, x, y=0, z=0):
        return self.fct_x(x,y, z)
    def y(self, x, y, z=0):
        return self.fct_y(x,y, z)
    def z(self, x, y, z):
        return self.fct_z(x,y, z)
    def xx(self, x, y=0, z=0):
        return self.fct_xx(x,y,z )
    def yy(self, x, y, z=0):
        return self.fct_yy(x,y,z)
    def zz(self, x, y, z):
        return self.fct_zz(x,y,z)
    def d(self, i, x, y, z):
        if i==2:    return self.fct_z(x,y,z)
        elif i==1:  return self.fct_y(x,y,z)
        return self.fct_x(x,y,z)
    def dd(self, i, j, x, y, z=0):
        if i==2:    
            if j==2:    return self.fct_zz(x,y,z)
            elif j==1:  return self.fct_yz(x,y,z)
            return self.fct_xz(x,y,z)
        if i==1:    
            if j==2:    return self.fct_yz(x,y,z)
            elif j==1:  return self.fct_yy(x,y,z)
            return self.fct_xy(x,y,z)
        if j==2:    return self.fct_xz(x,y,z)
        elif j==1:  return self.fct_xy(x,y,z)
        return self.fct_xx(x,y,z)

#=================================================================#
def analyticalSolution(function, dim, ncomp=1, random=True):
    """
    defines some analytical functions to be used in validation

    returns analytical function (if ncomp==1) or list of analytical functions (if ncomp>1)

    parameters:
        function: name of function
        ncomp: size of list
        random: use random coefficients
    """
    solexact = []
    if not random:
        from itertools import permutations
        perm = list(permutations((3.3, 2.2, 1.1)))
    for i in range(ncomp):
        if random:
            p = (4 * np.random.rand() - 2) / 3
            q = (4 * np.random.rand() - 2) / 3
            r = (4 * np.random.rand() - 2) / 3
        else:
            p, q, r = perm[i%ncomp]
        if dim==1:
            if function == 'Constant':
                fct = '{:3.1f}'.format(p)
            elif function == 'Linear':
                fct = '{:3.1f} * x'.format(p)
            elif function == 'Quadratic':
                fct = '{:3.1f}*x*x'.format(p)
            elif function == 'Sinus':
                fct = 'sin({:3.1f}*x)'.format(p)
            else:
                raise NotImplementedError("unknown analytic solution: {}".format(function))
        elif dim == 2:
            if function == 'Constant':
                fct = '{:3.1f}'.format(p)
            elif function == 'Linear':
                fct = '{:3.1f} * x + {:3.1f} * y'.format(p, q)
            elif function == 'Quadratic':
                fct = '{:3.1f}*x*x + {:3.1f}*y*y'.format(p, q)
            elif function == 'Sinus':
                fct = 'sin({:3.1f}*x + {:3.1f}*y)'.format(p, q)
            else:
                raise NotImplementedError("unknown analytic solution: {}".format(function))
        elif dim==3:
            if function == 'Constant':
                fct = '{:3.1f}'.format(p)
            elif function == 'Linear':
                fct = '{:3.1f}*x + {:3.1f}*y + {:3.1f}*z'.format(p, q, r)
            elif function == 'Quadratic':
                fct = '{:3.1f}*x*x + {:3.1f}*y*y + {:3.1f}*z*z'.format(p, q, r)
            elif function == 'Sinus':
                fct = 'sin({:3.1f}*x + {:3.1f}*y + {:3.1f}*z)'.format(p, q, r)
            else:
                raise NotImplementedError("unknown analytic solution: {}".format(function))
        else:
            raise NotImplementedError("dim: {}".format(dim))
        solexact.append(AnalyticalSolution3d(fct))
    if ncomp==1: return solexact[0]
    return solexact
        

# ------------------------------------------------------------------- #
if __name__ == '__main__':
    def test1D():
        u = AnalyticalSolution3d('x*x')
        x = np.linspace(0,2,3)
        print("u.xx(3)", u.xx(3))
        print("u.x(x)", x, u.x(x))
        print("u(x)", x, u(x))
    def test3D():
        u = AnalyticalSolution3d('x*x+2*y*y+ 4*z*z')
        for (x,y,z) in [(0,0,0), (1,1,1), (1,0,1), (0,1,0)]:
            print('Quadratic function', u, ' in x,y,z=:',x,y,z, ' equals: ', u(x,y,z))
            print('grad=', u.x(x,y,z), u.y(x,y,z), u.z(x,y,z))
            print('laplace=', u.xx(x,y,z) + u.yy(x,y,z) + u.zz(x,y,z))

    # test2D()
    # test3D()
    # u = AnalyticalSolution(dim=2, expr='x*y')
    # print("u(1,2)", u((1,2)))
    # print("Du(1,2)", u.x((1,2)), u.y((1,2)))
    # x = np.linspace(0, 2, 3)
    # y = np.linspace(1, 2, 2)
    # coord = np.meshgrid(x,y)
    # print("u(coord)", u.x(coord))
    u = AnalyticalSolution(dim=1, expr='x*x')
    print("u(2)", u([2]) )
    x = np.meshgrid(np.linspace(0, 2, 3))
    print("x", x, "u(x)", u(x), "u.x(x)", u.x(x))
    # print("u.x(x)", x, u.x(x))
    # print("u.xx(x)", x, u.xx(x))

    
    
