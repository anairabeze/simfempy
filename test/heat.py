assert __name__ == '__main__'
import os, sys
import numpy as np
simfempypath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(simfempypath)

import simfempy
import pygmsh
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- #
def createMesh():
    lcar = 0.1
    rect = [-1, 1, -1, 1]
    geom = pygmsh.built_in.Geometry()
    holescoord = []
    xc, yc, r = -0.5, -0.5, 0.2
    holescoord.append([[xc-r, yc-r], [xc-r, yc+r], [xc+r, yc+r], [xc+r, yc-r]])
    xc, yc, r = 0.4, 0.4, 0.15
    holescoord.append([[xc-r, yc-r], [xc-r, yc+r], [xc+r, yc+r], [xc+r, yc-r]])
    xholes = []
    for hole in holescoord:
        xholes.append(np.insert(np.array(hole), 2, 0, axis=1))
    holes = []
    for i, xhole in enumerate(xholes):
        # hole = geom.add_polygon(X=xhole, lcar=lcar,make_surface=False)
        hole = geom.add_polygon(X=xhole, lcar=lcar,make_surface=True)
        geom.add_physical(hole.surface, label=200 + i)
        # for j in range(len(hole.lines)): geom.add_physical(hole.lines[j], label=2000 + 10*i + j)
        holes.append(hole)
    p = geom.add_rectangle(*rect, 0.0, lcar=0.1, holes=holes)
    geom.add_physical(p.surface, label=100)
    for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=1000 + i)
    mesh = pygmsh.generate_mesh(geom)
    return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)

# ---------------------------------------------------------------- #
def createData():
    data = simfempy.applications.problemdata.ProblemData()
    bdrycond =  data.bdrycond
    bdrycond.type[1000] = "Neumann"
    bdrycond.type[1001] = "Dirichlet"
    bdrycond.type[1002] = "Neumann"
    bdrycond.type[1003] = "Dirichlet"
    bdrycond.fct[1000] = lambda x,y,z, nx, ny, nz: 0
    bdrycond.fct[1002] = lambda x,y,z, nx, ny, nz: 100
    bdrycond.fct[1001] = bdrycond.fct[1003] = lambda x,y,z: 120
    postproc = data.postproc
    postproc.type['bdrymean_low'] = "bdrymean"
    postproc.color['bdrymean_low'] = [1000]
    postproc.type['bdrymean_up'] = "bdrymean"
    postproc.color['bdrymean_up'] = [1002]
    postproc.type['fluxn'] = "bdrydn"
    postproc.color['fluxn'] = [1001, 1003]
    def kheat(label, x, y, z):
        if label==100: return 0.0001
        return 1000.0
    data.params.fct_glob["kheat"] = kheat
    return data

# ---------------------------------------------------------------- #
def test(mesh, problemdata):
    fem = 'p1' # or fem = 'cr1
    heat = simfempy.applications.heat.Heat(problemdata=problemdata, fem=fem, plotk=True)
    heat.setMesh(mesh)
    point_data, cell_data, info = heat.solve()
    print(f"fem={fem} {info['timer']}")
    print(f"postproc: {info['postproc']}")
    simfempy.meshes.plotmesh.meshWithData(mesh, point_data=point_data, cell_data=cell_data, title=fem)
    plt.show()

# ================================================================c#

mesh = createMesh()
simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
problemdata = createData()
print("problemdata", problemdata)
problemdata.check(mesh)
test(mesh, problemdata)
