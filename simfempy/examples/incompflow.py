assert __name__ == '__main__'
# in shell
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygmsh
from simfempy.applications.stokes import Stokes
from simfempy.applications.navierstokes import NavierStokes
from simfempy.applications.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.meshes import plotmesh

# ================================================================c#
def main(testcase='drivenCavity'):
    testcases = ['drivenCavity', 'backwardFacingStep', 'poiseuille']
    # create mesh and data
    if testcase=='drivenCavity':
        mesh, data = drivenCavity(h=0.2, mu=0.00025)
    elif testcase=='backwardFacingStep':
        mesh, data = backwardFacingStep(h=0.1)
    elif testcase=='poiseuille':
        mesh, data = poiseuille(h=0.1)
    else:
        raise ValueError(f"test case must be in {testcases=}")
   # plotmesh.meshWithBoundaries(mesh)
    # create application
    # stokes = Stokes(mesh=mesh, problemdata=data, linearsolver='iter_gmres_10')
    stokes = Stokes(mesh=mesh, problemdata=data, linearsolver='umf')
    # stokes = NavierStokes(mesh=mesh, problemdata=data, linearsolver='iter_gmres')
    # stokes = NavierStokes(mesh=mesh, problemdata=data, linearsolver='iter_gcrotmk')
    # stokes = NavierStokes(mesh=mesh, problemdata=data, linearsolver='umf')
    result = stokes.solve()
    print(f"{result.info['timer']}")
    print(f"postproc:")
    for p, v in result.data['global'].items(): print(f"{p}: {v}")
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)
    plotmesh.meshWithBoundaries(mesh, fig=fig, outer=outer[0])
    plotmesh.meshWithData(mesh, data=result.data, title="Stokes", fig=fig, outer=outer[1])
    plotmesh.meshWithData(mesh, title="Stokes", fig=fig, outer=outer[2],
                          quiver_data={"V":list(result.data['point'].values())})
    plt.show()


# ================================================================c#
def drivenCavity(h=0.1, mu=0.001):
    with pygmsh.geo.Geometry() as geom:
        ms = [h*v for v in [1.,1.,0.2,0.2]]
        p = geom.add_rectangle(xmin=0, xmax=1, ymin=0, ymax=1, z=0, mesh_size=ms)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
        mesh = geom.generate_mesh()
    data = ProblemData()
    # boundary conditions
    # data.bdrycond.set("Dirichlet", [1000, 1001, 1002, 1003])
    data.bdrycond.set("Dirichlet", [1001, 1002, 1003])
    data.bdrycond.set("Navier", [1000])
    # data.bdrycond.fct[1002] = lambda x, y, z: np.vstack((np.ones(x.shape[0]),np.zeros(x.shape[0])))
    data.bdrycond.fct[1002] = [lambda x, y, z: 1, lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    #TODO pass ncomp with mesh ?!
    data.ncomp = 2
    return SimplexMesh(mesh=mesh), data


# ================================================================ #
def backwardFacingStep(h=0.2, mu=0.02):
    with pygmsh.geo.Geometry() as geom:
        X = []
        X.append([-1.0, 1.0])
        X.append([-1.0, 0.0])
        X.append([0.0, 0.0])
        X.append([0.0, -1.0])
        X.append([3.0, -1.0])
        X.append([3.0, 1.0])
        p = geom.add_polygon(points=np.insert(np.array(X), 2, 0, axis=1), mesh_size=h)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
        mesh = geom.generate_mesh()
    data = ProblemData()
    # boundary conditions
    data.bdrycond.set("Dirichlet", [1000, 1001, 1002, 1003])
    # data.bdrycond.set("Dirichlet", [1000, 1001, 1002, 1003, 1005])
    data.bdrycond.set("Neumann", [1004])
    data.bdrycond.set("Navier", [1005])
    # data.bdrycond.fct[1000] = [lambda x, y, z: 1,  lambda x, y, z: 0]
    data.bdrycond.fct[1000] = [lambda x, y, z: y*(1-y),  lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.params.scal_glob["navier"] = 0.01
    #TODO pass ncomp with mesh ?!
    data.ncomp = 2
    return SimplexMesh(mesh=mesh), data
# ================================================================ #
def poiseuille(h= 0.1, mu=0.02):
    with pygmsh.geo.Geometry() as geom:
        #ms = [h*v for v in [1.,1.,0.2,0.2]]
        ms = h
        p = geom.add_rectangle(xmin=-1.0, xmax=3.0, ymin=-1.0, ymax=1.0, z=0, mesh_size=ms)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
        mesh = geom.generate_mesh()
    data = ProblemData()
   # boundary conditions choice :

    # """
    # %%%%%%%%%%%%%%%%%%%%% Dirichlet %%%%%%%%%%%%%%%%%%    y = \frac{d}{2}
    # %%%%                                             %
    #  %%Lambda%       below the solution      %Neumann%    y = 0  
    # %%%%                                             %
    # %%%%%%%%%%%%%%%%%%%%% Dirichlet %%%%%%%%%%%%%%%%%%    y = \frac{-d}{2}
    # xmin---------------------------------------------xmax
    # """
    # data.bdrycond.set("Dirichlet", [1000, 1002, 1003])
    # data.bdrycond.set("Neumann", [1001])

    # #Analytical Poiseuille's Solution
    # #$\Lambda(x,y,z) = \Lambda(y) = \frac{C}{2\mu}[(\frac{d}{2})^{2} - y^{2}]$
    # # With $C = constant = \frac{P_{in} - P_{out}}{|xmin-xmax|}$

    # data.bdrycond.fct[1003] = [lambda x, y, z: -y**2 + 0.5,  lambda x, y, z: 0]

# #----------------------------------------------------------------------
#     """
#     %%%%%%%%%%%%%%%%%%%%% Dirichlet %%%%%%%%%%%%%%%%%%    y = \frac{d}{2}
#     %%%%                                             %
#      %%Lambda%       below the solution      %Neumann%    y = 0  
#     %%%%                                             %
#     %%%%%%%%%%%%%%%%%%%%% Navier    %%%%%%%%%%%%%%%%%%    y = \frac{-d}{2}
#     xmin---------------------------------------------xmax
#     """
#     data.bdrycond.set("Dirichlet", [1002, 1003])
#     data.bdrycond.set("Neumann", [1001])
#     data.bdrycond.set("Navier", [1000])

#     #navier_slip_boundary
#     # data.bdrycond.fct[1000] = [lambda x, y, z:  x, lambda x, y, z: 0]

#     #Analytical Poiseuille's Solution
#     #$\Lambda(x,y,z) = \Lambda(y) = \frac{C}{2\mu}[(\frac{d}{2})^{2} - y^{2}] + \frac{v_{0}(y)}{2}[\frac{d}{2} - y]$
#     # With $C = constant = \frac{P_{in} - P_{out}}{|xmin-xmax|}$ and $v_{0}(y) = data.bdrycond.fct[1000]$

#     data.bdrycond.fct[1003] = [lambda x, y, z: -y**2 + 0.5 - y/2,  lambda x, y, z: 0]

#----------------------------------------------------------------------
    """
    %%%%%%%%%%%%%%%%%%%%% Navier    %%%%%%%%%%%%%%%%%%    y = \frac{d}{2}
    %%%%                                             %
     %%Lambda%       below the solution      %Neumann%    y = 0  
    %%%%                                             %
    %%%%%%%%%%%%%%%%%%%%% Navier    %%%%%%%%%%%%%%%%%%    y = \frac{-d}{2}
    xmin---------------------------------------------xmax
    """
    data.bdrycond.set("Dirichlet", [1003])
    data.bdrycond.set("Neumann", [1001])
    data.bdrycond.set("Navier", [1000, 1002])

    #navier_slip_boundary
    # data.bdrycond.fct[1000] = [lambda x, y, z:  1, lambda x, y, z: 0]
    # data.bdrycond.fct[1002] = [lambda x, y, z:  2, lambda x, y, z: 0]

    #Analytical Poiseuille's Solution
    # $\Lambda(x,y,z) = \Lambda(y)  = \frac{C}{2\mu}[(\frac{d}{2})^{2} - y^{2}] + (a(y) - b(y))[\frac{y}{d} - \frac{1}{2}] + a(y)$
    # With $C = constant = \frac{P_{in} - P_{out}}{|xmin-xmax|}$ and $a(y) = data.bdrycond.fct[1002], b(y)= data.bdrycond.fct[1000]$

    data.bdrycond.fct[1003] = [lambda x, y, z: -y**2 + y + 2,  lambda x, y, z: 0]
    #---------------------------------------------------------------------------
    # parameters
    data.params.scal_glob["mu"] = mu
    data.params.scal_glob["navier"] = 0.01
    #TODO pass ncomp with mesh ?!
    data.ncomp = 2
    return SimplexMesh(mesh=mesh), data

# ================================================================c#
main()
