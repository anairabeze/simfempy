import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)
import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.heat import Heat
from simfempy.tools.comparemethods import CompareMethods
import simfempy.applications.problemdata

#----------------------------------------------------------------#
def test_analytic(createMesh, h, data, exactsolution="Linear", fems=['p1'], dirichlets=['new','trad'], stabs=['supg','lps']):
    if isinstance(fems,str): fems = [fems]
    if isinstance(dirichlets,str): methods = [dirichlets]
    if isinstance(stabs,str): stab = [stabs]
    sims = {}
    for fem in fems:
        for dirichlet in dirichlets:
            for stab in stabs:
                kwargs = {'problemdata':data, 'fem':fem, 'dirichlet':dirichlet, 'stab':stab, 'masslumpedbdry':False}
                kwargs['exactsolution'] = exactsolution
                kwargs['random'] = False
                kwargs['linearsolver'] = 'pyamg'
                # kwargs['linearsolver'] = 'umf'
                name = fem
                if len(dirichlets)>1: name += dirichlet
                if len(stabs)>1: name += stab
                sims[name] = Heat(**kwargs)
    comp = CompareMethods(sims, createMesh=createMesh, plot=False)
    result = comp.compare(h=h)
    # global refine
    # comp = CompareMethods(sims, createMesh=createMesh, h=0.5)
    # result = comp.compare(niter=3)

#----------------------------------------------------------------#
def test(dim, exactsolution='Linear', fems=['p1','cr1'], dirichlets=['new','trad'], stabs=['supg','lps']):
    data = simfempy.applications.problemdata.ProblemData()
    data.params.scal_glob['kheat'] = 0.01
    # data.params.fct_glob['convection'] = ["y", "-x"]
    # data.params.fct_glob['convection'] = dim*["1"]
    # stabs = ['centered']
    if dim==1:
        createMesh = testmeshes.unitline
        colors = [10000,10001]
        colorsrob = []
        colorsneu = [10001]
        h = [1, 0.5, 0.25, 0.125, 0.06, 0.03, 0.02]
    elif dim==2:
        createMesh = testmeshes.unitsquare
        colors = [1000, 1001, 1002, 1003]
        colorsrob = []
        # colorsneu = [1001, 1002]
        #TODO P1 wrong if several neumann bdries meet
        colorsneu = [1001]
        h = [1, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015]
    else:
        createMesh = testmeshes.unitcube
        colors = [100, 101, 102, 103, 104, 105]
        colorsrob = []
        colorsneu = [102,105]
        h = [1.0, 0.5, 0.25, 0.125]
    colorsdir = [col for col in colors if col not in colorsrob and col not in colorsneu]
    # print(f"{colorsdir=}")
    # from simfempy.meshes import plotmesh
    # plotmesh.meshWithBoundaries(createMesh(h[0]))
    if exactsolution == "Constant" or exactsolution == "Linear": h = h[:3]
    data.bdrycond.set("Dirichlet", colorsdir)
    data.bdrycond.set("Neumann", colorsneu)
    data.bdrycond.set("Robin", colorsrob)
    for col in colorsrob: data.bdrycond.param[col] = 11.
    data.postproc.set(name='bdrymean', type='bdry_mean', colors=colorsneu)
    data.postproc.set(name='bdrynflux', type='bdry_nflux', colors=colorsdir[0])
    test_analytic(createMesh=createMesh, h=h, data=data, exactsolution=exactsolution, fems=fems, dirichlets=dirichlets, stabs=stabs)

#================================================================#
if __name__ == '__main__':
    # test(dim=2, exactsolution = 'Quadratic', fems=['p1','cr1'], stabs=['supg','lps'], dirichlets=['new'])
    # test(dim=2, exactsolution = 'Linear', fems=['p1','cr1'], stabs=['supg'], dirichlets=['new'])
    test(dim=2, exactsolution = 'Quadratic', fems=['p1','cr1'], stabs=['supg'])