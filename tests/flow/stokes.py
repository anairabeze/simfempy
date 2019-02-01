from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

from simfempy import applications

#----------------------------------------------------------------#
def test_analytic(problem="Analytic_Sinus", geomname = "unitsquare", verbose=5):
    import simfempy.tools.comparerrors
    postproc = {}
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    if geomname == "unitsquare":
        problem += "_2d"
        ncomp = 2
        h = [2, 1, 0.5, 0.25]
        bdrycond.type[1000] = "Dirichlet"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Dirichlet"
        bdrycond.type[1003] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:1000,1002"
        postproc['bdrydn'] = "bdrydn:1001,1003"
    if geomname == "unitcube":
        problem += "_3d"
        ncomp = 3
        h = [2, 1, 0.5, 0.25]
        bdrycond.type[100] = "Dirichlet"
        bdrycond.type[105] = "Dirichlet"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Dirichlet"
        bdrycond.type[104] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:100,105"
        postproc['bdrydn'] = "bdrydn:101,102,103,104"
    compares = {}
    for fem in ['cr1']:
        compares[fem] = applications.stokes.Stokes(problem=problem, bdrycond=bdrycond, postproc=postproc,fem=fem, ncomp=ncomp)
    comp = simfempy.tools.comparerrors.CompareErrors(compares, verbose=verbose)
    result = comp.compare(geomname=geomname, h=h)
    res = {}
    res['L2-V-cr1'] = result[3]['error']['L2-V']['cr1']
    res['L2-P-cr1'] = result[3]['error']['L2-P']['cr1']
    return res


#================================================================#
if __name__ == '__main__':
    # result = test_analytic(problem="Analytic_Linear", geomname = "unitcube", verbose=2)
    result = test_analytic(problem="Analytic_Quadratic")
    print("result", result)
