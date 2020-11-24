# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
from ..meshes.simplexmesh import SimplexMesh


#=================================================================#
class Fem(object):
    def __init__(self, mesh=None):
        if mesh is not None: self.setMesh(mesh)
    def setMesh(self, mesh):
        self.mesh = mesh
    def downWind(self, v):
        # v is supposed RT0
        dim, ncells, fofc, sigma = self.mesh.dimension, self.mesh.ncells, self.mesh.facesOfCells, self.mesh.sigma
        vp = np.maximum(v[fofc]*sigma, 0)
        vps = vp.sum(axis=1)
        if not np.all(vps > 0): raise ValueError(f"{vps=}\n{vp=}")
        vp /= vps[:,np.newaxis]
        lamd = (np.ones(ncells)[:,np.newaxis] - vp)/dim
        # vm = np.minimum(v[fofc]*sigma, 0)
        # vms = vm.sum(axis=1)
        # if not np.all(vms < 0): raise ValueError(f"{vms=}\n{vm=}")
        # vm /= vms[:,np.newaxis]
        # lamd = vm
        # print(f"{v[fofc].shape} {lamd2.shape=}")
        points, simplices = self.mesh.points, self.mesh.simplices
        xd = np.einsum('nji,nj -> ni', points[simplices], lamd)
        # if not np.allclose(lamd, lamd2): print(f"{lamd=} {lamd2=}")
        return xd, lamd




# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
