# Finite Element Analysis of 3D truss and frame
# Author: Qiren Gao, Michigan State University
# Latest modifications made by Abhiroop Ghosh, Michigan State University

import numpy as np


def run_fea(coordinates, connectivity, fixed_nodes, load_nodes, force, density, elastic_modulus,
            structure_type='truss'):

    preXr = coordinates
    Xr = np.transpose(preXr)
    preconnec0 = (connectivity[:, :2] - 1).astype(int)
    connec0 = np.transpose(preconnec0)
    numel0 = connec0.shape[1]
    nx = Xr.shape[1]

    x0 = Xr[:3, :]
    R = connectivity[:, 2]

    radius_nel = np.zeros([numel0, 1])
    lengths = np.zeros([numel0, 1])
    for nel in range(numel0):
        nodes = connec0[:, nel]
        radius_nel[nel, 0] = R[nel]
        xnel = x0[:, nodes]
        dx = xnel[0, 1] - xnel[0, 0]
        dy = xnel[1, 1] - xnel[1, 0]
        dz = xnel[2, 1] - xnel[2, 0]
        lnel = np.sqrt(dx*dx + dy*dy + dz*dz)
        lengths[nel, 0] = lnel

    Fx0 = force[0]
    Fy0 = force[1]
    Fz0 = force[2]

    numel = numel0
    connec = np.copy(connec0)
    x = np.copy(x0)
    ndf = 6
    numnodes = x.shape[1]
    neq = numnodes * ndf
    # tdofs = np.array([np.arange(1, neq, 6), np.arange(2, neq, 6), np.arange(3, neq, 6)]).reshape(3 * numnodes, 1)
    # tdofs = np.transpose(tdofs).reshape(3 * numnodes, 1)

    #==========================================================================
    # Boundary Conditions: Fixed nodes and dof
    # Type of b.c.:
    #      1,1,1 means "x,y,z disp fixed"
    #      0,1,1 means "y,z   disp fixed"
    #      1,0,0 means "x     disp fixed"   etc...
    #==========================================================================

#     xmin= min(x')
#     xmax= max(x')
    xmin = np.min(x, axis=0)
    xmax = np.max(x, axis=0)
    # eps = np.linalg.norm(xmax - xmin) * 1e-5
    #     fixednodes=find(abs(x(1,:)-min(x(1,:)))<eps)
    fixed_nodes = (fixed_nodes - 1).reshape(1, -1).astype(int)
    fixeddof = np.ones([fixed_nodes.shape[1], 6])

    # ==========================================================================
    # f truss, remove rotations
    # ==========================================================================
    if structure_type == 'truss':
        tmp = np.tile(np.array([0, 0, 0, 1, 1, 1]), (numnodes, 1))
        fixeddof = np.append(fixeddof, tmp, axis=0)
        fixed_nodes = np.append(fixed_nodes, np.arange(0, numnodes)).reshape(1, -1).astype(int)

    freedofs, fixeq = setbc(numnodes, ndf, fixed_nodes, fixeddof)

    load_nodes = np.transpose(load_nodes).astype(int)

    nFx = 6 * (load_nodes - 1)
    nFy = 6 * (load_nodes - 1) + 1
    nFz = 6 * (load_nodes - 1) + 2

    elastimod = elastic_modulus * np.ones([1, numel])
    shearmod = elastimod / (2 * (1 + 0.33))

    skBy, skBz, skA, skT = buildKBeamLatticeUnscaled(numel, x, connec, elastimod, shearmod)
    if structure_type == 'truss':
        skBy = np.zeros(skBy.shape)
        skBz = np.zeros(skBz.shape)
        skT = np.zeros(skT.shape)

    # ==========================================================================
    #  precompute   loads
    # ==========================================================================
    F = np.zeros([neq, 1])
    F[nFx, 0] = Fx0 / load_nodes.shape[1]
    #     F(nFx2,1)=Fx2/length(loadn2)
    F[nFy, 0] = Fy0 / load_nodes.shape[1]
    F[nFz, 0] = Fz0 / load_nodes.shape[1]

    xval = 2 * radius_nel

    areas = np.pi * xval * xval / 4
    bendingIZ = np.pi * xval ** 4 / 64
    bendingIY = np.pi * xval ** 4 / 64
    torsionJ = bendingIY + bendingIZ
    # =========================== FEM analysis only ============================
    sk = assembleKBeamLattice(numel, neq, connec, areas, bendingIZ, bendingIY, torsionJ, skBy, skBz, skA, skT)
    
    # Deformation of nodes for every dof
    U = np.zeros([neq, 1])

    sk_subset = np.zeros([freedofs.shape[0], freedofs.shape[0]])
    for i in range(freedofs.shape[0]):
        sk_subset[i, :] = sk[freedofs[i], freedofs]
    # U[freedofs, :] = np.linalg.inv(sk[freedofs, freedofs]) * F[freedofs, :]
    U[freedofs, :] = np.matmul(np.linalg.inv(sk_subset), F[freedofs, :])
#     if any(isnan(U))
#         compliance = 10000
#     else
#         compliance=0
#         compliance = compliance +U(:,1)'*F(:,1)
# #         compliance = compliance +abs(U(:,1))'*abs(F(:,1)))
# #         compliance = compliance + nansum(abs(U(:, 1) ./ F(:, 1)))
#     end
    # TODO: Make compliance metric selectable by the user
    compliance = np.sum(U[:, 0] * F[:, 0])
    # compliance = np.sum(np.abs(U[:, 0] * F[:, 0]))
    final_volume = np.sum(lengths * areas)

#     csvwrite(final_compliance_file_name,compliance)
#     csvwrite(final_volume_file_name,final_volume)
    weight = final_volume * density

    U = U.flatten()
    x0_new = np.zeros(x0.shape)
    # Get the new coordinates of each node based on the displacements U
    for ii in range(nx):
        x0_new[0:3, ii] = x0[0:3, ii] + U[6 * ii:6*ii + 3]

    # x0_new[0:3, np.arange(0, nx)] = x0[0:3, np.arange(0, nx)] + U[6 * np.arange(0, nx):6*np.arange(0, nx) + 3]
#     x0_new(1:3, 1:nx) = x0(1:3, 1:nx) + U(6 * (1:nx-1) + 1: 6 * (1:nx-1) + 3)
    
    new_lengths = np.zeros([numel, 1])
    strain = np.zeros([numel, 1])
    stress = np.zeros([numel, 1])
    # Get the new lengths of members from the connectivity matrix connec0
    for ii in range(numel0):
        elem_nodes = connec0[:, ii]
        new_lengths[ii] = np.linalg.norm(x0_new[:, elem_nodes[0]] - x0_new[:, elem_nodes[1]])
        strain[ii] = (new_lengths[ii] - lengths[ii]) / lengths[ii]
        stress[ii] = elastic_modulus * strain[ii]

#     elem_nodes = connec0(:, 1:numel0)'
#     new_lengths(1:numel0) = norm(x0_new(:, elem_nodes(1)) - x0_new(:, elem_nodes(2)))
#     strain(1:numel0) = (new_lengths(1:numel0) - lengths(1:numel0)) ./ lengths(1:numel0)
#     stress(1:numel0) = elastic_modulus * strain(1:numel0)
    
    x0_new = np.transpose(x0_new)

    return weight, compliance, stress.flatten(), strain.flatten(), U.flatten(), x0_new
    

def setbc(numnodes, ndf, fixednodes, fixeddof):
    numfix = fixednodes.shape[1]

    # Total number of equations
    neq = numnodes * ndf

    # Apply  fixed boundary condition.  Build list of fixed dof.
    ndoffixed = np.sum(fixeddof[:numfix, :ndf])
    fixeq = np.zeros([1, ndoffixed.astype(int)])
    j = 0
    for i in range(numfix):
        nodei = fixednodes[0, i]
        for idof in range(ndf):
            id = fixeddof[i, idof]
            if id == 1:
                gieq = nodei * ndf + idof
                fixeq[0, j] = gieq
                j = j + 1

    alldofs = np.arange(0, neq)
    freedofs = np.setdiff1d(alldofs, fixeq)

    return freedofs, fixeq


def gradientsKBeamLattice(numel, numdesvar, mcons, xval, connec, U, lengths, skBy, skBz, skA, skT):

    dfdx = np.zeros(mcons, numdesvar)
    df0dx = np.zeros(numdesvar, 1)

    for nel in range(numel):
        nodes = np.transpose(connec[:, nel])
        ndofs = np.transpose(np.concatenate([6 * (nodes - 1) + 1,
                                     6 * (nodes - 1) + 2,
                                     6 * (nodes - 1) + 3,
                                     6 * (nodes - 1) + 4,
                                     6 * (nodes - 1) + 5,
                                     6 * nodes], axis=0).reshape(12, 1))
        ue = U[ndofs, :]
        xv = xval[nel]
        dkby = skBy[:, nel].reshape(12, 12)
        dkbz = skBz[:, nel].reshape(12, 12)
        dka = skA[:, nel].reshape(12, 12)
        dkt = skT[:, nel].reshape(12, 12)
        dk = np.pi * ((4 * (xv ** 3) / 64) * (dkby + dkbz + 2*dkt) + 2*xv / 4*dka)
        df0dx[nel, 0] = - np.transpose(ue[:, 0]) * dk * ue[:, 0]
        dfdx[0, nel] = np.pi * lengths(nel) * xv / 2

    return df0dx, dfdx


def assembleKBeamLattice(numel, neq, connec, areas, bendingIZ, bendingIY, torsionJ, skBy, skBz, skA, skT):

    sk = np.zeros([neq, neq])
    for nel in range(numel):
        #     anel=areas(nel)
        #     jnel=torsionJ(nel)
        #     inelZ=bendingIZ(nel)
        #     inelY=bendingIY(nel)
        nodes = np.transpose(connec[:, nel])

        ndofs = np.transpose(np.concatenate([[6 * nodes + 1],
                                             [6 * nodes + 2],
                                             [6 * nodes + 3],
                                             [6 * nodes + 4],
                                             [6 * nodes + 5],
                                             [6 * (nodes + 1)]], axis=0)).flatten() - 1

        # sk[ndofs, ndofs] = (sk[ndofs, ndofs]
        #                     + (bendingIY[nel]*skBy[:, nel]).reshape(12, 12)
        #                     + (bendingIZ[nel]*skBz[:, nel]).reshape(12, 12)
        #                     + (areas[nel]*skA[:, nel]).reshape(12, 12)
        #                     + (torsionJ[nel]*skT[:, nel]).reshape(12, 12))
        # print(ndofs)
        for i in range(ndofs.shape[0]):
            sk[ndofs[i], ndofs] = (sk[ndofs[i], ndofs]
                                   + (bendingIY[nel]*skBy[:, nel].reshape(12, 12)[i, :])
                                   + (bendingIZ[nel]*skBz[:, nel].reshape(12, 12)[i, :])
                                   + (areas[nel]*skA[:, nel].reshape(12, 12)[i, :])
                                   + (torsionJ[nel]*skT[:, nel]).reshape(12, 12)[i, :])

    return sk


def buildKBeamLatticeUnscaled(numel, x, connec, elastimod, shearmod):

    skBy = np.zeros([144, numel])
    skBz = np.zeros([144, numel])
    skA = np.zeros([144, numel])
    skT = np.zeros([144, numel])
    # rotArray=sparse(144,numel)

    for nel in range(numel):

        # Information about element "nel"
        nodes = np.transpose(connec[:, nel])
        xnel = x[:, nodes]

        enel = elastimod[0, nel]
        gnel = shearmod[0, nel]
        kbbz, kbby, kaxial, ktorsion, rot12 = beam3d(xnel, 1, 1, 1, 1, enel, gnel, 0)

        skBy[:, nel] = (np.matmul(np.matmul(np.transpose(rot12), kbby), rot12)).reshape(144,)
        skBz[:, nel] = (np.matmul(np.matmul(np.transpose(rot12), kbbz), rot12)).reshape(144,)
        skA[:, nel] = (np.matmul(np.matmul(np.transpose(rot12), kaxial), rot12)).reshape(144,)
        skT[:, nel] = (np.matmul(np.matmul(np.transpose(rot12), ktorsion), rot12)).reshape(144,)
        #     rotArray(:,nel)=reshape(rot12,144,1)

    return skBy, skBz, skA, skT


def beam3d(xnel, anel, jnel, inelZ, inelY, enel, gnel, tnel):
    dofz = np.array([2, 6, 8, 12]) - 1
    dofy = np.array([3, 5, 9, 11]) - 1
    dofa = np.array([1, 7]) - 1
    dofj = np.array([4, 10]) - 1
    dx = xnel[0, 1] - xnel[0, 0]
    dy = xnel[1, 1] - xnel[1, 0]
    dz = xnel[2, 1] - xnel[2, 0]
    lnel = np.sqrt(dx*dx + dy*dy + dz*dz)
    l = dx / lnel
    m = dy / lnel
    n = dz / lnel

    D = np.sqrt(l**2 + m**2)
    if D > 0:
        rot33 = np.array([[l, m, n],  [-m/D, l/D, 0], [-l*n/D, -m*n/D, D]])
    else:
        rot33 = np.array([[0, 0, np.sign(n)],  [-1, 0, 0],  [0, -np.sign(n), 0]])

    rot12 = np.zeros([12, 12])
    rot12[:3, :3] = rot33
    rot12[3:6, 3:6] = rot33
    rot12[6:9, 6:9] = rot33
    rot12[9:12, 9:12] = rot33

    kz = enel * inelZ / lnel**3
    ky = enel * inelY / lnel**3
    kg = (tnel / lnel / 30)
    kbbZ = np.array([[12,       6*lnel,    -12,      6*lnel],
                    [6*lnel,   4 * lnel**2,  -6*lnel,  2 * lnel**2],
                    [-12,      -6*lnel,     12,     -6 * lnel],
                    [6*lnel,   2 * lnel**2,  -6*lnel,  4 * lnel**2]])

    kbbY = np.array([[12,       -6 * lnel,    -12,      -6*lnel],
                    [-6*lnel,    4 * lnel**2,   6*lnel,  2 * lnel**2],
                    [-12,        6 * lnel,     12,      6*lnel],
                    [-6*lnel,     2 * lnel**2,   6*lnel,  4 * lnel**2]])

    kbbz= kz * kbbZ
    kbby= ky * kbbY
    # kggZ=kg*[    36,    3*lnel,    -36,       3*lnel ...
    #     3*lnel,    4*lnel^2, -3*lnel,   -lnel^2 ...
    #     -36,   -3*lnel,     36,      -3*lnel ...
    #     3*lnel,   -lnel^2,   -3*lnel,   4*lnel^2]
    #
    # kggY=kg*[   36,    -3*lnel,    -36,      -3*lnel ...
    #     -3*lnel,    4*lnel^2,  3*lnel,   -lnel^2 ...
    #     -36,     3*lnel,     36,       3*lnel ...
    #     -3*lnel,   -lnel^2,    3*lnel,   4*lnel^2]
    # kgg=kg*[36, 3*lnel, -36, 3*lnel 3*lnel, 4*lnel^2, -3*lnel, -lnel^2 -36, -3*lnel, 36, -3*lnel 3*lnel, -lnel^2, -3*lnel, 4*lnel^2]

    kaxial = (enel * anel / lnel) * np.array([[1, -1], [-1, 1]])
    ktorsion = (gnel * jnel / lnel) * np.array([[1, -1], [-1, 1]])

    kbbz12 = np.zeros([12, 12])
    for i in range(kbbZ.shape[0]):
        kbbz12[dofz[i], dofz] = kbbz[i, :]
    # kbbz12[dofz, dofz] = kbbz

    kbby12 = np.zeros([12, 12])
    for i in range(kbby.shape[0]):
        kbby12[dofy[i], dofy] = kbby[i, :]
    # kbby12[dofy, dofy] = kbby

    kaxial12 = np.zeros([12, 12])
    for i in range(kaxial.shape[0]):
        kaxial12[dofa[i], dofa] = kaxial[i, :]
    # kaxial12[dofa, dofa] = kaxial

    ktorsion12 = np.zeros([12, 12])
    for i in range(ktorsion.shape[0]):
        ktorsion12[dofj[i], dofj] = ktorsion[i, :]
    # ktorsion12[dofj, dofj] = ktorsion

    return kbbz12, kbby12, kaxial12, ktorsion12, rot12
