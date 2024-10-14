# -*- coding: utf-8 -*-

from abaqus import *
from abaqusConstants import *
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import numpy as np
from scipy.spatial import KDTree
from numpy import pi
import odbAccess
import numpy as np
import shutil
import time
import os
import csv
import sys


def material_properties():
    # Material properties A (Bambulabs PLA-CF) -------------------------------------------------------------------------
    YoungsModulus = 2800
    PoissonsRatio = 0.35
    YieldStress = 10
    UltimateStress = 46
    FractureStrain = 0.01
    PlasticTable = ((YieldStress, 0.0),
                    (29.0, 0.0005), (32.0, 0.001), (39.0, 0.002), (45.5, 0.006),
                    (UltimateStress, FractureStrain))
    mdl.Material(name='Material-A')
    mdl.materials['Material-A'].Elastic(table=((YoungsModulus, PoissonsRatio),))
    mdl.materials['Material-A'].Plastic(table=PlasticTable)

    # Material properties B (Bambulabs PETG)--------------------------------------------------------------------------
    YoungsModulus = 1400
    PoissonsRatio = 0.33
    YieldStress = 7
    UltimateStress = 33
    FractureStrain = 0.0082
    PlasticTable = ((YieldStress, 0.0),
                    (20.0, 0.0005), (22.0, 0.001), (27.0, 0.002), (32, 0.006),
                    (UltimateStress, FractureStrain))
    mdl.Material(name='Material-B')
    mdl.materials['Material-B'].Elastic(table=((YoungsModulus, PoissonsRatio),))
    mdl.materials['Material-B'].Plastic(table=PlasticTable)

    '''# Material properties A (Ultimaker PLA)----------------------------------------------------------------------------------
    YoungsModulus = 2800
    PoissonsRatio = 0.35
    YieldStress = 10
    UltimateStress = 46
    FractureStrain = 0.01
    PlasticTable = ((YieldStress, 0.0),
                    (29.0, 0.0005), (32.0, 0.001), (39.0, 0.002), (45.5, 0.006),
                    (UltimateStress, FractureStrain))
    mdl.Material(name='Material-A')
    mdl.materials['Material-A'].Elastic(table=((YoungsModulus, PoissonsRatio),))
    mdl.materials['Material-A'].Plastic(table=PlasticTable)

    # Material properties B (Ultimaker TPU) ----------------------------------------------------------------------------------
    YoungsModulus = 100
    PoissonsRatio = 0.48
    YieldStress = 3
    UltimateStress = 100
    FractureStrain = 1.3
    PlasticTable = ((YieldStress, 0.0),
                    (8, 0.1), (11.5, 0.2), (14.5, 0.3),
                    (17.5, 0.4), (22, 0.5), (27, 0.6),
                    (36, 0.7), (55, 0.8), (72, 0.9),
                    (UltimateStress, FractureStrain))
    mdl.Material(name='Material-B')
    mdl.materials['Material-B'].Elastic(table=((YoungsModulus, PoissonsRatio),))
    mdl.materials['Material-B'].Plastic(table=PlasticTable)'''

    return


def euc_dist(a, b):  # Euclidean distance (1*2,1*2)
    return np.linalg.norm(a - b)


def mapping(mtx_coord, tlr_v):  # Mapping function (4*2 or 3*1,f,f) , tlr_e
    if mtx_coord.shape == (4, 2):  # 2d mapping of 3d model
        # Check if node P is an one-to-one node (near a node)
        if euc_dist(mtx_coord[0], mtx_coord[1]) < tlr_v:
            w = [1, 0, 0]
        else:
            mtx_agmt = np.vstack((mtx_coord[3, :] - mtx_coord[1, :],
                                  mtx_coord[3, :] - mtx_coord[2, :],
                                  mtx_coord[3, :] - mtx_coord[0, :])).T
            w = np.linalg.solve(mtx_agmt[:, :2], mtx_agmt[:, -1])
            w = np.append(w, 1 - w[0] - w[1])
    else:  # 1d mapping of 2d model
        # w = np.zeros((2, 1))
        d0 = mtx_coord[2, 0] - mtx_coord[1, 0]
        d1 = mtx_coord[0, 0] - mtx_coord[1, 0]
        d2 = mtx_coord[2, 0] - mtx_coord[0, 0]
        # Check if node P is a vertex node (near a node)
        if np.abs(d1) < tlr_v:
            w = [1, 0]
        else:
            w = [d2 / d0, d1 / d0]
    return w


def identifunc(coord_lmt, coord_sgl, tlr_s):  # Identify function (2*2 or 2*3, 1*2 or 1*3)
    if coord_lmt.shape[1] != len(coord_sgl):
        return 'Coordinates and limitations do not match in dimension'
    else:
        dim = len(coord_sgl)  # dimension of model
        bcon = np.zeros(dim, dtype=int)  # beacon vector for single coordinate
        for i in range(dim):
            if coord_sgl[i] > coord_lmt[0, i] - tlr_s:  # upper limit
                bcon[i] = 1
            if coord_sgl[i] < coord_lmt[1, i] + tlr_s:  # lower limit
                bcon[i] = -1
        return bcon


def collinear_3p(mtx_coord):  # three points collinear (3*2)
    k1 = (mtx_coord[0, :] - mtx_coord[1, :]) / euc_dist(mtx_coord[0, :], mtx_coord[1, :])
    k2 = (mtx_coord[0, :] - mtx_coord[2, :]) / euc_dist(mtx_coord[0, :], mtx_coord[2, :])
    if euc_dist(np.abs(k1), np.abs(k2)) <= 0.00001:
        return True
    else:
        return False


def nns(coord, tgt, rng, dim, dir):  # Nearest Neighbor Searching function (k*3, int lst, int lst, int lst, 2 or 3, int)
    if len(tgt) == 0 or len(rng) == 0:
        return 'invalid target or range input'
    else:
        # coord matrix of range nodes
        coord_rng = np.empty(shape=(0, 3))
        for j in rng:
            coord_rng = np.vstack([coord_rng, coord[j, :]])  # node coordinates in range set
        tree = KDTree(coord_rng)
        # index matrix of knn
        idx = np.empty(shape=(0, dim))
        for i in tgt:
            coord_tgt = coord[i, :]  # single node coordinates in target set
            dist, idx_rng = tree.query(coord_tgt, k=dim)
            if dim == 3:
                coord_nn = coord_rng[idx_rng.T, :]
                coord_nn = np.delete(coord_nn, dir, axis=1)
                k_x = dim
                clnr = collinear_3p(coord_nn)
                while clnr:
                    k_x += 1
                    dist_x, idx_rng_x = tree.query(coord_tgt, k=k_x)
                    idx_rng[2] = idx_rng_x[-1]
                    coord_nn = coord_rng[idx_rng.T, :]
                    clnr = collinear_3p(coord_nn)
            idx = np.vstack([idx, idx_rng])
        # convert local index into global index
        for m in range(len(tgt)):
            for d in range(dim):
                idx[m, d] = int(rng[int(idx[m, d])])
    return idx


def sum_quot(a, b):  # sum of quotients
    return a/b + b/a


def sum_reci(a, b):  # sum of reciprocal
    return 1/a + 1/b


def IsClose(a, b, t):
    # a, b: numbers to be judged (float)
    # t: tolerance (float)
    if b - t <= a <= b + t:
        return True
    else:
        return False


def vector_angle(a, b):  # angle of two vectors
    if type(a) != np.array:
        a = np.asarray(a)
    if type(b) != np.array:
        b = np.asarray(b)
    return np.arccos(a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b)))


def IsCollinear(a, b, t):
    # a, b: vectors to be judged (tuple)
    # t: angle tolerance (float)
    if vector_angle(a, b) <= t or vector_angle(a, b) >= pi-t:
        return True
    else:
        return False


def IsCSVector(a, t):
    # a: vector to be judged (tuple)
    # t: angle tolerance (float)
    v_x, v_y, v_z = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    if IsCollinear(a, v_x, t) or IsCollinear(a, v_y, t) or IsCollinear(a, v_z, t):
        return True
    else:
        return False


def GetSurfaceFromNodeSet(inputNodes):
    '''
    Return MeshFaceArray containing meshfaces that only touch nodes in InputNodes

    Inputs:
    inputNodes -      Mesh node array

    Output:
    MeshFaceArray containing faces that have all nodes in inputNodes
    '''
    nodesInSet = set([(a.label, a.instanceName) for a in inputNodes])
    FacesOnSurf = []
    FacesTouchingSurf = {}
    for n in inputNodes:
        for face in n.getElemFaces():
            if n.instanceName:
                FacesTouchingSurf[(face.label, face.face, n.instanceName)] = face
                continue
            FacesTouchingSurf[(face.label, face.face)] = face

    for f in FacesTouchingSurf.values():
        IsOnSurf = True
        for n in f.getNodes():
            if (n.label, n.instanceName) not in nodesInSet:
                IsOnSurf = False
                break
                continue
        if len(f.getElements()) == 2:
            IsOnSurf = False
        if IsOnSurf == True:
            FacesOnSurf.append(f)
            continue
    MyMFA = mesh.MeshFaceArray(FacesOnSurf)
    return MyMFA


def ExportData(job_name):
    odb = odbAccess.openOdb(r'C:\\temp\\' + job_name + '.odb')  # ,readOnly=
    # OUTPUT: Von Mises Stress & Element Volume ====================================================================
    fop = odb.steps['Step-1'].frames[-1].fieldOutputs

    elm_qty_a = len(odb.rootAssembly.instances['PART-A-1'].elements)
    for i in range(elm_qty_a):
        sgm_a = fop['S'].values[i].mises  # maxPrincipal
        v_a = fop['EVOL'].values[i].data  # elementVolume
        with open('C:\\temp\\TPMS-A.txt', 'a') as f:
            f.write('%.6f %.6f\n' % (sgm_a, v_a))
        f.close()

    elm_qty_b = len(odb.rootAssembly.instances['Part-B-1'].elements)
    for i in range(elm_qty_b):
        sgm_b = fop['S'].values[i + elm_qty_a].mises  # maxPrincipal
        v_b = fop['EVOL'].values[i + elm_qty_a].data  # elementVolume
        with open('C:\\temp\\TPMS-B.txt', 'a') as f:
            f.write('%.6f %.6f\n' % (sgm_b, v_b))
        f.close()

    # OUTPUT: Strain Energy ==================================================================================
    hop = odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs
    ew_total = hop['ALLWK'].data[-1][-1]

    with open('C:\\temp\\TPMS-C.txt', 'a') as f:
        f.write('%.6f\n' % ew_total)
    f.close()


def remove_prev_files(job):
    list_suffix = ['inp', 'sta', 'msg', 'log', 'lck', 'odb']
    for suffix in list_suffix:
        prev_file = './' + job + '.' + suffix
        if os.path.exists(prev_file):
            os.remove(prev_file)
    if os.path.exists('result.txt'):
        os.remove('result.txt')


work_dir = 'C:\\temp\\'
model_name = 'intlck-tpms'
job_name = 'job-' + model_name
sta_file = job_name + '.sta'
odb_filepath = work_dir + job_name + '.odb'
lck_filepath = work_dir + job_name + '.lck'

mesh_name = 'tpms'
mesh_path = 'C:\\temp\\TPMS11\\'
trimesh_name = mesh_path + mesh_name + '-tri-'
tetmesh_name = mesh_path + mesh_name + '-tet-'

StepTime = 1.

ucs = 10  # unit cell size
thk_l = ucs*3
u_tens = 10  # displacement of tensile

T_v = 0.0001  # tolerance of vertex
T_e = 0.001  # tolerance of edge
T_s = 0.01  # tolerance of surface
T_a = 0.1  # tolerance of angle

tet = False

# ====================================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Pre-Processing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ====================================================================================
# close opened odb
if len(session.odbs.keys()) == 0:
    pass
else:
    for key in session.odbs.keys():
        session.odbs[key].close()

remove_prev_files(job_name)

Mdb()
session.viewports['Viewport: 1'].setValues(displayedObject=None)
mdb.models.changeKey(fromName='Model-1', toName=model_name)
mdl = mdb.models[model_name]

# Import Part-Tri-A & B
mdl.PartFromInputFile(inputFileName=trimesh_name+'A.inp')
mdl.parts.changeKey(fromName='PART-1', toName='Part-Tri-A')
p_tri_a = mdl.parts['Part-Tri-A']
mdl.PartFromInputFile(inputFileName=trimesh_name+'B.inp')
mdl.parts.changeKey(fromName='PART-1', toName='Part-Tri-B')
p_tri_b = mdl.parts['Part-Tri-B']
print('>>> Tri-meshes imported...')

if tet:
    # Import Part-Tet-A
    mdl.PartFromInputFile(inputFileName=tetmesh_name+'A.inp')
    mdl.parts.changeKey(fromName='PART-1', toName='Part-Tet-A')
    p_a = mdl.parts['Part-Tet-A']
    session.viewports['Viewport: 1'].setValues(displayedObject=p_a)
    # Import Part-Tet-B
    mdl.PartFromInputFile(inputFileName=tetmesh_name+'B.inp')
    mdl.parts.changeKey(fromName='PART-1', toName='Part-Tet-B')
    p_b = mdl.parts['Part-Tet-B']
    session.viewports['Viewport: 1'].setValues(displayedObject=p_b)
    print('>>> Tet-meshes imported...')
else:
    # Convert Part-Tri-A to Part-Tet-A
    p_a = mdl.Part(name='Part-Tet-A', objectToCopy=p_tri_a)
    p_a.generateMesh(elemShape=TET)
    p_a.setValues(space=THREE_D, type=DEFORMABLE_BODY)
    session.viewports['Viewport: 1'].setValues(displayedObject=p_a)
    # Convert Part-Tri-B to Part-Tet-B
    p_b = mdl.Part(name='Part-Tet-B', objectToCopy=p_tri_b)
    p_b.generateMesh(elemShape=TET)
    p_b.setValues(space=THREE_D, type=DEFORMABLE_BODY)
    session.viewports['Viewport: 1'].setValues(displayedObject=p_b)
    print('>>> Tet-meshes converted...')

# Create Set
p_a.Set(elements=p_a.elements, name='Set-ALL')
p_b.Set(elements=p_b.elements, name='Set-ALL')
print('>>> Sets created...')

# Create Surf to Part-Tet-A&B=============================================================================
prt_tri, prt_tet = [p_tri_a, p_tri_b], [p_a, p_b]
for i_prt in range(2):
    lst_MN_Surf = []  # list of surface mesh nodes
    lst_MN_Bdry = []  # list of contact mesh nodes
    BB_high = prt_tri[i_prt].elements.getBoundingBox()['high']
    BB_low = prt_tri[i_prt].elements.getBoundingBox()['low']

    for MN_tri in prt_tri[i_prt].nodes:
        coord = MN_tri.coordinates  # tuple
        MN_tet = prt_tet[i_prt].nodes.getByBoundingSphere(center=coord, radius=T_v)
        lst_MN_Surf += MN_tet
        isOnBdry = False
        for i_dir in range(3):
            if IsClose(coord[i_dir], BB_high[i_dir], T_s) or IsClose(coord[i_dir], BB_low[i_dir], T_s):
                isOnBdry = True
            if i_dir == 2 and (IsClose(coord[i_dir], -thk_l/2, T_s) or IsClose(coord[i_dir], thk_l/2, T_s)):
                isOnBdry = True  # End surface of Z direction
        if isOnBdry:
            lst_MN_Bdry += MN_tet

    MNA_Surf = mesh.MeshNodeArray(lst_MN_Surf)  # surface mesh node array
    MFA_Surf = GetSurfaceFromNodeSet(MNA_Surf)  # surface mesh face array
    prt_tet[i_prt].Surface(face1Elements=MFA_Surf, name='Surf-all')

    lst_MF_Surf = list(MFA_Surf)

    MNA_Bdry = mesh.MeshNodeArray(lst_MN_Bdry)  # contact mesh node array
    MFA_Bdry = GetSurfaceFromNodeSet(MNA_Bdry)  # boundary mesh face array

    lst_MF_Bdry = []
    for MF_tet in MFA_Bdry:
        if IsCSVector(MF_tet.getNormal(), T_a):
            lst_MF_Bdry.append(MF_tet)
    MFA_Bdry = mesh.MeshFaceArray(lst_MF_Bdry)

    lst_MF_Cont = []
    for MF_tet in MFA_Surf:
        if MF_tet not in lst_MF_Bdry:
            lst_MF_Cont.append(MF_tet)
    MFA_Cont = mesh.MeshFaceArray(lst_MF_Cont)

    prt_tet[i_prt].Surface(face1Elements=MFA_Bdry, name='Surf-Bdry')
    prt_tet[i_prt].Surface(face1Elements=MFA_Cont, name='Surf-Cont')
print('>>> Surfaces created...')


"""
# assign elemType
elemType1 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT,
                          elemDeletion=ON)
p_a.setElementType(regions=(p_a.elements,), elemTypes=(elemType1,))
p_b.setElementType(regions=(p_b.elements,), elemTypes=(elemType1,))
print('>>> Element types assigned...')
"""

# PROPERTY =============================================================================
material_properties()
# Section A
mdl.HomogeneousSolidSection(name='Section-A', material='Material-A', thickness=None)
p_a.SectionAssignment(region=p_a.sets['Set-ALL'], sectionName='Section-A', offset=0.0,
                      offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
# Section B
mdl.HomogeneousSolidSection(name='Section-B', material='Material-B', thickness=None)
p_b.SectionAssignment(region=p_b.sets['Set-ALL'], sectionName='Section-B', offset=0.0,
                      offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
print('>>> Material properties assigned...')


# Prepare sets and parameters ----------------------------------------------------------
asm = mdl.rootAssembly
asm.DatumCsysByDefault(CARTESIAN)
asm.Instance(name='Part-A-1', part=p_a, dependent=ON)
asm.Instance(name='Part-B-1', part=p_b, dependent=ON)

print('==========================================================')
print('>>>>>>>>>>>>>> PBC for Interlocking Lattice <<<<<<<<<<<<<<')
print('==========================================================')
inst_name = asm.instances.keys()  # list of instance name
Num_inst = len(inst_name)  # number of instance
nde_set = []  # node set (asm)
Num_nde = []  # list of node number (inst)
# Mapping nodes for positive surface nodes [m_s_f, m_s_t, m_s_l]
MapNdeSet = [[], [], []]
# Surface node sets [[s_f, s_k], [s_t, s_b], [s_l, s_r]]
SrfNdeSet = [[[], []], [[], []], [[], []]]
# Edge node sets
EdgNde = []
nde_e = []
EdgSet = []
EdgSetName = []
# Tensile node sets
TensNde = []
# Fix node sets
FixNde = []
mt = np.empty(shape=(0, 3))  # empty array(0*3)
Coord = mt  # coordinates of all nodes (asm)

# Extract node coordinates from the assembly -------------------------------------------
for j in range(Num_inst):
    nde_set.append(asm.instances[inst_name[j]].nodes)
    Num_nde.append(len(nde_set[j]))
    for i in nde_set[j]:
        Coord = np.vstack([Coord, np.asarray(i.coordinates)])

Coord_lmt = np.vstack([np.amax(Coord, axis=0), np.amin(Coord, axis=0)])  # Coordinate upper and lower limits

size = Coord_lmt[0, :] - Coord_lmt[1, :]

asm.deleteSets(setNames=tuple(asm.allSets.keys()))  # Delete previous Sets

# Create Reference Points --------------------------------------------------------------
if 'RP-1' in asm.features.keys():
    del asm.features['RP-1']
if 'RP-nrm' in asm.features.keys():
    del asm.features['RP-nrm']
if 'RP-shr' in asm.features.keys():
    del asm.features['RP-shr']
if 'RP-pin' in asm.features.keys():
    del asm.features['RP-pin']

asm.ReferencePoint(point=tuple((Coord_lmt[0, :] + Coord_lmt[1, :]) / 2))  # RP for fixing rigid body translation
asm.features.changeKey(fromName='RP-1', toName='RP-pin')
asm.Set(referencePoints=(asm.referencePoints[int(asm.referencePoints.keys()[0])],), name='RP-PIN')

asm.ReferencePoint(point=tuple(Coord_lmt[1, :] - size * 0.025))  # RP for normal deformation
asm.features.changeKey(fromName='RP-1', toName='RP-nrm')
asm.Set(referencePoints=(asm.referencePoints[int(asm.referencePoints.keys()[0])],), name='RP-NRM')

asm.ReferencePoint(point=tuple(Coord_lmt[1, :] - size * 0.05))  # RP for shear deformation
asm.features.changeKey(fromName='RP-1', toName='RP-shr')
asm.Set(referencePoints=(asm.referencePoints[int(asm.referencePoints.keys()[0])],), name='RP-SHR')

Dim = 3

print('>>> Creating Node Sets...')
# Create Node Sets (3D) ---------------------------------------------------------------
k = 0
for j in range(Num_inst):
    for d_s in range(Dim):
        for s in [0, 1]:
            SrfNdeSet[d_s][s].append([])  # create a sublist for this instance
    n = asm.instances[inst_name[j]].nodes
    for i in range(int(Num_nde[j])):
        sn = i  # serial num
        lb = i + 1  # node label
        bcon = identifunc(Coord_lmt[:, :Dim], Coord[i+k, :Dim], T_s)
        if (bcon == [0, 0, 0]).all():
            continue  # Ignore all inner nodes
        else:  # boundary nodes
            if bcon[2] == -1 and bcon[0] != 1 and bcon[1] != 1:  # record (?,?,-1) as the fix nodes label
                FixNde.append(lb)
            elif bcon[2] == 1 and bcon[0] != 1 and bcon[1] != 1:  # record (?,?,1) as the tensile nodes label
                TensNde.append(lb)
            if abs(bcon[0]) == 1 or abs(bcon[1]) == 1:  # X & Y boundary nodes
                # create single node set
                nde = n.sequenceFromLabels(labels=(lb,), )
                asm.Set(nodes=nde, name='N' + str(j + 1) + '-' + str(lb))
                # classify the boundary nodes (serial num)
                for d in range(Dim):
                    if (bcon == [1, -1, 1]).all() and d == 0:  # Remove redundant constraints on vertex nodes
                        continue  # remove V_3 from front surface
                    elif (bcon == [1, 1, 1]).all() and d == 0:
                        continue  # remove V_0 from front surface
                    elif (bcon == [1, 1, -1]).all() and d == 0:
                        continue  # remove V_4 from front surface
                    elif (bcon == [1, 1, 0]).all() and d == 0:  # Remove redundant constraints on edge nodes
                        continue  # remove E_Z0 from front surface
                    elif bcon[d] != 0:
                        SrfNdeSet[d][int((1-bcon[d])/2)][j].append(sn)  # Put the node into corresponding surface set
    k += Num_nde[j]

# Create Fix node set and Tens node set
n_a, n_b = asm.instances['Part-A-1'].nodes, asm.instances['Part-B-1'].nodes
nde_t = n_b.sequenceFromLabels(labels=tuple(TensNde), )
nde_f = n_a.sequenceFromLabels(labels=tuple(FixNde), )
asm.Set(nodes=nde_t, name='TensNde')
asm.Set(nodes=nde_f, name='FixNde')

# Find Coupling Nodes and Mapping Relationships ----------------------------------------
print('>>> Finding Coupling Nodes and Mapping Relationships...')
k = 0
for j in range(Num_inst):
    for d_s in range(Dim-1):
        map_nde = nns(Coord[k:k + Num_nde[j], :], SrfNdeSet[d_s][0][j], SrfNdeSet[d_s][1][j], Dim, d_s)
        MapNdeSet[d_s].append(map_nde)
    k += Num_nde[j]

# Apply Equation Constraints ----------------------------------------------------------
print('>>> Applying Equation Constraints...')
mdl.constraints.delete(tuple(mdl.constraints.keys()))  # Delete previous constraints
Dir_s, Dir_c, SetRP = ['f', 't', 'l'], ['x', 'y', 'z'], ['RP-SHR', 'RP-NRM']
# Equation Constraints for PBC -------------------------------------------------------
for d_s in range(Dim - 1):  # direction of surface
    k = 0
    for j in range(Num_inst):  # serial num of instance
        for i in range(len(SrfNdeSet[d_s][0][j])):  # serial num of nodes
            idx_mst = SrfNdeSet[d_s][0][j][i]
            idx_slv = map(int, MapNdeSet[d_s][j][i, :])
            idx_row = np.insert(idx_slv, 0, idx_mst)
            SetMst = 'N' + str(j + 1) + '-' + str(idx_mst + 1)
            Mtx_coord = Coord[idx_row + k, :]
            Mtx_coord = np.delete(Mtx_coord, d_s, axis=1)
            W = mapping(Mtx_coord, T_v)
            for d_c in range(Dim):  # direction of constraint
                CstrName = 'Eqn-' + Dir_s[d_s] + str(j + 1) + '-' + str(idx_mst + 1) + Dir_c[d_c]
                SetSlv = 'N' + str(j + 1) + '-'
                SetSlv0 = SetSlv + str(idx_slv[0] + 1)
                SetSlv1 = SetSlv + str(idx_slv[1] + 1)
                # direction of constraint on RP
                if d_s == d_c:
                    Dir_c_RP = d_c + 1
                else:
                    Dir_c_RP = d_s + d_c
                # Constraint equation terms
                SetSlv2 = SetSlv + str(idx_slv[2] + 1)
                EqnTerms = ((1.0, SetMst, d_c + 1), (-1, SetRP[d_s == d_c], Dir_c_RP),
                            (-W[0], SetSlv0, d_c + 1), (-W[1], SetSlv1, d_c + 1), (-W[2], SetSlv2, d_c + 1))
                mdl.Equation(name=CstrName, terms=EqnTerms)
        k += Num_nde[j]

# Equation Constraints for tensile ----------------------------------------------------------------------------
mdl.Equation(name='Eqn-Tens', terms=((1.0, 'TensNde', 3), (-1.0, 'RP-NRM', 3)))

rgn_f = asm.sets['FixNde']
rgn_t = asm.sets['TensNde']
rgn_p = asm.sets['RP-PIN']
rgn_n = asm.sets['RP-NRM']

# STEP =================================================================================
print('>>> Creating Steps...')
# delete previous steps
for stp_name in mdl.steps.keys():
    if stp_name == 'Initial':
        continue
    else:
        del mdl.steps[stp_name]
# create static Step
mdl.StaticStep(name='Step-1', previous='Initial',
               stabilizationMagnitude=0.0002, stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
               continueDampingFactors=False, adaptiveDampingRatio=0.05,
               initialInc=0.001, maxInc=0.1, nlgeom=ON)

# Field output
mdl.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'RF',
                                                           'CSTRESS', 'CDISP', 'EVOL', 'STATUS'))

# History output
mdl.HistoryOutputRequest(name='H-Output-1', createStepName='Step-1', variables=('U1', 'U2', 'U3', 'RF3',),
                         region=rgn_n, sectionPoints=DEFAULT, rebar=EXCLUDE)

# Apply other BCs/loads/constraints ====================================================
print('>>> Applying Other BCs/Loads/constraints...')
# Contact
mdl.ContactProperty('IntProp-1')
mdl.interactionProperties['IntProp-1'].TangentialBehavior(formulation=FRICTIONLESS)
rgn_m = asm.instances['Part-A-1'].surfaces['Surf-Cont']
rgn_s = asm.instances['Part-B-1'].surfaces['Surf-Cont']

mdl.SurfaceToSurfaceContactStd(name='Int-Cont', createStepName='Initial', master=rgn_m, slave=rgn_s,
                               sliding=SMALL, thickness=ON, interactionProperty='IntProp-1',
                               surfaceSmoothing=AUTOMATIC, adjustMethod=OVERCLOSED,
                               initialClearance=OMIT, datumAxis=None, clearanceRegion=None)

# BC_Fix
mdl.ZsymmBC(name='BC-Fix', createStepName='Initial', region=rgn_f, localCsys=None)

# BC_Tens
mdl.TabularAmplitude(name='Amp-1', timeSpan=STEP, smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (StepTime, 1.0)))
mdl.DisplacementBC(name='BC-Tens', createStepName='Step-1', region=rgn_n,
                   u1=UNSET, u2=UNSET, u3=u_tens, ur1=0.0, ur2=0.0, ur3=0.0, amplitude='Amp-1', fixed=OFF,
                   distributionType=UNIFORM, fieldName='', localCsys=None)

# Pin
mdl.DisplacementBC(name='BC-Pin', createStepName='Initial', region=rgn_p,
                   u1=SET, u2=SET, u3=UNSET, ur1=SET, ur2=SET, ur3=SET, amplitude=UNSET,
                   distributionType=UNIFORM, fieldName='', localCsys=None)

# Couple RP-PIN with Cp nodes
mdl.Coupling(name='Cpl-Pin', controlPoint=rgn_p, surface=rgn_f, influenceRadius=WHOLE_SURFACE,
             couplingType=DISTRIBUTING, weightingMethod=UNIFORM, localCsys=None,
             u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

print('----------------------------------------------------------')
print('================ Pre-processing Complete =================')
print('----------------------------------------------------------')

mdb.Job(name=job_name, model=model_name, description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0,
        queue=None, memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF,
        historyPrint=OFF, userSubroutine='', scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT,
        numCpus=6, numDomains=6, numGPUs=0)

mdb.jobs[job_name].submit(consistencyChecking=OFF)
print('>>> Solving the job...')

# ====================================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Solving & monitoring <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ====================================================================================
def get_line_intersection(p1, p2, p3, p4):
    """
    Computes the intersection of two line segments (p1, p2) and (p3, p4).
    Returns the intersection point (x, y), or None if there is no intersection or the line segments do not intersect.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None  # Parallel or colinear

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)

    return None


def find_intersections(A, B):
    """
    Find all intersections of polyline data A and B.
    A and B are lists of points [(x1, y1), (x2, y2), ...]
    Returns a list of intersection points [(x, y), ...]
    """
    intersections = []

    for i in range(len(A) - 1, 0, -1):
        p1, p2 = A[i - 1], A[i]
        p3, p4 = B[i - 1], B[i]

        intersection = get_line_intersection(p1, p2, p3, p4)
        if intersection:
            intersections.append(intersection)

    return intersections


postprc = int(sys.argv[-1])

if postprc == 1:
    inc_thresh = 10
    # do-while loop 1
    while True:
        # do-while loop 2
        while True:
            # do-while loop 3
            while True:
                valid_increment_updated = False
                # Checking valid increment
                if sta_file in os.listdir(work_dir):  # STA file Detected
                    with open(sta_file, 'r') as file:
                        last_line = file.readlines()[-1].strip().split()  # Read the last line of sta file
                        if last_line[2].isdigit():  # Valid increment updated
                            valid_increment_updated = True
                # Break the do-while loop 3
                if valid_increment_updated:
                    break
                time.sleep(5)
            # Break the do-while loop 2
            if int(last_line[1]) >= inc_thresh:
                break

        # Remove the lck file if existed
        if os.path.exists(lck_filepath):
            os.remove(lck_filepath)
        # Open & Read the odb file
        odb = odbAccess.openOdb(r'C:\\temp\\' + job_name + '.odb')  # ,readOnly=
        ho = odb.steps['Step-1'].historyRegions['Node ASSEMBLY.2'].historyOutputs
        Force3 = [row[1] for row in ho['RF3'].data]
        Disp1 = [row[1] for row in ho['U1'].data]
        Disp2 = [row[1] for row in ho['U2'].data]
        Disp3 = [row[1] for row in ho['U3'].data]
        # Calculation for Curve 1 & 2
        Stress_eq = [f3 / ((ucs + u1) * (ucs + u2)) for f3, u1, u2 in zip(Force3, Disp1, Disp2)]  # Equivalent Stress
        Strain_eq = [u3 / thk_l for u3 in Disp3]  # Equivalent Strain
        E_eq = Stress_eq[1] / Strain_eq[1]  # Equivalent Young's Modulus
        Stress_ofst = [E_eq * (strain - 2e-3) for strain in Strain_eq]  # Offsetted Stress
        Curve1, Curve2 = list(zip(Strain_eq, Stress_eq)), list(zip(Strain_eq, Stress_ofst))  # Curve 1 & 2
        # Find intersection
        intersections = find_intersections(Curve1, Curve2)

        if len(intersections) == 0:
            pass
            # print(">>> No intersection found...")
        else:
            os.system('abaqus terminate job=' + job_name)
            print(">>> Proof stress point reached, Job terminated...")
            strain_proof, stress_proof = intersections[0]
            print(">>> Equivalent Proof Strain = %.3f %%..." % (strain_proof*100))
            print(">>> Equivalent Proof Stress = %.3f MPa..." % stress_proof)
            print(">>>>>>>>>>>> %.3f, %.3f" % (stress_proof, strain_proof * 100))
            with open('result.txt', 'w') as result_file:
                result_file.write("{:.3f}\n".format(stress_proof))
            print("==========================================================")
            file_path = 'C:/temp/TPMS11/initial_guesses.csv'
            with open(file_path, 'r') as file:
                rows = list(csv.reader(file))
                rows[-1][4] = '{:.3f}'.format(-stress_proof)
                rows[-1][5] = '{:.3f}'.format(strain_proof * 100)
            with open(file_path, 'wb') as file:
                csv.writer(file).writerows(rows)
            break
