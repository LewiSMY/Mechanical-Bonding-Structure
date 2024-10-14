import pyvista as pv
import numpy as np
import pymeshlab as pml
import trimesh as tm
import pycork
import tetgen


def topo_check(self):
    topo_rst = self.get_topological_measures()
    # print(topo_rst)
    noe_bound = topo_rst['boundary_edges']
    nocc = topo_rst['connected_components_number']
    nof_n2mnfd_e = topo_rst['incident_faces_on_non_two_manifold_edges']
    nof_n2mnfd_v = topo_rst['incident_faces_on_non_two_manifold_vertices']
    is_2mnfd = topo_rst['is_mesh_two_manifold']
    noe_n2mnfd = topo_rst['non_two_manifold_edges']
    nov_n2mnfd = topo_rst['non_two_manifold_vertices']
    noh = topo_rst['number_holes']
    nov_unref = topo_rst['unreferenced_vertices']
    rst_list = [noe_bound, nocc, nof_n2mnfd_e, nof_n2mnfd_v, is_2mnfd, noe_n2mnfd, nov_n2mnfd, noh, nov_unref]
    return rst_list


def polydata2trimesh(pdata):
    faces_as_array = pdata.faces.reshape((pdata.n_faces_strict, 4))[:, 1:]
    return tm.Trimesh(pdata.points, faces_as_array)


def trimesh_boolean(polydata1, polydata2, type):  # pv
    trimesh1, trimesh2 = polydata2trimesh(polydata1), polydata2trimesh(polydata2)
    if type == 'd':
        trimesh = tm.boolean.difference([trimesh1, trimesh2], engine='blender')
    elif type == 'u':
        trimesh = tm.boolean.union([trimesh1, trimesh2], engine='blender')
    elif type == 'i':
        trimesh = tm.boolean.intersection([trimesh1, trimesh2], engine='blender')
    polydata = pv.wrap(trimesh)
    return polydata


def trimesh_invert(polydata):  # pv
    trimesh = polydata2trimesh(polydata)
    trimesh.invert()
    polydata_inv = pv.wrap(trimesh)
    return polydata_inv


def pycork_boolean(polydata1, polydata2, type):  # pv
    trimesh1, trimesh2 = polydata2trimesh(polydata1), polydata2trimesh(polydata2)
    verts1, tris1 = trimesh1.vertices, trimesh1.faces
    verts2, tris2 = trimesh2.vertices, trimesh2.faces

    pycork.isSolid(verts1, tris1)
    pycork.isSolid(verts2, tris2)

    if type == 'd':
        verts, tris = pycork.difference(verts1, tris1, verts2, tris2)
    elif type == 'u':
        verts, tris = pycork.union(verts1, tris1, verts2, tris2)
    elif type == 'i':
        verts, tris = pycork.intersection(verts1, tris1, verts2, tris2)

    trimesh = tm.Trimesh(vertices=verts, faces=tris, process=True)
    polydata = pv.wrap(trimesh)
    return polydata


def pyvista_boolean(polydata1, polydata2, type):  # pv
    if type == 'd':
        polydata = polydata1.boolean_difference(polydata2)
    elif type == 'u':
        polydata = polydata1.boolean_union(polydata2)
    elif type == 'i':
        polydata = polydata1.boolean_intersection(polydata2)
    return polydata


def meshlab_boolean(polydata1, polydata2, type):  # pv
    polydata1.save('temp1.stl')
    polydata2.save('temp2.stl')

    ms = pml.MeshSet()
    ms.load_new_mesh('temp1.stl')
    ms.load_new_mesh('temp2.stl')
    ms.generate_boolean_intersection()
    ms.save_current_mesh('temp.stl')

    if type == 'd':
        ms.generate_boolean_difference()
    elif type == 'u':
        ms.generate_boolean_union()
    elif type == 'i':
        ms.generate_boolean_intersection()
    ms.save_current_mesh('temp.stl')
    polydata = pv.read('temp.stl')

    return polydata


def boolean(polydata1, polydata2, type, method):  # pv
    if method == 'tm':
        polydata = trimesh_boolean(polydata1, polydata2, type)
    elif method == 'pc':
        polydata = pycork_boolean(polydata1, polydata2, type)
    elif method == 'pv':
        polydata = pyvista_boolean(polydata1, polydata2, type)
    elif method == 'ml':
        polydata = meshlab_boolean(polydata1, polydata2, type)
    return polydata


def meshlab_remesh(polydata, len_pct):  # pv
    polydata.save('temp.stl')
    ms = pml.MeshSet()
    ms.load_new_mesh('temp.stl')
    ms.meshing_isotropic_explicit_remeshing(iterations=10, targetlen=pml.PercentageValue(len_pct), featuredeg=30)
    ms.save_current_mesh('temp.stl')
    polydata = pv.read('temp.stl')
    return polydata


def meshlab_process(mesh_file, len_pct):  # str
    ms = pml.MeshSet()
    ms.load_new_mesh(mesh_file)
    print(">>> Mesh processing using PyMeshLab...")
    print(">>> New surface mesh " + mesh_file + " loaded...")
    ms.meshing_remove_duplicate_faces()
    print("Duplicate faces removed...")
    noi = 3  # number of iterations

    while noi >= 1:
        print(">>> Iteration " + str(noi) + "...")
        noi -= 1
        ms.meshing_merge_close_vertices(threshold=pml.PercentageValue(3))
        print(">>> Close vertices merged...")
        ms.meshing_isotropic_explicit_remeshing(iterations=3*(5-noi), adaptive=True,
                                                targetlen=pml.PercentageValue(len_pct * 0.6**noi))

        print(">>> Surface mesh Remeshed...")
    ms.meshing_merge_close_vertices(threshold=pml.PercentageValue(5))
    rst_list = topo_check(ms)
    # Check & Repair for Two-Manifold
    while not rst_list[4]:
        if rst_list[2] != 0:
            ms.meshing_repair_non_manifold_edges(method=0)
        if rst_list[3] != 0:
            ms.meshing_repair_non_manifold_vertices()
        rst_list = topo_check(ms)
    rst_list = topo_check(ms)
    print(rst_list)
    print(">>> Surface mesh " + mesh_file + " is two-manifold...")
    # Check & Repair for Holes
    NoH_count = 0
    while rst_list[7] != 0:
        NoH_last = rst_list[8]
        ms.meshing_close_holes(maxholesize=30)
        rst_list = topo_check(ms)
        NoH_now = rst_list[8]
        NoH_diff = NoH_last - NoH_now
        if NoH_diff == 0:
            print('NoH no longer reduces')
            NoH_count += 1
            if NoH_count >= 2:
                print('>>> Mesh quality not acceptable, regenerate the mesh...')
                return None
    print(">>> Surface mesh " + mesh_file + " has 0 holes...")
    # Check & Repair for Unreferenced Vertices
    while rst_list[8] != 0:
        ms.meshing_remove_unreferenced_vertices()
        rst_list = topo_check(ms)
    print(">>> Surface mesh " + mesh_file + " has 0 unreferenced vertices...")
    print(">>> Surface Mesh " + mesh_file + " repaired...")
    # print(ms.get_topological_measures())
    ms.save_current_mesh(mesh_file)

    return pv.read(mesh_file)
