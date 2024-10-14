import pyvista as pv
import numpy as np
from numpy import pi, cos, sin
import random
import Implicit2TriMesh as i2tm
import tetgen
import pandas as pd

pv.set_plot_theme('document')


def fun_tpms(x, y, z, lmd, mu, kpa, bta, scl=1.):
    rnd_x, rnd_y = random.random(), random.random()
    print('[rnd_x, rnd_y] = [', f"{rnd_x:.4f}", f"{rnd_y:.4f}", ']')
    x += pi * 8 * rnd_x
    y += pi * 8 * rnd_y

    x, y, z = x / scl, y / scl, z / scl

    sx, sy, sz, s2x, s2y, s2z = sin(x), sin(y), sin(z), sin(2 * x), sin(2 * y), sin(2 * z)
    cx, cy, cz, c2x, c2y, c2z = cos(x), cos(y), cos(z), cos(2 * x), cos(2 * y), cos(2 * z)
    c1_5z = cos(1.5 * (z + pi))
    '''p = cx + cy + cz - (1/(3*pi)) * z  # Primitive
    g = sx * cy + sy * cz + sz * cx - (1.4/(3*pi)) * z  # Gyroid - (1.4/(3*pi)) * z
    d = cx * cy * cz - sx * sy * sz - (.707/(3*pi)) * z  # Dimond
    fks = c2x * sy * cz + cx * c2y * sz + sx * cy * c2z - (.8/(3*pi)) * z  # FKS'''

    '''t_max = [1, 1.414, 0.707, 0.75]
    t = 2 * t_max[c] * lv - t_max[c]'''

    # t = 0.5 - lv
    t = 0
    p = cx + cy + c1_5z - 3.495 * t  # Primitive
    # 1
    g = sx * cy + sy * cz + sz * cx - 3.014 * t  # Gyroid - (1.4/(3*pi)) * z
    # 1.414 (0.25-0.75)
    d = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz - 2.414 * t  # Diamond cx * cy * cz - sx * sy * sz
    # 0.707
    fks = c2x * sy * cz + cx * c2y * sz + sx * cy * c2z - 1.954 * t  # FKS
    # 0.8
    f = 0 * p + lmd * g + mu * d + (1 - lmd - mu) * fks + (kpa * z + bta)
    return f


def mesh_generation(lmd, mu, kpa, bta, plot=None, len_pct=1.5):
    print('[lmd, mu, kpa, bta] = [',
          f"{lmd:.4f}", f"{mu:.4f}", f"{kpa:.4f}", f"{bta:.4f}", ']')
    wth = 10  # width of the RVE
    nol_ltc = int(3)  # Num of lattice layers
    thk_l = wth * nol_ltc  # lattice thickness
    thk_p = wth / 10  # plate thickness
    thk_c = 1.5  # corp rate
    len_pct_fine = len_pct*0.5

    dx0, dy0, dz0 = wth / 2, wth / 2, thk_l / 2
    dx1, dy1, dz1 = dx0 * thk_c, dy0 * thk_c, dz0 + thk_p
    dx2, dy2, dz2 = dx1 * thk_c, dy1 * thk_c, dz1 + thk_p

    m = 60  # Sampling density (m equal segments)
    spc = wth / m
    dims_tpms = np.array([2 * dx2, 2 * dy2, 2 * dz2])
    dims = tuple(int(i + 1) for i in dims_tpms / spc)

    scl_tpms = (pi / 2 * 1.01325)  # Scale factor 1.01325 !!!!!!

    # MOD ------------------------------------------------------------------------
    # TPMS surfaces ====================================================================================================
    grid = pv.ImageData(dimensions=dims, spacing=(spc, spc, spc), origin=(-dx2, -dy2, -dz2))
    X, Y, Z = grid.points.T
    values = fun_tpms(X, Y, Z, lmd, mu, kpa, bta, scl_tpms)
    tpms_mc = grid.contour(1, values, method='marching_cubes', rng=[-0., 0.])
    tpms_sole = tpms_mc.connectivity('largest')
    tpms = i2tm.meshlab_remesh(tpms_sole, len_pct=len_pct_fine)
    tpms = i2tm.trimesh_invert(tpms)

    print(">>> TPMS Generated...")

    # Tool_A ===========================================================================================================
    tl_a = pv.Box(bounds=(-dx0, dx0, -dy0, dy0, -dz1, dz0), quads=False)
    tl_a = i2tm.meshlab_remesh(tl_a, len_pct=len_pct_fine)
    # Tool_B -----------------------------------------------------------------------------------------------------------
    tl_b = pv.Box(bounds=(-dx0, dx0, -dy0, dy0, -dz0, dz1), quads=False)
    tl_b = i2tm.meshlab_remesh(tl_b, len_pct=len_pct_fine)
    # Tool_C -----------------------------------------------------------------------------------------------------------
    tl_c = pv.Box(bounds=(-dx1, dx1, -dy1, dy1, -dz1, dz1), quads=False)
    tl_c = i2tm.meshlab_remesh(tl_c, len_pct=len_pct_fine)
    # Plate_A ----------------------------------------------------------------------------------------------------------
    plt_a = pv.Box(bounds=(-dx2, dx2, -dy2, dy2, -dz2, -dz0), quads=False)
    plt_a = i2tm.meshlab_remesh(plt_a, len_pct=len_pct_fine)
    # Plate_B ----------------------------------------------------------------------------------------------------------
    plt_b = pv.Box(bounds=(-dx2, dx2, -dy2, dy2, dz0, dz2), quads=False)
    plt_b = i2tm.meshlab_remesh(plt_b, len_pct=len_pct_fine)
    print(">>> Tools & Plates Generated...")

    # Part_A ===========================================================================================================

    prt_a0 = i2tm.boolean(tl_c, tpms, type='i', method='pv')  # prt_a0 = tl_c * tpms
    prt_a1 = i2tm.boolean(prt_a0, plt_a, type='u', method='pc')  # + plt_a
    prt_a = i2tm.boolean(prt_a1, tl_a, type='i', method='pc')  # * tl_A
    print(">>> Part A Generated...")

    # Part_B ===========================================================================================================
    prt_b0 = i2tm.boolean(tl_c, tpms, type='d', method='pv')  # prt_b0 = tl_c - tpms
    prt_b1 = i2tm.boolean(prt_b0, plt_b, type='u', method='pc')  # + plt_b
    prt_b = i2tm.boolean(prt_b1, tl_b, type='i', method='pc')  # * tl_b
    print(">>> Part B Generated...")

    # =============================================================================
    mesh_name = 'tpms'
    mesh_path = 'C:/temp/TPMS11/'
    trimesh_name = mesh_path + mesh_name + '-tri-'
    tetmesh_name = mesh_path + mesh_name + '-tet-'

    pv.save_meshio(trimesh_name + 'A.obj', prt_a)
    pv.save_meshio(trimesh_name + 'B.obj', prt_b)

    print('>>> Tri-mesh Generated...')
    print("--------------------------------------------------------------------")

    # Remesh and repair =========================================================
    prt_a_rms = i2tm.meshlab_process(trimesh_name + 'A.obj', len_pct=len_pct)
    prt_b_rms = i2tm.meshlab_process(trimesh_name + 'B.obj', len_pct=len_pct)

    # Save the Tri-mesh ========================================================
    pv.save_meshio(trimesh_name + 'A.inp', prt_a_rms)
    pv.save_meshio(trimesh_name + 'B.inp', prt_b_rms)
    print("--------------------------------------------------------------------")

    tet = False
    if tet:
        # Tetrahedralization ========================================================
        print(">>> Tetrahedralization using TetGen...")

        prt_a_tet = tetgen.TetGen(prt_a_rms)
        prt_a_tet.tetrahedralize(order=1, mindihedral=20., minratio=1.1,
                                 nobisect=True, verbose=1)
        prt_a_grid = prt_a_tet.grid

        prt_b_tet = tetgen.TetGen(prt_b_rms)
        prt_b_tet.tetrahedralize(order=1, mindihedral=20., minratio=1.1,
                                 nobisect=True, verbose=1)
        prt_b_grid = prt_b_tet.grid

        # Save the Tet-mesh ========================================================
        pv.save_meshio(tetmesh_name + 'A.inp', prt_a_grid)
        pv.save_meshio(tetmesh_name + 'B.inp', prt_b_grid)

        # get cell centroids
        prt_a_cells = prt_a_grid.cells.reshape(-1, 5)[:, 1:]
        prt_a_cell_center = prt_a_grid.points[prt_a_cells].mean(1)
        prt_b_cells = prt_b_grid.cells.reshape(-1, 5)[:, 1:]
        prt_b_cell_center = prt_b_grid.points[prt_b_cells].mean(1)
        # extract cells below the 0 xy plane
        prt_a_mask = prt_a_cell_center[:, 1] < 0
        prt_a_cell_ind = prt_a_mask.nonzero()[0]
        prt_a_subgrid = prt_a_grid.extract_cells(prt_a_cell_ind)
        prt_b_mask = prt_b_cell_center[:, 1] < 0
        prt_b_cell_ind = prt_b_mask.nonzero()[0]
        prt_b_subgrid = prt_b_grid.extract_cells(prt_b_cell_ind)

    # Plot ===================================================================================================
    if plot == 'B':  # Boolean operation -----------------------------------------------------------------------------
        plt = pv.Plotter(shape=(2, 6))
        clr = 'purple'
        plt.subplot(0, 0)  # tpms via MC ------------------------------------------------
        plt.add_mesh(tpms_sole, color='#BED8BC', show_edges=True)
        plt.subplot(1, 0)  # remeshed tpms
        plt.add_mesh(tpms, opacity=0.75, color='#BED8BC', show_edges=False)
        plt.add_mesh(tl_c, opacity=0.5, color=clr, show_edges=False)
        plt.subplot(0, 1)  # ------------------------------------------------------------
        plt.add_mesh(prt_a0, color='#9DC3E7', show_edges=False)
        plt.subplot(1, 1)
        plt.add_mesh(prt_b0, color='#F0988C', show_edges=False)
        plt.subplot(0, 2)  # ------------------------------------------------------------
        plt.add_mesh(prt_a0, opacity=0.75, color='#9DC3E7', show_edges=False)
        plt.add_mesh(plt_a, opacity=0.5, color=clr, show_edges=False)
        plt.subplot(1, 2)
        plt.add_mesh(prt_b0, opacity=0.75, color='#F0988C', show_edges=False)
        plt.add_mesh(plt_b, opacity=0.5, color=clr, show_edges=False)
        plt.subplot(0, 3)  # ------------------------------------------------------------
        plt.add_mesh(prt_a1, color='#9DC3E7', show_edges=False)
        plt.subplot(1, 3)
        plt.add_mesh(prt_b1, color='#F0988C', show_edges=False)
        plt.subplot(0, 4)  # ------------------------------------------------------------
        plt.add_mesh(prt_a1, opacity=0.5, color='#9DC3E7', show_edges=False)
        plt.add_mesh(tl_a, opacity=0.5, color=clr, show_edges=False)
        plt.subplot(1, 4)
        plt.add_mesh(prt_b1, opacity=0.5, color='#F0988C', show_edges=False)
        plt.add_mesh(tl_b, opacity=0.5, color=clr, show_edges=False)
        plt.subplot(0, 5)  # ------------------------------------------------------------
        plt.add_mesh(prt_a, color='#999999', show_edges=False)
        plt.subplot(1, 5)
        plt.add_mesh(prt_b, color='#E7DAD2', show_edges=False)
    elif plot == 'R':  # Remeshing -----------------------------------------------------------------------------
        plt = pv.Plotter(shape=(2, 3))
        plt.subplot(0, 0)
        plt.add_mesh(tpms_mc, color='#BED8BC', show_edges=True)
        plt.subplot(1, 0)
        plt.add_mesh(prt_a_rms, color='#999999', show_edges=True)
        plt.add_mesh(prt_b_rms, color='#E7DAD2', show_edges=True)
        plt.subplot(0, 1)
        plt.add_mesh(prt_a, color='#9DC3E7', show_edges=True)
        plt.subplot(1, 1)
        plt.add_mesh(prt_b, color='#F0988C', show_edges=True)
        plt.subplot(0, 2)
        plt.add_mesh(prt_a_rms, color='#999999', show_edges=False)
        #plt.add_mesh(sharp_edges_a, color="red", line_width=5)
        plt.subplot(1, 2)
        plt.add_mesh(prt_b_rms, color='#E7DAD2', show_edges=False)
        #plt.add_mesh(sharp_edges_b, color="red", line_width=5)
        plt.link_views()
        plt.enable_parallel_projection()
        plt.show()
    elif plot == 'T':  # Tetrahedralization ----------------------------------------------------------------------------
        plt = pv.Plotter(shape=(2, 3))
        plt.subplot(0, 0)
        plt.add_mesh(prt_a_rms, color='#999999', show_edges=True)
        plt.subplot(0, 1)
        plt.add_mesh(prt_a_grid, color='#999999', show_edges=True)
        plt.subplot(0, 2)
        plt.add_mesh(prt_a_subgrid, color='#999999', lighting=True, show_edges=True)
        # plt.add_mesh(prt_a_rms, color='r', style='wireframe')
        plt.subplot(1, 0)
        plt.add_mesh(prt_b_rms, color='#E7DAD2', show_edges=True)
        plt.subplot(1, 1)
        plt.add_mesh(prt_b_grid, color='#E7DAD2', show_edges=True)
        plt.subplot(1, 2)
        plt.add_mesh(prt_b_subgrid, color='#E7DAD2', lighting=True, show_edges=True)
        # plt.add_mesh(prt_b_rms, color='r', style='wireframe')
        plt.link_views()
        plt.enable_parallel_projection()
        plt.show()
    elif plot == 'N':
        pass

    return [prt_a_rms, prt_b_rms]


def sharp_edge_num(mesh_list, thr_angle):
    sharp_edges_num_list = []
    for mesh in mesh_list:
        sharp_edges = mesh.extract_feature_edges(boundary_edges=False, feature_angle=180-thr_angle)
        sharp_edges_num_list.append(sharp_edges.n_cells)
    edge_num = max(sharp_edges_num_list)
    print('edge_num =', edge_num)
    return edge_num


def comp_vol_min(mesh_list):
    vol_list = []
    for mesh in mesh_list:
        conn_comps = mesh.connectivity()
        comp_vol = []
        for label in np.unique(conn_comps.point_data['RegionId']):
            comp = conn_comps.threshold(label, scalars='RegionId')
            vol = comp.extract_surface().volume
            comp_vol.append(vol)
        if len(comp_vol) < 2:
            comp_vol_indiv = comp_vol
        else:
            comp_vol_indiv = \
                [comp_vol[i] - comp_vol[i + 1] for i in range(len(comp_vol) - 1)]
            comp_vol_indiv.append(comp_vol[-1])
        vol_list.append(comp_vol_indiv)
    # print(vol_list)
    vol_min = min(min(inner_list) for inner_list in vol_list)
    print('vol_min =', vol_min)
    return vol_min


def good_mesh_generation(lmd, mu, kpa, bta, plot=None, len_pct=1.5):
    good_mesh = False
    while not good_mesh:
        mesh_list = mesh_generation(lmd, mu, kpa, bta, plot=plot, len_pct=len_pct)

        if mesh_list is None or sharp_edge_num(mesh_list, 5) != 0 or comp_vol_min(mesh_list) < 1:
            print('>>> Mesh quality not acceptable, regenerate the mesh...')
            print("====================================================================")
            continue
        else:
            good_mesh = True
            print('>>> Good mesh generated!!!')
            print("====================================================================")

            '''tet = False
            if tet:
                # Tetrahedralization ========================================================
                print(">>> Tetrahedralization using TetGen...")
        
                prt_a_tet = tetgen.TetGen(prt_a_rms)
                prt_a_tet.tetrahedralize(order=1, mindihedral=20., minratio=1.1,
                                         nobisect=True, verbose=1)
                prt_a_grid = prt_a_tet.grid
        
                prt_b_tet = tetgen.TetGen(prt_b_rms)
                prt_b_tet.tetrahedralize(order=1, mindihedral=20., minratio=1.1,
                                         nobisect=True, verbose=1)
                prt_b_grid = prt_b_tet.grid
        
                # Save the Tet-mesh ========================================================
                pv.save_meshio(tetmesh_name + 'A.inp', prt_a_grid)
                pv.save_meshio(tetmesh_name + 'B.inp', prt_b_grid)
        
                # get cell centroids
                prt_a_cells = prt_a_grid.cells.reshape(-1, 5)[:, 1:]
                prt_a_cell_center = prt_a_grid.points[prt_a_cells].mean(1)
                prt_b_cells = prt_b_grid.cells.reshape(-1, 5)[:, 1:]
                prt_b_cell_center = prt_b_grid.points[prt_b_cells].mean(1)
                # extract cells below the 0 xy plane
                prt_a_mask = prt_a_cell_center[:, 1] < 0
                prt_a_cell_ind = prt_a_mask.nonzero()[0]
                prt_a_subgrid = prt_a_grid.extract_cells(prt_a_cell_ind)
                prt_b_mask = prt_b_cell_center[:, 1] < 0
                prt_b_cell_ind = prt_b_mask.nonzero()[0]
                prt_b_subgrid = prt_b_grid.extract_cells(prt_b_cell_ind)'''

            '''# Plot ===================================================================================================
            if plot == 'B':  # Boolean operation -----------------------------------------------------------------------------
                plt = pv.Plotter(shape=(2, 6))
                clr = 'purple'
                plt.subplot(0, 0)  # tpms via MC ------------------------------------------------
                plt.add_mesh(tpms_sole, color='#BED8BC', show_edges=True)
                plt.subplot(1, 0)  # remeshed tpms
                plt.add_mesh(tpms, opacity=0.75, color='#BED8BC', show_edges=False)
                plt.add_mesh(tl_c, opacity=0.5, color=clr, show_edges=False)
                plt.subplot(0, 1)  # ------------------------------------------------------------
                plt.add_mesh(prt_a0, color='#9DC3E7', show_edges=False)
                plt.subplot(1, 1)
                plt.add_mesh(prt_b0, color='#F0988C', show_edges=False)
                plt.subplot(0, 2)  # ------------------------------------------------------------
                plt.add_mesh(prt_a0, opacity=0.75, color='#9DC3E7', show_edges=False)
                plt.add_mesh(plt_a, opacity=0.5, color=clr, show_edges=False)
                plt.subplot(1, 2)
                plt.add_mesh(prt_b0, opacity=0.75, color='#F0988C', show_edges=False)
                plt.add_mesh(plt_b, opacity=0.5, color=clr, show_edges=False)
                plt.subplot(0, 3)  # ------------------------------------------------------------
                plt.add_mesh(prt_a1, color='#9DC3E7', show_edges=False)
                plt.subplot(1, 3)
                plt.add_mesh(prt_b1, color='#F0988C', show_edges=False)
                plt.subplot(0, 4)  # ------------------------------------------------------------
                plt.add_mesh(prt_a1, opacity=0.5, color='#9DC3E7', show_edges=False)
                plt.add_mesh(tl_a, opacity=0.5, color=clr, show_edges=False)
                plt.subplot(1, 4)
                plt.add_mesh(prt_b1, opacity=0.5, color='#F0988C', show_edges=False)
                plt.add_mesh(tl_b, opacity=0.5, color=clr, show_edges=False)
                plt.subplot(0, 5)  # ------------------------------------------------------------
                plt.add_mesh(prt_a, color='#999999', show_edges=False)
                plt.subplot(1, 5)
                plt.add_mesh(prt_b, color='#E7DAD2', show_edges=False)
            elif plot == 'R':  # Remeshing -----------------------------------------------------------------------------
                plt = pv.Plotter(shape=(2, 3))
                plt.subplot(0, 0)
                plt.add_mesh(tpms_mc, color='#BED8BC', show_edges=True)
                plt.subplot(1, 0)
                plt.add_mesh(prt_a_rms, color='#999999', show_edges=True)
                plt.add_mesh(prt_b_rms, color='#E7DAD2', show_edges=True)
                plt.subplot(0, 1)
                plt.add_mesh(prt_a, color='#9DC3E7', show_edges=True)
                plt.subplot(1, 1)
                plt.add_mesh(prt_b, color='#F0988C', show_edges=True)
                plt.subplot(0, 2)
                plt.add_mesh(prt_a_rms, color='#999999', show_edges=False)
                #plt.add_mesh(sharp_edges_a, color="red", line_width=5)
                plt.subplot(1, 2)
                plt.add_mesh(prt_b_rms, color='#E7DAD2', show_edges=False)
                #plt.add_mesh(sharp_edges_b, color="red", line_width=5)
                plt.link_views()
                plt.enable_parallel_projection()
                plt.show()
            elif plot == 'T':  # Tetrahedralization ----------------------------------------------------------------------------
                plt = pv.Plotter(shape=(2, 3))
                plt.subplot(0, 0)
                plt.add_mesh(prt_a_rms, color='#999999', show_edges=True)
                plt.subplot(0, 1)
                plt.add_mesh(prt_a_grid, color='#999999', show_edges=True)
                plt.subplot(0, 2)
                plt.add_mesh(prt_a_subgrid, color='#999999', lighting=True, show_edges=True)
                # plt.add_mesh(prt_a_rms, color='r', style='wireframe')
                plt.subplot(1, 0)
                plt.add_mesh(prt_b_rms, color='#E7DAD2', show_edges=True)
                plt.subplot(1, 1)
                plt.add_mesh(prt_b_grid, color='#E7DAD2', show_edges=True)
                plt.subplot(1, 2)
                plt.add_mesh(prt_b_subgrid, color='#E7DAD2', lighting=True, show_edges=True)
                # plt.add_mesh(prt_b_rms, color='r', style='wireframe')
                plt.link_views()
                plt.enable_parallel_projection()
                plt.show()'''

if __name__ == '__main__':
    file_path = 'initial_guesses.csv'
    new_sample = pd.read_csv(file_path).iloc[-1].tolist()

    good_mesh_generation(new_sample[0], new_sample[1], new_sample[2], new_sample[3],
                         plot='N', len_pct=1.4)
