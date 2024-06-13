import os
import numpy as np
import orix
from orix.quaternion import Orientation, Rotation
from orix.vector import Vector3d
from orix.crystal_map import CrystalMap
from orix.io.plugins import ang

import matplotlib.pyplot as plt

def burgers_orientation(alpha_ori):
    # Burgers orientation relationship matrix
    burgers_matrix = np.array([
        [0.7071, -0.7071, 0],
        [0.7071, 0.7071, 0],
        [0, 0, 1]
    ])
    # Apply the Burgers matrix to the orientations
    burgers_R = Rotation.from_matrix(burgers_matrix)
    transformed_orientations = burgers_R * alpha_ori 
    return transformed_orientations

def step02(alpha_ebsd_folder, beta_ebsd_folder):
    print(f"\n\nAccessing folder: {alpha_ebsd_folder}\n")

    ipf_direction = Vector3d.y
    beta_threshold = 2.5
    min_beta_size_pixels = 15

    file_list = [f for f in os.listdir(alpha_ebsd_folder) if f.endswith('.ang')]
    n_files = len(file_list)

    reconstruction_plot_folder = os.path.join(beta_ebsd_folder, 'BetaGrainImages')
    os.system(f"mkdir {reconstruction_plot_folder}")

    for i_file in range(n_files):
        alpha_ebsd_file = os.path.join(alpha_ebsd_folder, file_list[i_file])
        local_ebsd_base_name = os.path.splitext(file_list[i_file])[0]
        
        ebsd = ang.file_reader(alpha_ebsd_file)
        #alpha_ebsd = CrystalMap.read_ang(alpha_ebsd_file, phase_name='Titanium (Alpha)')
        
        alpha_ebsd = ebsd["Titanium (Beta)"]
        # Assuming you have a method to get the spacing from alpha_ebsd
        spacing = alpha_ebsd.dx * 2
        # Plotting and saving figures can be implemented as needed
        #plot_save(alpha_ebsd, os.path.join(alpha_ebsd_folder, f"{local_ebsd_base_name}.png"))
        fig = alpha_ebsd.plot()
        plt.savefig(f"{local_ebsd_base_name}.png")
        #beta2alpha = Orientation.Burgers(alpha_ebsd.beta.cs, alpha_ebsd.alpha.cs)
        import pdb;pdb.set_trace()
        beta2alpha = burgers_orientation(alpha_ebsd.orientations)

        # Reconstruct grains
        grains = alpha_ebsd.grains(threshold=1.5 * np.pi / 180, 
                                    remove_quadruple_points=True)
        grains = grains.smooth(iterations=1, move_triple_points=True)

        plot_save(grains, os.path.join(alpha_ebsd_folder, f"{local_ebsd_base_name}_Marked.png"))

        # Extract alpha-alpha-alpha triple points and find the best fitting parent orientations
        triple_points = grains.triple_points(alpha='Titanium (Alpha)', beta='Titanium (Beta)')
        triple_point_orientations = grains[triple_points.grain_id].mean_orientation

        parent_id, fit = triple_points.calc_parent(triple_point_orientations, beta2alpha, num_fit=2, threshold=5 * np.pi / 180)
        consistent_tp = fit[:, 0] < 2.5 * np.pi / 180 and fit[:, 1] > 2.5 * np.pi / 180

        parent_id, num_votes = grains.majority_vote(triple_points[consistent_tp].grain_id, parent_id[consistent_tp, :, 0], max(grains.id), strict=True)

        parent_grains = grains.copy()
        parent_grains[num_votes > 2].mean_orientation = grains.variants(beta2alpha, grains[num_votes > 2].mean_orientation, parent_id[num_votes > 2])
        parent_grains.update()

        plot_save(parent_grains, os.path.join(reconstruction_plot_folder, f"{local_ebsd_base_name}_Step1.png"))

        parent_grains, parent_id = parent_grains.merge(threshold=beta_threshold * np.pi / 180, test_run=True)
        counts = np.bincount(parent_id, minlength=max(grains.id) + 1)

        set_back = counts[parent_id] < 2 and grains.phase_id == grains.name_to_id('Titanium (Alpha)')
        parent_grains[set_back].mean_orientation = grains[set_back].mean_orientation
        parent_grains.update()

        parent_grains, parent_id = parent_grains.merge(threshold=2.5 * np.pi / 180)

        parent_ebsd = alpha_ebsd.copy()
        parent_ebsd['indexed'].grain_id = parent_id[alpha_ebsd['indexed'].grain_id]

        plot_save(parent_grains, os.path.join(reconstruction_plot_folder, f"{local_ebsd_base_name}_Step2.png"))

        grain_pairs = parent_grains.neighbors(alpha='Titanium (Alpha)', beta='Titanium (Beta)')

        ori_alpha = parent_grains[grain_pairs[:, 0]].mean_orientation
        ori_beta = parent_grains[grain_pairs[:, 1]].mean_orientation

        parent_id, fit = triple_points.calc_parent(ori_alpha, ori_beta, beta2alpha, num_fit=2, threshold=2.5 * np.pi / 180)
        consistent_pairs = fit[:, 0] < 5 * np.pi / 180 and fit[:, 1] > 5 * np.pi / 180

        parent_id = parent_grains.majority_vote(grain_pairs[consistent_pairs, 0], parent_id[consistent_pairs, 0], max(parent_grains.id), strict=True)

        has_vote = ~np.isnan(parent_id)
        parent_grains[has_vote].mean_orientation = grains.variants(beta2alpha, parent_grains[has_vote].mean_orientation, parent_id[has_vote])
        parent_grains.update()

        parent_grains, parent_id = parent_grains.merge(threshold=5 * np.pi / 180)
        parent_ebsd['indexed'].grain_id = parent_id[parent_ebsd['indexed'].grain_id]

        plot_save(parent_grains, os.path.join(reconstruction_plot_folder, f"{local_ebsd_base_name}_Step3.png"))

        is_now_beta = (parent_grains.phase_id[max(1, parent_ebsd.grain_id)] == alpha_ebsd.name_to_id('Titanium (Beta)')) & (parent_ebsd.phase_id == alpha_ebsd.name_to_id('Titanium (Alpha)'))

        parent_ebsd[is_now_beta].orientations, fit = parent_ebsd[is_now_beta].calc_parent(parent_grains[parent_ebsd[is_now_beta].grain_id].mean_orientation, beta2alpha)

        plot_save(parent_ebsd, os.path.join(reconstruction_plot_folder, f"{local_ebsd_base_name}_Step4.png"))

        parent_grains, parent_ebsd.grain_id = parent_ebsd.calc_grains(angle=5 * np.pi / 180)
        parent_ebsd = parent_ebsd[parent_grains[parent_grains.grain_size > min_beta_size_pixels]]

        parent_grains, parent_ebsd.grain_id = parent_ebsd.calc_grains(angle=5 * np.pi / 180)
        parent_grains = parent_grains.smooth(iterations=5)

        F = HalfQuadraticFilter(alpha=0.1)
        parent_ebsd = parent_ebsd.smooth(F, fill=parent_grains)

        plot_save(parent_ebsd, os.path.join(reconstruction_plot_folder, f"{local_ebsd_base_name}_Step6.png"))

        grid_parent_ebsd = parent_ebsd.gridify()
        grid_parent_ebsd_rotations = grid_parent_ebsd.rotations
        grid_parent_ebsd_phase = grid_parent_ebsd.phase
        size_grid = grid_parent_ebsd.shape
        n_cols_x = size_grid[0]
        n_cols_y = size_grid[1]
        numel_rotations = np.prod(size_grid)

        linearized_euler_angles = np.zeros((numel_rotations, 3))
        linearized_phase = np.zeros(numel_rotations)
        counter = 0
        for ii in range(n_cols_x):
            for jj in range(n_cols_y):
                counter += 1
                rot = grid_parent_ebsd_rotations[ii, jj]
                if not np.isnan(rot.phi1):
                    linearized_euler_angles[counter - 1] = [rot.phi1, rot.Phi, rot.phi2]
                else:
                    linearized_euler_angles[counter - 1] = [0, 0, 0]
                linearized_phase[counter - 1] = grid_parent_ebsd_phase[ii, jj]

        out_file = os.path.join(beta_ebsd_folder, f"{local_ebsd_base_name}.dat")
        np.savetxt(out_file, np.column_stack((linearized_euler_angles, linearized_phase)), fmt='%f')

        plot_save(parent_grains, os.path.join(reconstruction_plot_folder, f"{local_ebsd_base_name}_Step5.png"))


def plot_save(obj, filename):
    fig, ax = plt.subplots(figsize=(10, 10))
    obj.plot(ax=ax)
    fig.savefig(filename)
    plt.close(fig)

if __name__ == "__main__":
    sample = "../A01_XY_100_nm.ang"
    alpha_ebsd_folder = "./alpha"
    beta_ebsd_folder = "./beta"
    print(f'\nAnalyzing sample: {sample}\n')
    step02(alpha_ebsd_folder, beta_ebsd_folder)
    print(f'\n\nFinished sample: {sample}\n')
