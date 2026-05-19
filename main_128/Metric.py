import os
import numpy as np
import porespy as ps
import openpnm as op
from scipy.interpolate import make_interp_spline
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_WORKERS = os.cpu_count() or 4

def process_file(file_path, shape=(128, 128, 128), dtype=np.uint8):
    """Process a single 3D .raw file and extract the pore network and properties"""

    vol_3d = np.fromfile(file_path, dtype=dtype).reshape(shape)
    vol_3d_inverted = vol_3d == 0  # The pores are True, while the solid is False.

    snow = ps.networks.snow2(vol_3d_inverted, boundary_width=0, voxel_size=1)
    pn = op.io.network_from_porespy(snow.network)

    throat_conns = pn['throat.conns']
    num_pores = pn['pore.coords'].shape[0]
    all_pores = np.arange(num_pores)

    conn_counts = np.zeros(num_pores, dtype=int)

    for conn in throat_conns:
        conn_counts[conn[0]] += 1
        conn_counts[conn[1]] += 1
    connected_pores = all_pores[conn_counts > 0]

    pore_volume = pn['pore.volume'][connected_pores].astype(np.float32)
    pore_surface_area = pn['pore.surface_area'][connected_pores].astype(np.float32)
    pore_equivalent_diameter = pn['pore.equivalent_diameter'][connected_pores].astype(np.float32)
    throat_length = pn['throat.total_length'].astype(np.float32)
    throat_surface_area = (pn['throat.perimeter'] * throat_length).astype(np.float32)
    throat_equivalent_diameter = pn['throat.equivalent_diameter'].astype(np.float32)

    return {
        'pore_volume': pore_volume,
        'pore_surface_area': pore_surface_area,
        'pore_equivalent_diameter': pore_equivalent_diameter,
        'throat_length': throat_length,
        'throat_surface_area': throat_surface_area,
        'throat_equivalent_diameter': throat_equivalent_diameter,
        'file_name': os.path.basename(file_path),
        'vol_3d_inverted': vol_3d_inverted,
        'vol_3d': vol_3d
    }


def calculate_porosity_and_stats(pn, shape=(128, 128, 128), voxel_size=1):
    """Calculate the statistical indicators such as porosity and throat number of 3D volumetric data"""

    total_pore_volume = np.sum(pn['pore.volume']).astype(np.float32) * (voxel_size ** 3)
    total_volume = np.prod(shape).astype(np.float32) * (voxel_size ** 3)
    total_porosity = (total_pore_volume / total_volume).astype(np.float32)

    throat_conns = pn['throat.conns']
    num_pores = pn['pore.coords'].shape[0]
    all_pores = np.arange(num_pores)

    conn_counts = np.zeros(num_pores, dtype=int)
    for conn in throat_conns:
        conn_counts[conn[0]] += 1
        conn_counts[conn[1]] += 1

    connected_pores = all_pores[conn_counts > 0]
    connected_pore_volume = np.sum(pn['pore.volume'][connected_pores]).astype(np.float32) * (voxel_size ** 3)
    connected_porosity = (connected_pore_volume / total_volume).astype(np.float32)
    num_throats = pn['throat.conns'].shape[0]

    return total_porosity, connected_porosity, num_pores, len(connected_pores), num_throats


def process_stats_file(args):
    file_path, shape = args
    try:
        file_name = os.path.basename(file_path)
        vol_3d = np.fromfile(file_path, np.uint8).reshape(shape)
        vol_3d_inverted = vol_3d == 0
        snow = ps.networks.snow2(vol_3d_inverted, boundary_width=0, voxel_size=1)
        pn = op.io.network_from_porespy(snow.network)
        stats = calculate_porosity_and_stats(pn, shape)
        return file_name, stats
    except Exception as e:
        print(f"File {file_path} had an error in TPC calculation: {e}")

        return None, None


def calculate_stats_for_folder(folder_path, shape=(128, 128, 128)):

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.raw')]
    all_stats = []
    all_file_names = []

    # Parallel processing of files
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_stats_file, (path, shape)): path for path in file_paths}
        for future in as_completed(futures):
            file_name, stats = future.result()
            if file_name and stats:
                all_file_names.append(file_name)
                all_stats.append(stats)

    avg_total_porosity = np.mean([s[0] for s in all_stats]).astype(np.float32) if all_stats else 0.0
    avg_connected_porosity = np.mean([s[1] for s in all_stats]).astype(np.float32) if all_stats else 0.0
    avg_total_pores = np.mean([s[2] for s in all_stats]).astype(np.float32) if all_stats else 0.0
    avg_connected_pores = np.mean([s[3] for s in all_stats]).astype(np.float32) if all_stats else 0.0
    avg_throat_numbers = np.mean([s[4] for s in all_stats]).astype(np.float32) if all_stats else 0.0

    return all_file_names, all_stats, (avg_total_porosity, avg_connected_porosity,
                                       avg_total_pores, avg_connected_pores, avg_throat_numbers)


def process_tpc_file(args):

    file_path, shape, bins, voxel_size = args
    try:
        vol_3d = np.fromfile(file_path, dtype=np.uint8).reshape(shape)
        vol_3d_inverted = vol_3d == 0
        tpc_data = ps.metrics.two_point_correlation(
            im=vol_3d_inverted,
            bins=bins,
            voxel_size=voxel_size
        )
        return tpc_data.distance.astype(np.float32), tpc_data.pdf.astype(np.float32)
    except Exception as e:
        print(f"File {file_path} had an error in TPC calculation: {e}")

        return None, None


def get_all_tpc_data(folder_path, shape=(128, 128, 128), bins=120, voxel_size=1):

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.raw')]
    all_distances = []
    all_pdfs = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_tpc_file, (path, shape, bins, voxel_size)): path
            for path in file_paths
        }
        for future in as_completed(futures):
            dist, pdf = future.result()
            if dist is not None and pdf is not None:
                all_distances.append(dist)
                all_pdfs.append(pdf)

    return all_distances, all_pdfs


def calculate_average_tpc(all_distances, all_pdfs):
    if not all_distances or not all_pdfs:
        return None, None

    min_dist = np.min([d.min() for d in all_distances]).astype(np.float32)
    max_dist = np.max([d.max() for d in all_distances]).astype(np.float32)
    common_dist = np.linspace(min_dist, max_dist, 500, dtype=np.float32)
    avg_pdf = np.zeros_like(common_dist, dtype=np.float32)
    count = 0

    for dist, pdf in zip(all_distances, all_pdfs):

        try:
            spline = make_interp_spline(dist, pdf, k=3)
            interp_pdf = spline(common_dist).astype(np.float32)
            avg_pdf += interp_pdf
            count += 1
        except Exception as e:
            print(f"Error occurred during interpolation processing: {e}")
            continue

    if count == 0:
        return None, None
    avg_pdf /= count

    return common_dist, avg_pdf.astype(np.float32)


def align_tpc_data(real_dist, real_pdf, fake_dist, fake_pdf):

    min_dist = min(real_dist.min(), fake_dist.min()).astype(np.float32)
    max_dist = max(real_dist.max(), fake_dist.max()).astype(np.float32)
    common_dist = np.linspace(min_dist, max_dist, 500, dtype=np.float32)

    try:
        real_spline = make_interp_spline(real_dist, real_pdf, k=3)
        fake_spline = make_interp_spline(fake_dist, fake_pdf, k=3)

        aligned_real = real_spline(common_dist).astype(np.float32)
        aligned_fake = fake_spline(common_dist).astype(np.float32)

        real_mask = (common_dist < real_dist.min()) | (common_dist > real_dist.max())
        fake_mask = (common_dist < fake_dist.min()) | (common_dist > fake_dist.max())
        aligned_real[real_mask] = 0.0
        aligned_fake[fake_mask] = 0.0

        return common_dist, aligned_real, aligned_fake

    except Exception as e:
        print(f"TPC data alignment failed: {str(e)}")

        return None, None, None


def calculate_tpc_relative_error(real_pdf, fake_pdf, epsilon=1e-8):
    """Calculate the relative error of the true and false sample TPC"""
    if len(real_pdf) != len(fake_pdf):
        print("Error: The TPC lengths of the real sample and the output sample do not match.")
        return None

    mask = real_pdf < epsilon
    relative_error = np.full_like(real_pdf, np.nan, dtype=np.float32)

    valid_mask = ~mask
    relative_error[valid_mask] = (
            np.abs(fake_pdf[valid_mask] - real_pdf[valid_mask])
            / real_pdf[valid_mask]
            * 100
    ).astype(np.float32)

    return relative_error


def extract_pore_throat_data(file_path, shape):
    try:
        result = process_file(file_path, shape)
        return (result['pore_equivalent_diameter'],
                result['throat_equivalent_diameter'])
    except Exception as e:
        print(f"File {file_path} data extraction failed: {e}")

        return None, None


def extract_data_parallel(folder_path, shape):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.raw')]
    pore_data_list = []
    throat_data_list = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_pore_throat_data, path, shape): path for path in file_paths}
        for future in as_completed(futures):
            pore_data, throat_data = future.result()
            if pore_data is not None and len(pore_data) > 0:
                pore_data_list.append(pore_data)
            if throat_data is not None and len(throat_data) > 0:
                throat_data_list.append(throat_data)

    return pore_data_list, throat_data_list