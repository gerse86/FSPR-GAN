"""
MetricChecker for validating 3D generated samples against real data.
Computes errors for porosity, pore/throat counts, two-point correlation, and size distributions.
"""

import os
import csv
import numpy as np
from Metric import (
    calculate_stats_for_folder, get_all_tpc_data, calculate_average_tpc,
    align_tpc_data, calculate_tpc_relative_error, extract_data_parallel
)
import porespy as ps
ps.settings.tqdm['disable'] = True


class MetricChecker:
    """
    Checks the quality of generated 3D volumes against a real dataset.

    Attributes:
        real_dir (str): Path to the real .raw files.
        shape (tuple): Volume shape (D, H, W).
        results_dir (str): Directory to save metric results.
        metrics_order (list): Ordered list of metric names for CSV output.
        tolerance (dict): Tolerance values for each metric (percentage or absolute).
    """

    def __init__(self, real_dir, shape=(128, 128, 128), results_dir='results'):
        """
        Initializes the MetricChecker.

        Args:
            real_dir (str): Directory containing real .raw volumes.
            shape (tuple): Shape of the volumes (D, H, W).
            results_dir (str): Root directory for saving results.
        """
        self.real_dir = real_dir
        self.shape = shape
        self.results_dir = results_dir
        self.metrics_order = [
            "step",
            "Total porosity",
            "Connected porosity",
            "Total number of pores",
            "Number of connected pores",
            "Number of throats",
            "TPC",
            "equivalent diameters of pores",
            "equivalent diameters of throats",
            "Total 1-6 (Relative Error)",
            "Total:7-8 (absolute error)"
        ]
        self.tolerance = {
            "Total porosity": 1,           # relative error percentage
            "Connected porosity": 1,
            "Total number of pores": 15,
            "Number of connected pores": 15,
            "Number of throats": 20,
            "TPC": 10,
            "equivalent diameters of pores": 10,    # absolute error (max count difference)
            "equivalent diameters of throats": 10
        }

        self.real_stats_cached = False
        self.real_all_stats = None
        self.real_avg_stats = None
        self.real_avg_tpc_dist = None
        self.real_avg_tpc_pdf = None
        self.real_pore_data = None
        self.real_throat_data = None

        os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
        self.check_csv_path = os.path.join(results_dir, 'metrics', 'check.csv')

    def _cache_real_samples_stats(self):
        """
        Precomputes and caches all real dataset statistics (porosity, TPC, pore/throat sizes).
        This avoids recomputing for every check.
        """
        if self.real_stats_cached:
            return
        _, self.real_all_stats, self.real_avg_stats = calculate_stats_for_folder(self.real_dir, self.shape)
        real_tpc_distances, real_tpc_pdfs = get_all_tpc_data(self.real_dir, shape=self.shape)
        self.real_avg_tpc_dist, self.real_avg_tpc_pdf = calculate_average_tpc(real_tpc_distances, real_tpc_pdfs)
        self.real_pore_data, self.real_throat_data = extract_data_parallel(self.real_dir, self.shape)
        self.real_stats_cached = True

    def generate_samples(self, generator, num_samples=300, temp_dir='temp_fake_samples'):
        """
        Generates a specified number of samples using the provided generator and saves them as .raw files.

        Args:
            generator: The generator object with a `generate_volumes(batch_size)` method.
            num_samples (int): Total number of samples to generate.
            temp_dir (str): Temporary directory to store generated .raw files.

        Returns:
            str: Path to the temporary directory containing the generated files.
        """
        os.makedirs(temp_dir, exist_ok=True)
        batch_size = min(30, num_samples)
        total_generated = 0

        while total_generated < num_samples:
            current_batch_size = min(batch_size, num_samples - total_generated)
            volumes, _ = generator.generate_volumes(current_batch_size)

            for i in range(current_batch_size):
                vol = volumes[i, 0].detach().cpu().numpy()
                vol = (vol > 0).astype(np.uint8)
                file_path = os.path.join(temp_dir, f'fake_{total_generated + i}.raw')
                vol.tofile(file_path)

            total_generated += current_batch_size

        return temp_dir

    def check(self, generator, step, num_samples=300):
        """
        Evaluates the generator against real data and logs the metric errors to a CSV file.

        The check proceeds stepwise:
            1. Porosity and pore/throat counts (relative errors).
            2. Two-point correlation (TPC) relative error.
            3. Pore and throat equivalent diameter histograms (maximum absolute count difference).

        Args:
            generator: The generator object.
            step (int): Current training step (used for logging).
            num_samples (int): Number of samples to generate for evaluation.
        """
        self._cache_real_samples_stats()
        if not self.real_stats_cached:
            return

        temp_dir = os.path.join(self.results_dir, 'temp_fake_samples')
        fake_dir = self.generate_samples(generator, num_samples, temp_dir)

        error_results = {name: "Not calculated" for name in self.metrics_order}
        error_results["step"] = step
        rel_errors_1_6 = []
        base_passed = False

        # 1. Compute porosity and pore/throat counts
        _, fake_all_stats, fake_avg_stats = calculate_stats_for_folder(fake_dir, self.shape)

        metrics = [
            ("Total porosity", self.real_avg_stats[0], True),
            ("Connected porosity", self.real_avg_stats[1], True),
            ("Total number of pores", self.real_avg_stats[2], False),
            ("Number of connected pores", self.real_avg_stats[3], False),
            ("Number of throats", self.real_avg_stats[4], False)
        ]

        for i, (name, real_val, is_porosity) in enumerate(metrics):
            fake_val = fake_avg_stats[i]

            if real_val == 0:
                rel_error = float('inf')
            else:
                rel_error = abs(fake_val - real_val) / real_val * 100

            rounded_error = round(rel_error, 2)
            error_results[name] = rounded_error
            rel_errors_1_6.append(rounded_error)

            if rel_error > self.tolerance[name]:
                rel_errors_1_6.pop()
                break
        else:
            base_passed = True

        # 2. Two-point correlation (TPC) if base metrics passed
        tpc_rel_error = None
        if base_passed:
            fake_tpc_distances, fake_tpc_pdfs = get_all_tpc_data(fake_dir, shape=self.shape)
            fake_avg_tpc_dist, fake_avg_tpc_pdf = calculate_average_tpc(fake_tpc_distances, fake_tpc_pdfs)

            if base_passed:
                common_tpc_dist, aligned_real_tpc, aligned_fake_tpc = align_tpc_data(
                    self.real_avg_tpc_dist, self.real_avg_tpc_pdf,
                    fake_avg_tpc_dist, fake_avg_tpc_pdf
                )

            if base_passed:
                tpc_rel_errors = calculate_tpc_relative_error(aligned_real_tpc, aligned_fake_tpc)
                if tpc_rel_errors is None:
                    base_passed = False
                else:
                    valid_tpc_errors = tpc_rel_errors[~np.isnan(tpc_rel_errors)]
                    if len(valid_tpc_errors) == 0:
                        base_passed = False
                    else:
                        tpc_rel_error = np.mean(valid_tpc_errors).astype(np.float32)
                        tpc_rounded_error = round(tpc_rel_error, 2)
                        error_results["TPC"] = tpc_rounded_error
                        rel_errors_1_6.append(tpc_rounded_error)

                        if tpc_rel_error > self.tolerance["TPC"]:
                            base_passed = False
                            rel_errors_1_6.pop()

        # 3. Pore and throat equivalent diameter histograms (absolute errors)
        abs_errors_7_8 = []
        if base_passed and tpc_rel_error is not None:
            fake_pore_data, fake_throat_data = extract_data_parallel(fake_dir, self.shape)

            if not self.real_pore_data or not fake_pore_data or not self.real_throat_data or not fake_throat_data:
                base_passed = False
            else:
                # Pore diameters
                bins_step_pore = 5
                all_pore_data = np.concatenate([np.concatenate(self.real_pore_data), np.concatenate(fake_pore_data)]).astype(np.float32)
                min_pore = np.min(all_pore_data).astype(np.float32)
                max_pore = np.max(all_pore_data).astype(np.float32)
                bins_pore = np.arange(
                    np.floor(min_pore / bins_step_pore) * bins_step_pore,
                    np.ceil(max_pore / bins_step_pore) * bins_step_pore + bins_step_pore,
                    bins_step_pore,
                    dtype=np.float32
                )

                real_pore_counts = [np.histogram(data, bins=bins_pore)[0] for data in self.real_pore_data]
                real_pore_avg = np.mean(real_pore_counts, axis=0).astype(np.float32) if real_pore_counts else np.zeros_like(bins_pore[:-1], dtype=np.float32)

                fake_pore_counts = [np.histogram(data, bins=bins_pore)[0] for data in fake_pore_data]
                fake_pore_avg = np.mean(fake_pore_counts, axis=0).astype(np.float32) if fake_pore_counts else np.zeros_like(bins_pore[:-1], dtype=np.float32)

                epsilon = 1e-8
                valid_pore_mask = real_pore_avg >= epsilon
                if np.sum(valid_pore_mask) == 0:
                    base_passed = False
                else:
                    pore_errors = np.abs(fake_pore_avg[valid_pore_mask] - real_pore_avg[valid_pore_mask]).astype(np.float32)
                    pore_avg_error = np.max(pore_errors).astype(np.float32)
                    pore_rounded_error = round(pore_avg_error, 2)
                    error_results["equivalent diameters of pores"] = pore_rounded_error

                    if pore_avg_error > self.tolerance["equivalent diameters of pores"]:
                        base_passed = False
                    else:
                        abs_errors_7_8.append(pore_rounded_error)

                # Throat diameters (only if pore check passed)
                if base_passed:
                    bins_step_throat = 1
                    all_throat_data = np.concatenate([np.concatenate(self.real_throat_data), np.concatenate(fake_throat_data)]).astype(np.float32)
                    min_throat = np.min(all_throat_data).astype(np.float32)
                    max_throat = np.max(all_throat_data).astype(np.float32)
                    bins_throat = np.arange(
                        np.floor(min_throat / bins_step_throat) * bins_step_throat,
                        np.ceil(max_throat / bins_step_throat) * bins_step_throat + bins_step_throat,
                        bins_step_throat,
                        dtype=np.float32
                    )

                    real_throat_counts = [np.histogram(data, bins=bins_throat)[0] for data in self.real_throat_data]
                    real_throat_avg = np.mean(real_throat_counts, axis=0).astype(np.float32) if real_throat_counts else np.zeros_like(bins_throat[:-1], dtype=np.float32)

                    fake_throat_counts = [np.histogram(data, bins=bins_throat)[0] for data in fake_throat_data]
                    fake_throat_avg = np.mean(fake_throat_counts, axis=0).astype(np.float32) if fake_throat_counts else np.zeros_like(bins_throat[:-1], dtype=np.float32)

                    valid_throat_mask = real_throat_avg >= epsilon
                    if np.sum(valid_throat_mask) == 0:
                        base_passed = False
                    else:
                        throat_errors = np.abs(fake_throat_avg[valid_throat_mask] - real_throat_avg[valid_throat_mask]).astype(np.float32)
                        throat_avg_error = np.max(throat_errors).astype(np.float32)
                        throat_rounded_error = round(throat_avg_error, 2)
                        error_results["equivalent diameters of throats"] = throat_rounded_error

                        if throat_avg_error > self.tolerance["equivalent diameters of throats"]:
                            base_passed = False
                        else:
                            abs_errors_7_8.append(throat_rounded_error)

        # Summarize grouped errors
        if len(rel_errors_1_6) == 6:
            total_rel_1_6 = round(sum(rel_errors_1_6), 2)
            error_results["Total 1-6 (Relative Error)"] = total_rel_1_6
        else:
            error_results["Total 1-6 (Relative Error)"] = "Not fully qualified"

        if len(abs_errors_7_8) == 2:
            total_abs_7_8 = round(sum(abs_errors_7_8), 2)
            error_results["Total:7-8 (absolute error)"] = total_abs_7_8
        else:
            error_results["Total:7-8 (absolute error)"] = "Not fully qualified"

        # Write results to CSV
        file_exists = os.path.isfile(self.check_csv_path) and os.path.getsize(self.check_csv_path) > 0
        with open(self.check_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_order)
            if not file_exists:
                writer.writeheader()
            writer.writerow(error_results)

        # Cleanup temporary generated files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)