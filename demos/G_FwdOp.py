import numpy as np
from scipy.sparse import csr_matrix
from sensray import CoordinateConverter


class G:
    def __init__(self, model, srp):
        self.__model = model
        self.__srp = srp  # list of source, rec, phases of interest

        # set up kernel matrix dict
        self.__kernel_matrices = {}

        # Calculate kernel matrix
        self.__calcMatrix__()
        
    def __calcMatrix__(self):
        # calculate kernels and add to dense matrix
        for (source_lat, source_lon, source_depth), (receiver_lat, receiver_lon), phases in self.__srp:
            self.__computeKernels__(self.__model, source_lat, source_lon, source_depth, receiver_lat, receiver_lon, phases)

        # transform dense matrices to sparse
        for phase, kernel_mat in self.__kernel_matrices.items():
            self.__kernel_matrices[phase] = csr_matrix(np.array(kernel_mat))
            print(f"{phase} Kernel Matrix shape: {self.__kernel_matrices[phase].shape}, nnz: {self.__kernel_matrices[phase].nnz}")

    def __computeKernels__(self, model, source_lat, source_lon, source_depth, receiver_lat, receiver_lon, phases):
        print(f"Source: ({source_lat}°, {source_lon}°, {source_depth} km)")
        print(f"Receiver: ({receiver_lat}°, {receiver_lon}°, 0 km)")
        print(f"Phases: {phases}")
        
        # Compute great-circle plane normal for cross-sections
        plane_normal = CoordinateConverter.compute_gc_plane_normal(
            source_lat, source_lon, receiver_lat, receiver_lon
        )

        # Get ray paths for P and S waves
        rays = model.taupy_model.get_ray_paths_geo(
            source_depth_in_km=source_depth,
            source_latitude_in_deg=source_lat,
            source_longitude_in_deg=source_lon,
            receiver_latitude_in_deg=receiver_lat,
            receiver_longitude_in_deg=receiver_lon,
            phase_list=phases
        )

        print(f"Found {len(rays)} ray paths:")
        for i, ray in enumerate(rays):
            print(f"  {i+1}. {ray.phase.name}: {ray.time:.2f} s, {len(ray.path)} points")
            length = model.mesh.add_ray_to_mesh(ray, f"{ray.phase.name}_wave")
            kernel = model.mesh.compute_sensitivity_kernel(
                ray, property_name=f'v{ray.phase.name[0].lower()}', attach_name=f'K_{ray.phase.name}_v{ray.phase.name[0].lower()}', epsilon=1e-6
            )
            # Adds a new entry if phase not already in dict
            if ray.phase.name not in self.__kernel_matrices:
                self.__kernel_matrices[ray.phase.name] = []
            self.__kernel_matrices[ray.phase.name].append(kernel)


    def __apply__(self, model=None):
        # compute travel times from kernels
        print("Computing travel times from kernels...")
        times = {}
        for phase, kernel_matrix in self.__kernel_matrices.items():
            if kernel_matrix.shape[0] > 0:
                # use provided model or default
                model = self.__model if model is None else model
                times[phase] = kernel_matrix.dot(model.mesh.mesh.cell_data['v' + phase[0].lower()])
                print(f"{phase} travel times: min {times[phase].min():.2f} s, max {times[phase].max():.2f} s")

        return times