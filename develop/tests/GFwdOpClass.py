import numpy as np
from scipy.sparse import csr_matrix

class GFwdOp:
    def __init__(self, model, rays):
        self.__model = model
        self.__rays = rays  # list of source, rec, ray

        # set up kernel matrix dict
        print(set(ray.phase.name for ray in rays))
        self.__kernel_matrices = {item: [] for item in set(ray.phase.name for ray in rays)}

        # Calculate kernel matrix
        self.__calcMatrix__()
        
    def __calcMatrix__(self):
        # calculate kernels and add to dense matrix
        for ray in self.__rays:
            kernel = self.__model.mesh.compute_sensitivity_kernel(
                    ray, property_name=f'v{ray.phase.name[0].lower()}', attach_name=f'K_{ray.phase.name}_v{ray.phase.name[0].lower()}', epsilon=1e-6
                )
            self.__kernel_matrices[ray.phase.name].append(kernel)

        # transform dense matrices to sparse
        for phase, kernel_mat in self.__kernel_matrices.items():
            self.__kernel_matrices[phase] = csr_matrix(np.array(kernel_mat))
            print(f"{phase} Kernel Matrix shape: {self.__kernel_matrices[phase].shape}, nnz: {self.__kernel_matrices[phase].nnz}")

    def __computeKernels__(self, source_lat, source_lon, source_depth, receiver_lat, receiver_lon, ray):
        print(f"Source: ({source_lat}°, {source_lon}°, {source_depth} km)")
        print(f"Receiver: ({receiver_lat}°, {receiver_lon}°, 0 km)")
        print(f"Ray: {ray}")

        kernel = self.__model.mesh.compute_sensitivity_kernel(
                ray, property_name=f'v{ray.phase.name[0].lower()}', attach_name=f'K_{ray.phase.name}_v{ray.phase.name[0].lower()}', epsilon=1e-6
            )
        # Adds a new entry if phase not already in dict
        if ray.phase.name not in self.__kernel_matrices:
            self.__kernel_matrices[ray.phase.name] = []
        self.__kernel_matrices[ray.phase.name].append(kernel)

    def getNnz(self, phase=["all"]):
        # return number of non-zero entries in kernel matrix
        if phase == ["all"]:
            nnz = {ph: self.__kernel_matrices[ph].nnz for ph in self.__kernel_matrices}
        else:
            nnz = {ph: self.__kernel_matrices[ph].nnz for ph in phase if ph in self.__kernel_matrices}
        return nnz
    
    def get_voxelNum(self, phase=["all"]):
        # return number of voxels in kernel matrix
        if phase == ["all"]:
            voxels = {ph: self.__kernel_matrices[ph].shape[1] for ph in self.__kernel_matrices}
        else:
            voxels = {ph: self.__kernel_matrices[ph].shape[1] for ph in phase if ph in self.__kernel_matrices}
        return voxels      

    def __apply__(self, velocity_model):
        # compute travel times from kernels
        print("Computing travel times from kernels...")
        times = {}
        for phase, kernel_matrix in self.__kernel_matrices.items():
            if kernel_matrix.shape[0] > 0:
                print(kernel_matrix.shape[1])
                # use provided model or default
                times[phase] = kernel_matrix.dot(velocity_model)
                print(f"{phase} travel times: min {times[phase].min():.2f} s, max {times[phase].max():.2f} s")

        return times
