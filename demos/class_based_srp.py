import numpy as np
from scipy.sparse import csr_matrix
from random import seed
from itertools import product
from sensray import PlanetModel, CoordinateConverter

seed(0)

# Load model and create mesh
model = PlanetModel.from_standard_model('prem')
# Create mesh and save if not exist, otherwise load existing
mesh_path = "prem_mesh"
try:
    model.create_mesh(from_file=mesh_path)
    print(f"Loaded existing mesh from {mesh_path}")
except FileNotFoundError:
    print("Creating new mesh...")
    radii = [1221.5, 3480.0, 6371]
    H_layers = [1000, 1000, 600]
    model.create_mesh(mesh_size_km=1000, radii=radii, H_layers=H_layers)
    model.mesh.populate_properties(['vp', 'vs', 'rho'])
    model.mesh.save("prem_mesh")  # Save mesh to VT
print(f"Created mesh: {model.mesh.mesh.n_cells} cells")


# function to make points
def point(pointType="Source", minLat=-90, maxLat=90, minLon=-180, maxLon=180, minDepth=0, maxDepth=700):
    if pointType == "Source":
        lat = np.random.uniform(minLat, maxLat)
        lon = np.random.uniform(minLon, maxLon)
        depth = np.random.uniform(minDepth, maxDepth)  # depth in km
        return (lat, lon, depth)
    elif pointType == "Receiver":
        lat = np.random.uniform(minLat, maxLat)
        lon = np.random.uniform(minLon, maxLon)
        return (lat, lon)
    else:
        raise ValueError("pointType must be 'Source' or 'Receiver'")

# Generate source and receiver points and combinations
sources = [point("Source", minDepth=150, maxDepth=150) for _ in range(2)]
receivers = [point("Receiver", maxDepth=0) for _ in range(5)]
phases = ["P", "S", "ScS"]

srp = [prod + tuple([phases]) for prod in product(sources, receivers)]

# # store kernels
# p_kernels = []
# s_kernels = []
# scs_kernels = []


# def ComputeKernels(model, source_lat, source_lon, source_depth, receiver_lat, receiver_lon):
#     print(f"Source: ({source_lat}°, {source_lon}°, {source_depth} km)")
#     print(f"Receiver: ({receiver_lat}°, {receiver_lon}°, 0 km)")
    
#     # Compute great-circle plane normal for cross-sections
#     plane_normal = CoordinateConverter.compute_gc_plane_normal(
#         source_lat, source_lon, receiver_lat, receiver_lon
#     )

#     # Get ray paths for P and S waves
#     rays = model.taupy_model.get_ray_paths_geo(
#         source_depth_in_km=source_depth,
#         source_latitude_in_deg=source_lat,
#         source_longitude_in_deg=source_lon,
#         receiver_latitude_in_deg=receiver_lat,
#         receiver_longitude_in_deg=receiver_lon,
#         phase_list=["P", "S", "ScS"]
#     )

#     print(f"Found {len(rays)} ray paths:")
#     for i, ray in enumerate(rays):
#         print(f"  {i+1}. {ray.phase.name}: {ray.time:.2f} s, {len(ray.path)} points")

#     # only compute kernels if rays found
#     P_kernel, S_kernel, ScS_kernel = None, None, None
#     if len(rays) >= 3:
#         # plot rays
#         rays.plot()

#         # Compute and store path lengths for each ray
#         P_ray = rays[0]  # First ray (P wave)
#         S_ray = rays[1] if len(rays) > 1 else rays[0]  # Second ray (S wave)
#         ScS_ray = rays[2]

#         # Method 1: Simple computation and storage
#         P_lengths = model.mesh.add_ray_to_mesh(P_ray, "P_wave")
#         S_lengths = model.mesh.add_ray_to_mesh(S_ray, "S_wave")
#         ScS_lengths = model.mesh.add_ray_to_mesh(ScS_ray, "ScS_wave")

#         # print(f"P wave: {P_lengths.sum():.1f} km total, {np.count_nonzero(P_lengths)} cells")
#         # print(f"S wave: {S_lengths.sum():.1f} km total, {np.count_nonzero(S_lengths)} cells")
#         # print(f"ScS wave: {ScS_lengths.sum():.1f} km total, {np.count_nonzero(ScS_lengths)} cells")

#         # Show stored properties
#         ray_keys = [k for k in model.mesh.mesh.cell_data.keys() if 'ray_' in k]
#         # print(f"Stored ray properties: {ray_keys}")

#         # Compute sensitivity kernels for P and S waves
#         P_kernel = model.mesh.compute_sensitivity_kernel(
#             P_ray, property_name='vp', attach_name='K_P_vp', epsilon=1e-6
#         )
#         S_kernel = model.mesh.compute_sensitivity_kernel(
#             S_ray, property_name='vs', attach_name='K_S_vs', epsilon=1e-6
#         )
#         ScS_kernel = model.mesh.compute_sensitivity_kernel(
#             ScS_ray, property_name='vs', attach_name='K_ScS_vs', epsilon=1e-6
#         )

#     return P_kernel, S_kernel, ScS_kernel





'''
pass in model and list of (source, receiver, phases of interest) and computes the travel times
'''
class G:
    def __init__(self, model, srp):
        self.__model = model
        self.__srp = srp  # source, rec, phases of interest

        # set up kernel matrix dict
        self.__kernel_matrices = {}

        # # Set up matrix variables
        # self.__P_kernel_matrix = None
        # self.__S_kernel_matrix = None
        # self.__ScS_kernel_matrix = None

        # Calculate kernel matrix
        self.__calcMatrix__()
        
    def __calcMatrix__(self):
        # calculate kernels and add to dense matrix
        for (source_lat, source_lon, source_depth), (receiver_lat, receiver_lon), phase in self.__srp:
            self.__computeKernels__(model, source_lat, source_lon, source_depth, receiver_lat, receiver_lon, phases)

        # transform dense matrices to sparse
        for phase, kernel_mat in self.__kernel_matrices.items():
            self.__kernel_matrices[phase] = csr_matrix(np.array(kernel_mat))
        
            print(f"{phase} Kernel Matrix shape: {self.__kernel_matrices[phase].shape}, nnz: {self.__kernel_matrices[phase].nnz}")
        
        # self.__P_kernel_matrix = csr_matrix(np.array(p_kernels))
        # self.__S_kernel_matrix = csr_matrix(np.array(s_kernels))
        # self.__ScS_kernel_matrix = csr_matrix(np.array(scs_kernels))

        # print(f"P Kernel Matrix shape: {self.__P_kernel_matrix.shape}, nnz: {P_kernel_matrix.nnz}")
        # print(f"S Kernel Matrix shape: {self.__S_kernel_matrix.shape}, nnz: {S_kernel_matrix.nnz}")
        # print(f"ScS Kernel Matrix shape: {self.__ScS_kernel_matrix.shape}, nnz: {ScS_kernel_matrix.nnz}")


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


        '''
        # only compute kernels if rays found
        # for ray in rays:
        #     #length = model.mesh.add_ray_to_mesh(P_ray, "P_wave")


        # P_kernel, S_kernel, ScS_kernel = None, None, None
        kernel = None
        #if len(rays) >= 3:
        if rays:
            # plot rays
            # rays.plot()

            # Compute and store path lengths for each ray
            P_ray = rays[0]  # First ray (P wave)
            S_ray = rays[1] if len(rays) > 1 else rays[0]  # Second ray (S wave)
            ScS_ray = rays[2]

            # Method 1: Simple computation and storage
            # P_lengths = model.mesh.add_ray_to_mesh(P_ray, "P_wave")
            # S_lengths = model.mesh.add_ray_to_mesh(S_ray, "S_wave")
            # ScS_lengths = model.mesh.add_ray_to_mesh(ScS_ray, "ScS_wave")
            lengths = model.mesh.add_ray_to_mesh(rays[0], f"{phase}_wave")

            # print(f"P wave: {P_lengths.sum():.1f} km total, {np.count_nonzero(P_lengths)} cells")
            # print(f"S wave: {S_lengths.sum():.1f} km total, {np.count_nonzero(S_lengths)} cells")
            # print(f"ScS wave: {ScS_lengths.sum():.1f} km total, {np.count_nonzero(ScS_lengths)} cells")

            # Show stored properties
            ray_keys = [k for k in model.mesh.mesh.cell_data.keys() if 'ray_' in k]
            # print(f"Stored ray properties: {ray_keys}")

            
            # Compute sensitivity kernels for P and S waves
            P_kernel = model.mesh.compute_sensitivity_kernel(
                P_ray, property_name='vp', attach_name='K_P_vp', epsilon=1e-6
            )
            S_kernel = model.mesh.compute_sensitivity_kernel(
                S_ray, property_name='vs', attach_name='K_S_vs', epsilon=1e-6
            )
            ScS_kernel = model.mesh.compute_sensitivity_kernel(
                ScS_ray, property_name='vs', attach_name='K_ScS_vs', epsilon=1e-6
            )

            p_kernels.append(P_kernel)
            s_kernels.append(S_kernel)
            scs_kernels.append(ScS_kernel)
        '''

    def __apply__(self, model):
        # compute travel times from kernels
        print("Computing travel times from kernels...")
        times = {}
        for phase, kernel_matrix in self.__kernel_matrices.items():
            if kernel_matrix.shape[0] > 0:
                times[phase] = kernel_matrix.dot(model.mesh.mesh.cell_data['v' + phase[0].lower()])
                print(f"{phase} travel times: min {times[phase].min():.2f} s, max {times[phase].max():.2f} s")


        # if self.__P_kernel_matrix.shape[0] > 0:
        #     times['P'] = self.__P_kernel_matrix.dot(model.mesh.mesh.cell_data['vp'])
        #     print(f"P travel times: min {times['P'].min():.2f} s, max {times['P'].max():.2f} s")
        # if self.__S_kernel_matrix.shape[0] > 0:
        #     times['S'] = self.__S_kernel_matrix.dot(model.mesh.mesh.cell_data['vs'])
        #     print(f"S travel times: min {times['S'].min():.2f} s, max {times['S'].max():.2f} s")
        # if self.__ScS_kernel_matrix.shape[0] > 0:
        #     times['ScS'] = self.__ScS_kernel_matrix.dot(model.mesh.mesh.cell_data['vs'])
        #     print(f"ScS travel times: min {times['ScS'].min():.2f} s, max {times['ScS'].max():.2f} s")

        return times

# testing with one source-receiver pair - same as initial test
source_lat, source_lon, source_depth = 0.0, 0.0, 150.0  # Equator, 150 km depth
receiver_lat, receiver_lon = 30.0, 45.0  # Surface station
srp = [((source_lat, source_lon, source_depth), (receiver_lat, receiver_lon), ["P", "S", "ScS"])]

appl = G(model, srp)
travel_times = appl.__apply__(model)
print(travel_times)