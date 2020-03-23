import matlab.engine
import matlab
import numpy as np

eng = matlab.engine.start_matlab()

# coordinates_file = 'truss/sample_input/Coordinates.csv'
# connectivity_file = 'truss/sample_input/Connectivity.csv'
# fixednodes_file = 'truss/sample_input/fixnodes.csv'
# loadn_file = 'truss/sample_input/loadnodes.csv'
# force_file = 'truss/sample_input/force.csv'

# density = matlab.double([7.12140021e-6])
# elastic_modulus = matlab.double([2850.0])

coordinates_file = 'truss/sample_input/coord_iscso.csv'
connectivity_file = 'truss/sample_input/connect_iscso.csv'
fixednodes_file = 'truss/sample_input/fixn_iscso.csv'
loadn_file = 'truss/sample_input/loadn_iscso.csv'
force_file = 'truss/sample_input/force.csv'
density = matlab.double([7121.4])
elastic_modulus = matlab.double([200e9])

coordinates = matlab.double(np.loadtxt(coordinates_file, delimiter=',').tolist())
connectivity = matlab.double(np.loadtxt(connectivity_file, delimiter=',').tolist())
fixed_nodes = matlab.double(np.loadtxt(fixednodes_file, delimiter=',').reshape(-1, 1).tolist())
load_nodes = matlab.double(np.loadtxt(loadn_file, delimiter=',').reshape(-1, 1).tolist())
force = matlab.double(np.loadtxt(force_file, delimiter=',').reshape(-1, 1).tolist())

# draw_truss(Coordinates, Connectivity, fixednodes, loadn, force)
weight, compliance, stress, strain = eng.run_fea(coordinates, connectivity, fixed_nodes, load_nodes, force, density,
                                                 elastic_modulus, nargout=4)

print(f"Weight = {weight}")
print(f"Compliance = {compliance}")
print(f"Stress = {stress}")
print(f"Strain = {strain}")
# weight, compliance, stress, strain
# Correct results: weight = 2.3038e-4 compliance=4.1549
# print(eng.plus(a,b))
