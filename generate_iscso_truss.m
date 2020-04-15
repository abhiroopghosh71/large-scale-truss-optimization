clear
close all

n_nodes = 19 * 4;
truss_length = 18*4;  % in m
truss_width = 4;  % in m
density = 7121.4;  % kg/m3
elastic_modulus = 200e9;  % Pa
yield_stress = 248.2e6;  % Pa

connectivity = [];
% load_nodes = [1; 39; 19; 57];
% fixed_nodes = [1; 19; 20; 38; 39; 57; 58; 76];
% fixed_nodes = [20; 38; 58; 76];
fixed_nodes = [1; 19; 39; 57];

% Add all unsupported nodes as load nodes
load_nodes = [];
for ii = 1:n_nodes
    if sum(fixed_nodes == ii) ~= 1
        load_nodes = [load_nodes; ii];
    end
end
force_xyz = [5000, 1000, -5000];  % in N
coord = zeros(n_nodes, 3);
member_radius = 0.015;  % in m


%% Create connectivity matrix
% Connections in x-axis
for ii = 1:18
    connectivity = [connectivity; [ii, ii+1, member_radius]];
end
for ii = 20:37
    connectivity = [connectivity; [ii, ii+1, member_radius]];
end
for ii = 39:56
    connectivity = [connectivity; [ii, ii+1, member_radius]];
end
for ii = 58:75
    connectivity = [connectivity; [ii, ii+1, member_radius]];
end

% The cross shapes at two ends
connectivity = [connectivity; [1, 58, member_radius]];
connectivity = [connectivity; [20, 39, member_radius]];
connectivity = [connectivity; [19, 76, member_radius]];
connectivity = [connectivity; [38, 57, member_radius]];

% Connections in y-axis and z-axis
for ii = 1:19
    connectivity = [connectivity; [ii, ii+19, member_radius]]; % y-axis
    connectivity = [connectivity; [ii, ii+38, member_radius]]; % z-axis
end
for ii = 58:76
    connectivity = [connectivity; [ii-19, ii, member_radius]]; % y-axis
    connectivity = [connectivity; [ii-38, ii, member_radius]]; % z-axis
end

% Cross-connectivities along x-axis
for ii = 1:18
    connectivity = [connectivity; [ii, ii+39, member_radius]];
end
for ii = 20:37
    connectivity = [connectivity; [ii, ii+39, member_radius]];
end
for ii = 39:56
    connectivity = [connectivity; [ii, ii-37, member_radius]];
end
for ii = 58:75
    connectivity = [connectivity; [ii, ii-37, member_radius]];
end

% Cross-connectivities along z-axis
for ii = 20:28
    connectivity = [connectivity; [ii-18, ii, member_radius]];
end
for ii = 30:38
    connectivity = [connectivity; [ii-20, ii, member_radius]];
end
for ii = 58:66
    connectivity = [connectivity; [ii-18, ii, member_radius]];
end
for ii = 68:76
    connectivity = [connectivity; [ii-20, ii, member_radius]];
end

%% Create coordinate matrix
for ii = 1:19
    coord(ii,  :) = [(ii-1)*4, 0, 0];
end
for ii = 20:38
    coord(ii,  :) = [(ii-19-1)*4, 0, 4];
end
for ii = 39:57
    coord(ii,  :) = [(ii-38-1)*4, 4, 0];
end
for ii = 58:76
    coord(ii,  :) = [(ii-57-1)*4, 4, 4];
end

coordinates_file = 'truss/sample_input/coord_iscso.csv';
connectivity_file = 'truss/sample_input/connect_iscso.csv';
fixednodes_file = 'truss/sample_input/fixn_iscso.csv';
loadn_file = 'truss/sample_input/loadn_iscso.csv';
force_file = 'truss/sample_input/force_iscso.csv';
workspace_mat_file = 'truss/sample_input/workspace_iscso.mat';
% 
% writematrix(coord, coordinates_file, 'Delimiter',',')
% writematrix(connectivity, connectivity_file, 'Delimiter',',')
% writematrix(fixed_nodes, fixednodes_file)
% writematrix(load_nodes, loadn_file)
% writematrix(force_xyz, force_file, 'Delimiter',',')
% save(workspace_mat_file)
fprintf("Files not written\n")

[weight, compliance, stress, strain, U, x0_new] = run_fea(coord, connectivity, fixed_nodes, load_nodes, force_xyz, density, elastic_modulus);
draw_truss(coord, connectivity, fixed_nodes, load_nodes, force_xyz)

% x0_new_scaled = x0_new;
% x0_new_scaled(1:19, 3) = x0_new_scaled(1:19, 3) * 1000;
% x0_new_scaled(39:57, 3) = x0_new_scaled(39:57, 3) * 1000;
% draw_truss(x0_new_scaled, connectivity, fixed_nodes, load_nodes, force_xyz)


%% Test of winning iscso 2019 design
z = [3500
1623
-174
-1685
-3019
-4075
-4896
-5552
-6080
-6204
]';
z = z/1000;
coord_new = coord;
coord_new(1:10, 3) = z;
coord_new(11:19, 3) = flip(z(1:end-1));
coord_new(39:48, 3) = z;
coord_new(49:57, 3) = flip(z(1:end-1));
connec_new = connectivity;
connec_new(:, 3) = 0.017433732007913;  % Average radius of iscso 2019 winner

[weight_iscso, compliance_iscso, stress_iscso, strain_iscso, U_iscso, x0_new_iscso] = run_fea(coord_new, connec_new, fixed_nodes, load_nodes, force_xyz, density, elastic_modulus);

draw_truss(coord_new, connec_new, fixed_nodes, load_nodes, force_xyz)
title('ISCSO 2019 winner approx design')

z_inverted = z;
z_r = z_inverted(1);
z_max_depth = abs(z_inverted(10) - z_r);
z_inverted = z_r + abs(z_r - z_inverted) - z_max_depth;
coord_new_inverted = coord;
coord_new_inverted(1:10, 3) = z_inverted;
coord_new_inverted(11:19, 3) = flip(z_inverted(1:end-1));
coord_new_inverted(39:48, 3) = z_inverted;
coord_new_inverted(49:57, 3) = flip(z_inverted(1:end-1));
connec_new_inverted = connectivity;
connec_new_inverted(:, 3) = 0.017433732007913;  % Average radius of iscso 2019 winner

[weight_iscso_inverted, compliance_iscso_inverted, stress_iscso_inverted, strain_iscso_inverted, U_iscso_inverted, x0_new_iscso_inverted] = ...
    run_fea(coord_new_inverted, connec_new_inverted, fixed_nodes, load_nodes, force_xyz, density, elastic_modulus);

draw_truss(coord_new_inverted, connec_new_inverted, fixed_nodes, load_nodes, force_xyz)
title('ISCSO 2019 winner inverted design')
%% TEST of FEA parallel
% tic
% for i=1:500
%     [weight, compliance, stress, strain, U, x0_new] = run_fea(coord, connectivity, fixed_nodes, load_nodes, force_xyz, density, elastic_modulus);
% end
% toc
% 
% coord1 = {};
% conn1 = {};
% for ii = 1:500
%     coord1{end+1} = coord;
%     conn1{end+1} = connectivity;
% end
% tic
% [weight1, compliance1, stress1, strain1, U1, x0_new1] = run_fea_parallel(coord1, conn1, fixed_nodes, load_nodes, force_xyz, density, elastic_modulus);
% toc