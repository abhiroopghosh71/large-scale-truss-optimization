clear
close all

output_path = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/TrussResults/20200322_nsga2_truss/';
experiment_path = strcat(output_path, 'truss_nsga2_seed184716924_20200322-015313/');
F = dlmread(strcat(experiment_path, 'f_max_gen'));
X = dlmread(strcat(experiment_path, 'x_max_gen'));

coeff_var_X = std(X) ./ mean(X);

load('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization/truss/sample_input/workspace_iscso.mat', 'coord', 'connectivity', 'fixed_nodes', 'load_nodes', 'force_xyz', 'density', 'elastic_modulus')
addpath '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization'

scatter(F(:, 1), F(:, 2))
xlabel('Weight (kg)')
ylabel('Compliance (m/N)')
title('Final population')
% for ii = 5:99:500
% %     if rem(ii, 100) ~= 0
% %        continue 
% %     end
%     xreal = X(ii, :);
%     
%     connect_1 = connectivity;
%     coord_1 = coord;
%     
%     connect_1(:, 3) = xreal(1:260);
%     coord_1(1:10, 3) = xreal(261:270);
%     coord_1(39:48, 3) = xreal(261:270);
%     coord_1(11:19, 3) = flip(xreal(262:270));
%     coord_1(49:57, 3) = flip(xreal(262:270));
%     
%     draw_truss(coord_1, connect_1, fixed_nodes, load_nodes, force_xyz)
% %     break
%     
% %     [weight, compliance] = run_fea(coord_1, connect_1, fixed_nodes, load_nodes, force_xyz, density, elastic_modulus);
% end

%% Min weight solution
connect_1 = connectivity;
coord_1 = coord;

[min_F1, min_F1_indx] = min(F(:, 1));

xreal1 = X(min_F1_indx, :);
connect_1(:, 3) = xreal1(1:260);
coord_1(1:10, 3) = xreal1(261:270);
coord_1(39:48, 3) = xreal1(261:270);
coord_1(11:19, 3) = flip(xreal1(261:269));
coord_1(49:57, 3) = flip(xreal1(261:269));
draw_truss(coord_1, connect_1, fixed_nodes, load_nodes, force_xyz)
title(sprintf('Weight (min.) = %f kg    Compliance = %f m/N\n', F(min_F1_indx, 1), F(min_F1_indx, 2)))

%% Min compliance solution
connect_2 = connectivity;
coord_2 = coord;

[min_F2, min_F2_indx] = min(F(:, 2));

xreal2 = X(min_F2_indx, :);
connect_2(:, 3) = xreal2(1:260);
coord_2(1:10, 3) = xreal2(261:270);
coord_2(39:48, 3) = xreal2(261:270);
coord_2(11:19, 3) = flip(xreal2(261:269));
coord_2(49:57, 3) = flip(xreal2(261:269));
draw_truss(coord_2, connect_2, fixed_nodes, load_nodes, force_xyz)
title(sprintf('Weight = %f kg    Compliance (min.) = %f m/N\n', F(min_F2_indx, 1), F(min_F2_indx, 2)))