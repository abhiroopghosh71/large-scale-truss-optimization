clear
close all

addpath('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization')
output_path = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/TrussResults/20200326_truss_nsga2_unsupported node/';
% experiment_path = strcat(output_path, 'truss_nsga2_seed184716924_20200324-023902/');
experiment_path = strcat(output_path, 'truss_nsga2_seed184716924_20200326-001556/');
F = dlmread(strcat(experiment_path, 'f_max_gen'));
X = dlmread(strcat(experiment_path, 'x_max_gen'));

%%  3 constraints repair
% X = dlmread('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization/output/truss_nsga2_seed184716924_20200413-233858/x_current_gen');
% F = dlmread('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization/output/truss_nsga2_seed184716924_20200413-233858/f_current_gen');

X = dlmread('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization/output/truss_nsga2_seed184716924_20200413-233948/x_current_gen');
F = dlmread('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization/output/truss_nsga2_seed184716924_20200413-233948/f_current_gen');


coeff_var_X = std(X) ./ mean(X);

% load('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization/truss/sample_input/workspace_iscso.mat', 'coord', 'connectivity', 'fixed_nodes', 'load_nodes', 'force_xyz', 'density', 'elastic_modulus')
% addpath '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization/'
input_path = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization/';
coordinates_file = strcat(input_path, 'truss/sample_input/coord_iscso.csv');
connectivity_file = strcat(input_path, 'truss/sample_input/connect_iscso.csv');
fixednodes_file = strcat(input_path, 'truss/sample_input/fixn_iscso.csv');
loadn_file = strcat(input_path, 'truss/sample_input/loadn_iscso.csv');
force_file = strcat(input_path, 'truss/sample_input/force_iscso.csv');

coord = dlmread(coordinates_file);
connectivity = dlmread(connectivity_file);
fixed_nodes = dlmread(fixednodes_file);
load_nodes = dlmread(loadn_file);
force_xyz = dlmread(force_file);
density = 7121.4;  % kg/m3
elastic_modulus = 200e9;  % Pa

% scatter(F(:, 1), F(:, 2))
% xlabel('Weight (kg)')
% ylabel('Compliance (m/N)')
% title('Final population')
% for ii = 50:150:500
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
%     coord_1(11:19, 3) = flip(xreal(261:269));
%     coord_1(49:57, 3) = flip(xreal(261:269));
%     
%     draw_truss(coord_1, connect_1, fixed_nodes, load_nodes, force_xyz)
%     title(sprintf('Weight = %f kg    Compliance = %f m/N\n', F(ii, 1), F(ii, 2)))
%     zlim([-5, 10])
% %     break
%     
% %     [weight, compliance] = run_fea(coord_1, connect_1, fixed_nodes, load_nodes, force_xyz, density, elastic_modulus);
% end

%% KLUGE
% force_xyz(:, 3) = -force_xyz(:, 3);

%% Min weight solution
connect_1 = connectivity;
coord_1 = coord;

[min_F1, min_F1_indx] = min(F(:, 1));

xreal1 = X(min_F1_indx, :);
% REMOVE THIS!!
% fixed_nodes = [fixed_nodes; 20; 38; 58; 76];
% for n = 1:length(fixed_nodes)
%     load_nodes(load_nodes == fixed_nodes(n)) = [];
% end
% xreal1(261:270) = 1e-3*[3500, 1623, -174,...
%     -1685,...
% -3019,...
% -4075,...
% -4896,...
% -5552,...
% -6080,...
% -6204];

connect_1(:, 3) = xreal1(1:260);
coord_1(1:10, 3) = xreal1(261:270);
coord_1(39:48, 3) = xreal1(261:270);
coord_1(11:19, 3) = flip(xreal1(261:269));
coord_1(49:57, 3) = flip(xreal1(261:269));
draw_truss(coord_1, connect_1, fixed_nodes, load_nodes, force_xyz)
title(sprintf('Weight (min.) = %f kg    Compliance = %f m/N\n', F(min_F1_indx, 1), F(min_F1_indx, 2)))
zlim([-30, 20])

[weight1, compliance1, stress1, strain1, U1, x0_new1] = run_fea(coord_1, connect_1, fixed_nodes, load_nodes, force_xyz, density, elastic_modulus);
% draw_truss(x0_new1, connect_1, fixed_nodes, load_nodes, force_xyz)
% title(sprintf('Deformed Truss Weight (min.) = %f kg    Compliance = %f m/N\n', F(min_F1_indx, 1), F(min_F1_indx, 2)))
% zlim([-30, 20])

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
zlim([-30, 20])

[weight2, compliance2, stress2, strain2, U2, x0_new2] = run_fea(coord_2, connect_2, fixed_nodes, load_nodes, force_xyz, density, elastic_modulus);
% draw_truss(x0_new2, connect_2, fixed_nodes, load_nodes, force_xyz)
% title(sprintf('Deformed Truss Weight = %f kg    Compliance (min.) = %f m/N\n', F(min_F2_indx, 1), F(min_F2_indx, 2)))
zlim([-30, 20])


%% shape+size repair 1 obj
connect_1_obj = connectivity;
coord_1_obj = coord;
x_1_obj = dlmread('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/TrussResults/truss_1obj_nsga2_seed184716924_20200412-190022/x_max_gen');
% x_1_obj(261:270) = flip(sort(x_1_obj(261:270))');
x_1_obj(261:270) = [3.2818    1.5603    0.5152    -0.5037    -1.5185    -2.0678    -3.0214    -4.0046   -5.0088   -5.3845];
connect_1_obj(:, 3) = x_1_obj(1:260);
connect_1_obj(107, 3) = 0.0152051474452058;
coord_1_obj(1:10, 3) = x_1_obj(261:270);
coord_1_obj(39:48, 3) = x_1_obj(261:270);
coord_1_obj(11:19, 3) = flip(x_1_obj(261:269));
coord_1_obj(49:57, 3) = flip(x_1_obj(261:269));

[weight_1_obj, compliance_1_obj, stress_1_obj, strain_1_obj, U_1_obj, x0_new_1_obj] = run_fea(coord_1_obj, connect_1_obj, fixed_nodes, load_nodes, force_xyz, density, elastic_modulus);

draw_truss(coord_1_obj, connect_1_obj, fixed_nodes, load_nodes, force_xyz)
title(sprintf('Weight = %0.2f kg    Compliance = %0.2f m/N\n', weight_1_obj, compliance_1_obj))
zlim([-30, 20])




%% Correlation analysis
% figure
% hold on
% corr_xf = [];
% for ii = 1:260
%    corr_xf = [corr_xf; corr(X(:, ii), F)];
% end
% plot(1:260, corr_xf(:, 1), 'r')
% plot(1:260, corr_xf(:, 2), 'b')
% legend('Weight Correlation', 'Compliance Correlation')
% xlabel('Variable')
% ylabel('Correlation')
% yline(0.8, 'LineWidth', 2)
% yline(-0.8, 'LineWidth', 2)
% % scatter(1:260, corr_xf(:, 1), 'ro', 'filled')
% % scatter(1:260, -corr_xf(:, 2), 'bo', 'filled')
% title('Size variables')


% figure
% hold on
% corr_xf1 = [];
% for ii = 261:270
%    corr_xf1 = [corr_xf1; corr(X(:, ii), F)];
% end
% plot(1:10, corr_xf1(:, 1), 'r')
% plot(1:10, corr_xf1(:, 2), 'b')
% legend('Weight Correlation', 'Compliance Correlation')
% xlabel('Variable')
% ylabel('Correlation')
% yline(0.8, 'LineWidth', 2)
% yline(-0.8, 'LineWidth', 2)
% title('Shape variables')