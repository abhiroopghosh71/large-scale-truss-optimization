clear
close all

output_path = '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/TrussResults/20200324_nsga2_truss/';
experiment_path = strcat(output_path, 'truss_symmetric_nsga2_seed184716924_20200324-023906/');
F = dlmread(strcat(experiment_path, 'f_max_gen'));
X = dlmread(strcat(experiment_path, 'x_max_gen'));

coeff_var_X = std(X) ./ mean(X);

load('/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization/truss/sample_input/workspace_iscso.mat', 'coord', 'connectivity', 'fixed_nodes', 'load_nodes', 'force_xyz', 'density', 'elastic_modulus')
addpath '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/iscso_based_truss_optimization/large_scale_truss_optimization'

% scatter(F(:, 1), F(:, 2))
% xlabel('Weight (kg)')
% ylabel('Compliance (m/N)')
% title('Final population')

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
r1 = xreal1(1:187) ; % Radius of each element
z1 = xreal1(188:end);  % Z-coordinate of bottom members

%Horizontal elements in x-axis
connect_1(1:18, 3) = r1(1:18);
connect_1(37:54, 3) = connect_1(1:18, 2);
connect_1(19:36, 3) = r1(19:36);
connect_1(55:72, 3) = connect_1(19:36, 2);

% Cross elements at the end
connect_1(73:76, 3) = r1(37:40);

% For elements in y and z axis
connect_1(77:114, 3) = r1(41:78);
for e = 77:114
    % Repeat connectivity in y-axis elements
    if rem(e, 2) == 0
        connect_1(115 + (e-77), 3) = connect_1(e, 2);
    % Repeat connectivity in y-axis elements
    else
        connect_1(115 + (e-77), 3) = r1(round(79 + (e-77-1) / 2));
    end
end
% Bottom and top cross elements
connect_1(153:224, 3) = r1(98:169);

% y-axis sloped elements
connect_1(225:242, 3) = r1(170:187);
connect_1(243:260, 3) = connect_1(225:242, 3);

coord_1(1:10, 3) = z1;
coord_1(39:48, 3) = z1;
coord_1(11:19, 3) = flip(z1(1:end-1));
coord_1(49:57, 3) = flip(z1(1:end-1));
            
draw_truss(coord_1, connect_1, fixed_nodes, load_nodes, force_xyz)
title(sprintf('Weight (min.) = %f kg    Compliance = %f m/N\n', F(min_F1_indx, 1), F(min_F1_indx, 2)))

%% Min compliance solution
connect_2 = connectivity;
coord_2 = coord;

[min_F2, min_F2_indx] = min(F(:, 2));

xreal2 = X(min_F2_indx, :);
r2 = xreal2(1:187) ; % Radius of each element
z2 = xreal2(188:end);  % Z-coordinate of bottom members

%Horizontal elements in x-axis
connect_2(1:18, 3) = r2(1:18);
connect_2(37:54, 3) = connect_2(1:18, 2);
connect_2(19:36, 3) = r2(19:36);
connect_2(55:72, 3) = connect_2(19:36, 2);

% Cross elements at the end
connect_2(73:76, 3) = r2(37:40);

% For elements in y and z axis
connect_2(77:114, 3) = r2(41:78);
for e = 77:114
    % Repeat connectivity in y-axis elements
    if rem(e, 2) == 0
        connect_2(115 + (e-77), 3) = connect_2(e, 2);
    % Repeat connectivity in y-axis elements
    else
        connect_2(115 + (e-77), 3) = r2(round(79 + (e-77-1) / 2));
    end
end
% Bottom and top cross elements
connect_2(153:224, 3) = r2(98:169);

% y-axis sloped elements
connect_2(225:242, 3) = r2(170:187);
connect_2(243:260, 3) = connect_2(225:242, 3);

coord_2(1:10, 3) = z2;
coord_2(39:48, 3) = z2;
coord_2(11:19, 3) = flip(z2(1:end-1));
coord_2(49:57, 3) = flip(z2(1:end-1));
            
draw_truss(coord_2, connect_2, fixed_nodes, load_nodes, force_xyz)
title(sprintf('Weight = %f kg    Compliance (min.) = %f m/N\n', F(min_F2_indx, 1), F(min_F2_indx, 2)))


