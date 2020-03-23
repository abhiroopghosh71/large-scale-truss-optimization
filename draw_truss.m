% function draw_truss(coordinates_file, connectivity_file, fixednodes_file, loadn_file, force_file)
%     % Input files
%     Coordinates = load(coordinates_file);
%     Connectivity = load(connectivity_file);
%     fixednodes = load(fixednodes_file);
%     loadn = load(loadn_file);
%     force = load(force_file);

function draw_truss(Coordinates, Connectivity, fixednodes, loadn, force)
    % Just to test if 3d is working
    % Coordinates(:, 3) = rand([size(Coordinates, 1),1]) * 5;
    % Just to test if different widths for different radii is working
    % Connectivity(:, 3) = rand([size(Connectivity, 1),1]);
    figure
    hold on
    xlabel('x')
    ylabel('y')
    zlabel('z')

    % Legend=cell(3,1); %  three types of components 
    normal_radius = Connectivity(:, 3);
    if min(Connectivity(:, 3)) ~= max(Connectivity(:, 3))
        normal_radius = (Connectivity(:, 3) - min(Connectivity(:, 3))) / (max(Connectivity(:, 3)) - min(Connectivity(:, 3)));
    end
    % Plot the members
    for i = 1:size(Connectivity, 1)
        nodes = Connectivity(i, 1:2);
        member_radius = Connectivity(i, 3);
        p1 = Coordinates(nodes(1), :);
        p2 = Coordinates(nodes(2), :);
        h1 = plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], 'k', 'LineWidth', 0.5 + 2*normal_radius(i));
    end
    
    % Enhance the nodes
    for i = 1:size(Coordinates, 1)
        h1 = scatter3(Coordinates(i, 1), Coordinates(i, 2), Coordinates(i, 3), 'filled', 'r', 'LineWidth', 0.5 + normal_radius(i));
    end
    % Legend{1} = 'Members';

    % Plot the fixed nodes
    for i = 1:size(fixednodes, 1)
        node = fixednodes(i);
        p = Coordinates(node, :);
        h2 = scatter3(p(1), p(2), p(3), 100, 'bs', 'LineWidth', 1);
    end
    % Legend{2} = 'Fixed Nodes';

    % Plot the load nodes
    for i = 1:size(loadn, 1)
        node = loadn(i);
        p = Coordinates(node, :);
        h3 = scatter3(p(1), p(2), p(3), 100, 'rs', 'LineWidth', 1);
    end
    % Legend{3} = 'Load Nodes';

    h = [h1 h2 h3];
    legend(h, 'Members', 'Fixed Nodes', 'Load Nodes');
    xlim([-5, 80])
    ylim([-5, 10])
    zlim([-30, 20])
    view(0, 0)
%     view(45, 45)
end