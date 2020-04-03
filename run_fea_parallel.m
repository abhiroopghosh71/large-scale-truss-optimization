function [weight_pop, compliance_pop, stress_pop, strain_pop, U_pop, x0_new_pop] = run_fea_parallel(Coordinates, Connectivity, fixednodes, loadn, force, density, elastic_modulus)

    pop_size = size(Coordinates, 2);
    
    weight_pop = zeros([pop_size, 1]);
    compliance_pop = zeros([pop_size, 1]);
    stress_pop = cell([pop_size, 1]);
    strain_pop = cell([pop_size, 1]);
    U_pop = cell([pop_size, 1]);
    x0_new_pop = cell([pop_size, 1]);
    parfor i = 1:pop_size
        [weight, compliance, stress, strain, U, x0_new] = ...
            run_fea(Coordinates{i}, Connectivity{i}, fixednodes, loadn, ...
            force, density, elastic_modulus);
        weight_pop(i) = weight;
        compliance_pop(i) = compliance;
        stress_pop{i} = stress;
        strain_pop{i} = strain;
        U_pop{i} = U;
        x0_new_pop{i} = x0_new;
    end
end