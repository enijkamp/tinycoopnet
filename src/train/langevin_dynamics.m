function syn_mat = langevin_dynamics(opts, net, syn_mat)
% the input syn_mat should be a 3-D matrix

%% TODO zeros or ones? see fast (en)
dydz = zeros(opts.dydz_sz1, 'single');

for t = 1:opts.T    
    % Leapfrog half-step
    res = vl_simplenn(net, syn_mat, dydz, [], 'conserveMemory', true, 'cudnn', false);

    % part1: derivative on f(I; w)  part2: gaussian I
    syn_mat = syn_mat + opts.Delta1 * opts.Delta1 / 2 * res(1).dzdx ...  
        - opts.Delta1 * opts.Delta1 / 2 / opts.refsig1 / opts.refsig1 * syn_mat;
    
    % part3: white noise N(0, 1)
    syn_mat = syn_mat + opts.Delta1 * randn(size(syn_mat), 'single');
    
    clear res;
end

end