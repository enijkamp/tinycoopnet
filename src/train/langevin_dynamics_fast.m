function syn_mat = langevin_dynamics_fast(opts, net, syn_mat)
% the input syn_mat should be a 4-D matrix
numImages = size(syn_mat, 4);

dydz = gpuArray(ones(opts.dydz_sz1, 'single'));
dydz = repmat(dydz, 1, 1, 1, numImages);

for t = 1:opts.T
%     fprintf('Langevin dynamics sampling iteration %d\n', t);
    % forward-backward to compute df/dI
%     N_gaussian = gpuArray(randn(size(syn_mat), 'single'));
    res = vl_simplenn(net, syn_mat, dydz, [], 'conserveMemory', 1, 'cudnn', 1);
    
    % part1: derivative on f(I; w)  part2: gaussian I
    syn_mat = syn_mat + opts.Delta1 * opts.Delta1 / 2 * res(1).dzdx ...  
        - opts.Delta1 * opts.Delta1 / 2 / opts.refsig1 / opts.refsig1 * syn_mat;
    
    % part3: white noise N(0, 1)     
    syn_mat = syn_mat + opts.Delta1 * gpuArray(randn(size(syn_mat), 'single'));
    clear res;
end

end