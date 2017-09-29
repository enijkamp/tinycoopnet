function z = langevin_dynamics_generator(opts, net, z, syn_mat)
% the input syn_mat should be a 4-D matrix

for t = 1:opts.T
    res = vl_gan(net, z, syn_mat, [], 'conserveMemory', 1, 'cudnn', 1);
    delta_log = res(1).dzdx / opts.refsig2 / opts.refsig2 - z;
    z = z + opts.Delta2 * opts.Delta2 / 2 * delta_log;
    clear res;  
end
end