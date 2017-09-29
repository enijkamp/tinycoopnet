function net = add_conv_block(net, opts, id, h, w, in, out, stride, pad)

net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', 'conv', id), ...
    'weights', {{to_gpu_array(init_weight(opts, h, w, in, out, 'single'), opts.use_gpu), to_gpu_array(zeros(1, out, 'single'), opts.use_gpu)}}, ...
    'stride', [stride, stride], ...
    'pad', pad, ...
    'learningRate', [1, 2], ...
    'weightDecay', [opts.weightDecay 0]) ;

net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id));

end

function weights = init_weight(opts, h, w, in, out, type)
sc = 0.01/opts.scale;
weights = randn(h, w, in, out, type)*sc;
end