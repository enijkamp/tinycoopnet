function net = add_convt_block(net, opts, id, h, w, in, out, stride, pad, learning_rate)

if nargin < 11
    learning_rate = 1;
end

net.layers{end+1} = struct('type', 'convt', 'name', sprintf('%s%s', 'convt', id), ...
    'weights', {{to_gpu_array(init_weight(opts, h, w, out, in, 'single'), opts.use_gpu), to_gpu_array(zeros(1, 1, out, 'single'), opts.use_gpu)}}, ...
    'upsample', [stride, stride], ...
    'crop', pad, ...
    'numGroups', 1, ...
    'learningRate', learning_rate*[1, 2], ...
    'weightDecay', [opts.weightDecay 0]) ;
        
end

function weights = init_weight(opts, h, w, in, out, type)
sc = 0.01/opts.scale;
weights = randn(h, w, in, out, type)*sc;
end