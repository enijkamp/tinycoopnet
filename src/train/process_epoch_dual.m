function [net1, net2, syn_mats, z_mats] = process_epoch_dual(opts, getBatch, epoch, subset, imdb, net1, net2)

if opts.use_gpu
    net1 = vl_simplenn_move(net1, 'gpu');
    net2 = vl_simplenn_move(net2, 'gpu');
end

dydz_syn = to_gpu_array(ones(opts.dydz_sz1, 'single'), opts.use_gpu);
dydz_syn = repmat(dydz_syn, 1, 1, 1, opts.num_syn);
res1 = [];
res_syn = [];
res2 = [];
training = true;

num_cell = ceil(numel(subset) / opts.batchSize);
syn_mats = cell(1, num_cell);
z_mats = cell(1, num_cell);
zs = cell(1, num_cell);

for t=1:opts.batchSize:numel(subset)
    fprintf('Training: epoch %02d: batch %3d/%3d: \n', epoch, ...
        fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    batchTime = tic ;
    numDone = 0 ;
    
    for s=1:opts.numSubBatches
        % get this image batch
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        im = getBatch(imdb, batch);
        im = to_gpu_array(im, opts.use_gpu);
        cell_idx = (ceil(t / opts.batchSize) - 1) * numlabs + labindex;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Step 1: Generate synthesis from z
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % G0: generate Xi
        z_mats{cell_idx} = randn([opts.z_sz, opts.num_syn], 'single');
        z = to_gpu_array(z_mats{cell_idx}, opts.use_gpu);
        % D1: generate Yi
        syn_mat = vl_gan(net2, z, [], [],...
            'accumulate', s ~= 1, ...
            'disableDropout', ~training, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
        syn_mat = syn_mat(end).x;        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Step 2: (1) Update generator mats by descriptor net, (2) Update z
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % synthesize image according to current weights 
        % D1: generate y wave i
        
        %% TODO compare nummerically, fix langevin_dynamics_fast to be cpu compatible (en)
        if opts.use_gpu
            syn_mat = langevin_dynamics_fast(opts, net1, syn_mat);
        else
            for syn_ind = 1:opts.num_syn
                syn_mat(:,:,:,syn_ind) = langevin_dynamics(opts, net1, syn_mat(:,:,:,syn_ind));
            end
        end
        syn_mats{cell_idx} = gather(syn_mat);
        
        % G1: Y wave - syn_mat
        % Xj - z
        % run langevin IG steps to update z
        z = langevin_dynamics_generator(opts, net2, z, syn_mat);
        zs{cell_idx} = gather(z);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Step 3: Learn net1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        numImages = size(im, 4);
        dydz1 = to_gpu_array(ones(opts.dydz_sz1, 'single'), opts.use_gpu);
        dydz1 = repmat(dydz1, 1, 1, 1, numImages);
        
        res1 = vl_simplenn(net1, im, dydz1, res1, ...
            'accumulate', s ~= 1, ...
            'disableDropout', ~training, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn);
        
        res_syn = vl_simplenn(net1, syn_mat, dydz_syn, res_syn, ...
            'accumulate', s ~= 1, ...
            'disableDropout', ~training, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Step 4: Learn net2
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        res2 = vl_gan(net2, z, syn_mat, res2, ...
            'accumulate', s ~= 1, ...
            'disableDropout', ~training, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
        
        numDone = numDone + numel(batch) ;
    end
    
    net1 = accumulate_gradients1(opts, opts.Gamma1(t), batchSize, net1, res1, res_syn);
    net2 = accumulate_gradients2(opts, opts.Gamma2(t), batchSize, net2, res2);
   
    fprintf('max inferred z is %.2f, min inferred z is %.2f, and std is %.2f\n', max(z(:)), min(z(:)), std(z(:)))
    
    % time
    batchTime = toc(batchTime);
    speed = 1/batchTime;
    fprintf('Time: %.2f s (%.1f data/s)\n', batchTime, speed);
end

if opts.use_gpu
    net1 = vl_simplenn_move(net1, 'cpu') ;
    net2 = vl_simplenn_move(net2, 'cpu') ;
end
end


function [net] = accumulate_gradients2(opts, lr, batchSize, net, res)
layer_sets = opts.layer_sets2;

for l = layer_sets
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;
        
        if isfield(net.layers{l}, 'weights')
            % gradient descent
            gradient_dzdw = (1 / batchSize) * (1 / opts.s / opts.s) * res(l).dzdw{j};
            
            max_val = max(abs(gradient_dzdw(:)));
            
            if max_val > opts.cap2;
                gradient_dzdw = gradient_dzdw / max_val * opts.cap2;
            end
  
            net.layers{l}.momentum{j} = ...
                + opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                + gradient_dzdw;
            
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j};
            
            if j == 1
                res_l = min(l+2, length(res));
                fprintf('Net2: layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), thisLR);
            end
        end
    end
end
end

function [net] = accumulate_gradients1(opts, lr, batchSize, net, res, res_syn)
layer_sets = opts.layer_sets1;

for l = layer_sets
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;
        
        if isfield(net.layers{l}, 'weights')
            gradient_dzdw = ((1 / batchSize) * res(l).dzdw{j} -  ...
                (1 / opts.num_syn) * res_syn(l).dzdw{j}) / net.numFilters(l);
            if max(abs(gradient_dzdw(:))) > opts.cap1
                gradient_dzdw = gradient_dzdw / max(abs(gradient_dzdw(:))) * opts.cap1;
            end
            
            net.layers{l}.momentum{j} = ...
                + opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                + gradient_dzdw;
            
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j};
            
            if j == 1
                res_l = min(l+1, length(res));
                fprintf('Net1: layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), thisLR);
            end
        end
    end
end
end