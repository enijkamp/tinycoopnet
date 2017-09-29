function [net1, net2] = learn_dual_net(opts, imdb, get_batch, net1, net2)

%% setup

% generator
opts.z_sz = [1, 1, size(net2.layers{1}.weights{1}, 4)];
net2.z_sz = opts.z_sz;
opts.dydz_sz2 = [opts.z_sz(1:2), 1];
for l = 1:numel(net2.layers)
    if strcmp(net2.layers{l}.type, 'convt')
        f_sz = size(net2.layers{l}.weights{1});
        crops = [net2.layers{l}.crop(1)+net2.layers{l}.crop(2), ...
            net2.layers{l}.crop(3)+net2.layers{l}.crop(4)];
        opts.dydz_sz2(1:2) = net2.layers{l}.upsample.*(opts.dydz_sz2(1:2) - 1) ...
            + f_sz(1:2) - crops;
    end
end
net2.dydz_sz = opts.dydz_sz2;

z = randn(opts.z_sz, 'single');

if opts.use_gpu
    net2 = vl_simplenn_move(net2, 'gpu');
    res = vl_gan(net2, gpuArray(z));
    net2 = vl_simplenn_move(net2, 'cpu');
else
    res = vl_gan(net2, z);
end

net2.numFilters = zeros(1, length(net2.layers));
for l = 1:length(net2.layers)
    if isfield(net2.layers{l}, 'weights')
        sz = size(res(l+1).x);
        net2.numFilters(l) = sz(1) * sz(2);
    end
end

opts.layer_sets2 = numel(net2.layers):-1:1;

net2.normalization.imageSize = opts.dydz_sz2;
net2.normalization.averageImage = zeros(opts.dydz_sz2, 'single');
opts.sx = opts.dydz_sz2(1);
opts.sy = opts.dydz_sz2(2);
clear z;

% descriptor

net1.normalization.imageSize = [opts.sx, opts.sy, 3];
net1.normalization.averageImage = net2.normalization.averageImage;
img = randn(net1.normalization.imageSize, 'single');
if opts.use_gpu
    net1 = vl_simplenn_move(net1, 'gpu') ;
    res = vl_simplenn(net1, gpuArray(img));
    net1 = vl_simplenn_move(net1, 'cpu');
else
    res = vl_simplenn(net1, img);
end
opts.dydz_sz1 = size(res(end).x);

net1.numFilters = zeros(1, length(net1.layers));
for l = 1:length(net1.layers)
    if isfield(net1.layers{l}, 'weights')
        sz = size(res(l+1).x);
        net1.numFilters(l) = sz(1) * sz(2);
    end
end
opts.layer_sets1 = numel(net1.layers):-1:1;

clear res;
clear img;

% subset

num_images = size(imdb,4);
subset = 1:num_images;

% momemtum

net1 = initialize_momentum(net1);
net2 = initialize_momentum(net2);

% GPUs
if opts.use_gpu
    gpuDevice(1)
end

%% train and validate

learningTime = tic;

delete([opts.gen_im_folder,'*.png']);
delete([opts.syn_im_folder,'*.png']);

save([opts.trained_folder,'opts.mat'],'opts');

loss = zeros(opts.numEpochs, 1);
for epoch=1:opts.numEpochs

    % train
    %% TODO check correctness versus original code, check estimates of gradients (en)
    [net1, net2, syn_mats, z_mats] = process_epoch_dual(opts, get_batch, epoch, subset, imdb, net1, net2);
    
    % images
    %% TODO write synthesis for epochs, check image normalization (en)
    %for i = 1:opts.num_syn
    %    imwrite((gen_mats(:,:,:,i)+opts.mean_im)/256,[opts.gen_im_folder,'gen_im',num2str(i),'.png']);
    %    imwrite((syn_mats(:,:,:,k)+opts.mean_im)/256,[opts.syn_im_folder,'syn_im',num2str(k),'.png']);
    %end
    
    % nets
    save([opts.trained_folder,'des_net.mat'],'net1');
    save([opts.trained_folder,'gen_net.mat'],'net2');

    % loss
    loss(epoch) = compute_loss(opts, syn_mats, net2, z_mats);
    save([opts.trained_folder,'loss.mat'],'loss');
    disp(['Loss: ', num2str(loss(epoch))]);
    %% TODO plot (en)
end

% time
learningTime = toc(learningTime);
hrs = floor(learningTime / 3600);
learningTime = mod(learningTime, 3600);
mins = floor(learningTime / 60);
secds = mod(learningTime, 60);
fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);

end

function loss = compute_loss(opts, syn_mats, net2, z_mats)
if opts.use_gpu
    net2 = vl_simplenn_move(net2, 'gpu');
end
loss = 0;
res = [];
for i=1:numel(syn_mats)
    syn_mat = syn_mats{i};
    z = z_mats{i};

    res = vl_gan(net2, to_gpu_array(z, opts.use_gpu), to_gpu_array(syn_mat, opts.use_gpu), res, ...
        'accumulate', false, ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn) ;
    loss = loss + gather( mean(reshape(sqrt((res(end).x - syn_mat).^2), [], 1)));
end
end

function net = initialize_momentum(net)
for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
        J = numel(net.layers{i}.weights) ;
        for j=1:J
            net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
        end
        if ~isfield(net.layers{i}, 'learningRate')
            net.layers{i}.learningRate = ones(1, J, 'single') ;
        end
        if ~isfield(net.layers{i}, 'weightDecay')
            net.layers{i}.weightDecay = ones(1, J, 'single') ;
        end
    end
end
end