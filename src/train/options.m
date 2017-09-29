function [opts] = options(root)

% gpu
opts.use_gpu = true;

% matconvnet options
opts.conserveMemory = true;
opts.backPropDepth = +inf;
opts.sync = false;
opts.cudnn = true;
opts.weightDecay = 0.0001;
opts.momentum = 0.5;
opts.numSubBatches = 1;

% num epochs
opts.numEpochs = 200;
opts.batchSize = 32;

% sampling parameters
opts.num_syn = 32;

% descriptor / net1 parameters
opts.Delta1 = 0.3;
opts.Gamma1 = 0.00005 * logspace(-2, -3, opts.numEpochs)*100; % learning rate
opts.refsig1 = 1; % standard deviation for reference model q(I/sigma^2).
opts.T = 15;
opts.cap1 = 20;

% generator / net2 parameters
opts.Delta2 = 0.3;
opts.Gamma2 =  0.00005 * logspace(-2, -3, opts.numEpochs)*100; % learning rate
opts.refsig2 = 1;
opts.s = 0.3;
opts.cap2 = 8;
opts.infer_z = true;

% image size
opts.im_size = 32;

% image path: where the dataset locates
opts.inPath = [root 'data/'];

% name folders for results
opts.syn_im_folder = [root 'output/ims_syn/'];
opts.gen_im_folder = [root 'output/ims_gen/'];
opts.trained_folder = [root 'output/nets/'];

end