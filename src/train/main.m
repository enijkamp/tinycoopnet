function [] = main()
% simple example of training on CPU

% config
img_name = 'ivy';
%% TODO fix scaling (en)
img_size = 1; 
patch_size = 32;
num_patches = 200;
sample_patches = true;
use_gpu = false;
compile_convnet = true;

% setup
rng(123);
old_pwd = cd(fileparts(which(mfilename)));
root = setup_path();
setup_convnet(use_gpu, compile_convnet);
%% TODO factor out (en)
if sample_patches
    setup_data([root 'data/ivy/original/ivy.jpg'], [root 'data/ivy/' num2str(img_size) '/'], img_size, patch_size, num_patches);
end

% config
[opts] = options(root);
opts.use_gpu = use_gpu;
opts.numEpochs = 1;
opts.num_syn = 2;

% descriptor
net1 = create_descriptor_net(opts);

% generator
net2 = create_generator_net(opts);

% prep
prefix = ['main/' num2str(img_size) '/'];
[imdb, get_batch] = load_images([root 'data/' img_name '/' num2str(img_size) '/'], patch_size);
opts = setup_dirs(opts, prefix);

% run
learn_dual_net(opts, imdb, get_batch, net1, net2);

% restore
cd(old_pwd);

end

function root = setup_path()
root = '../../';
addpath([root 'src/train']);
addpath([root 'src/matconvnet']);
addpath([root 'src/util']);
end