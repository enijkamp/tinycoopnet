function [opts] = setup_dirs(opts, prefix)
opts.trained_folder = [opts.trained_folder prefix];
opts.gen_im_folder = [opts.gen_im_folder prefix];
opts.syn_im_folder = [opts.syn_im_folder prefix];
if ~exist(opts.trained_folder,'dir') mkdir(opts.trained_folder); end
if ~exist(opts.gen_im_folder,'dir') mkdir(opts.gen_im_folder); end
if ~exist(opts.syn_im_folder,'dir') mkdir(opts.syn_im_folder); end
end