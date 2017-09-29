function setup_data(path, save_path, factor, patch_size, num_patch)
mkdir(save_path);
im = imread(path);
im = imresize(im,factor);
for i = 1:num_patch
    start_pix = randi(min(size(im,1),size(im,2))-patch_size+1);
    patch = im(start_pix:(start_pix+patch_size-1),start_pix:(start_pix+patch_size-1),:);
    imwrite(patch,[save_path,num2str(i),'.png']);
end
end