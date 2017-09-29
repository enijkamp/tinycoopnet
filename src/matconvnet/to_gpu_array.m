function [array] = to_gpu_array(array, use_gpu)
if use_gpu
    array = gpuArray(array);
end
end