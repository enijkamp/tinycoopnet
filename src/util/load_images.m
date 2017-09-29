function [imdb, fn] = load_images(img_path, img_size)
files = dir([img_path, '*.png']);
imdb = single(zeros(img_size, img_size, 3, length(files)));
for i = 1:length(files)
    img = single(imread([img_path,files(i).name]));
    min_val = min(img(:));
    max_val = max(img(:));
    imdb(:,:,:,i) = (img - min_val) / (max_val - min_val)*2 - 1;
end
fn = @(imdb, batch) get_batch(imdb,batch);
end

function [im] = get_batch(imdb, batch)
im = imdb(:,:,:,batch);
end