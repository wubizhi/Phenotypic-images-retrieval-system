function extra_green(img_file,new_folder)
he = imread(img_file);
tmp_path = split(img_file,'/');
tmp_path(end-1) = new_folder;
tmp='';
for inx = 1:length(tmp_path)-1
    tmp = tmp+tmp_path(inx)+'/';
end
tmp = tmp+tmp_path(end);

cform = makecform('srgb2lab');
lab_he = applycform(he,cform);
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

nColors = 3;
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);

pixel_labels = reshape(cluster_idx,nrows,ncols);
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = he;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

det1 = evaluateLevelOfGreen(segmented_images{1});
det2 = evaluateLevelOfGreen(segmented_images{2});
det3 = evaluateLevelOfGreen(segmented_images{3});
det_max = max([det1,det2,det3]);

if(det1==det_max)
    save_img = segmented_images{1};
elseif(det2==det_max)
    save_img = segmented_images{2};    
else
    save_img = segmented_images{3};
end
imwrite(save_img,char(tmp))

mean_cluster_value = mean(cluster_center,2);
[tmp, idx] = sort(mean_cluster_value);
blue_cluster_num = idx(1);
L = lab_he(:,:,1);
blue_idx = find(pixel_labels == blue_cluster_num);
L_blue = L(blue_idx);
is_light_blue = imbinarize(L_blue);
nuclei_labels = repmat(uint8(0),[nrows ncols]);
nuclei_labels(blue_idx(is_light_blue==false)) = 1;
nuclei_labels = repmat(nuclei_labels,[1 1 3]);
blue_nuclei = he;
blue_nuclei(nuclei_labels ~= 1) = 0;
end
