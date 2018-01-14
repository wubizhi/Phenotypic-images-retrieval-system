function LAB_Seg(img_file,nColors,save_file_name)
    he = imread(img_file);
    cform = makecform('srgb2lab');
    lab_he = applycform(he,cform);
    ab = double(lab_he(:,:,2:3));
    nrows = size(ab,1);
    ncols = size(ab,2);
    ab = reshape(ab,nrows*ncols,2);
    % repeat the clustering 3 times to avoid local minima
    [cluster_idx, ~] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                          'Replicates',3);
    pixel_labels = reshape(cluster_idx,nrows,ncols);
    segmented_images = cell(1,nColors);
    rgb_label = repmat(pixel_labels,[1 1 3]);

    det_max = 0;
    for k = 1:nColors
        color = he;
        color(rgb_label ~= k) = 0;
        segmented_images{k} = color;
        det = evaluateLevelOfGreen(segmented_images{k});
        if(det>det_max)
            det_max = det;
            save_img = segmented_images{k};
        end
    end
    imwrite(save_img,char(save_file_name));
end