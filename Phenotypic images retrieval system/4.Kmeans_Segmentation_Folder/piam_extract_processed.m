tic
fileList = getAllFiles('/home/wubizhi/Downloads/TAIR_Net&Search_Net/piam_img_256');
for inx=1:length(fileList)
    extra_green(fileList{inx},'piam_img_extract_256');
end
toc