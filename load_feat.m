clc
clear
path2actions = 'E:\C3D-tensorflow-master\C3D_feat_fc2_NonOverlap_16frames';
Actions_Dir = dir(path2actions);
DataPerVideo = struct;
idx1 = 1; 
DataLength = 0;
tic
for i = 3:length(Actions_Dir)
    path2features = [path2actions,'\',Actions_Dir(i).name];
    feat_Dir = dir(path2features);
    for j = 3:length(feat_Dir)
        load([path2features,'\',feat_Dir(j).name]);
        NumFeat = size(features,1);
        DataPerVideo(idx1).Features = features;
        DataPerVideo(idx1).Labels = strrep(feat_Dir(j).name,'.mat','');
        DataPerVideo(idx1).NumFeat = NumFeat;
        idx1 = idx1 + 1;
        DataLength = DataLength + NumFeat;
    end
end
toc

features = zeros(DataLength,size(DataPerVideo(1).Features,2));
labels = cell(DataLength,1);
idx2 = 1;
tic
for i = 1: length(DataPerVideo)
    NumFeat = DataPerVideo(i).NumFeat;
    features(idx2:idx2+NumFeat-1,:) = DataPerVideo(i).Features;
    for j = idx2:idx2+NumFeat-1 
        labels{j,1} = DataPerVideo(i).Labels;
    end
    idx2 = idx2+NumFeat;
end
toc
save([pwd,'\C3D_feat_fc2_NonOverlap_16frames\DataPerVideo.mat'],'DataPerVideo')
save([pwd,'\C3D_feat_fc2_NonOverlap_16frames\features.mat'],'features')
save([pwd,'\C3D_feat_fc2_NonOverlap_16frames\labels.mat'],'labels')
