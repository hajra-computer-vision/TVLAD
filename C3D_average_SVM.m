clc
clear
% Add folders to path.
path2lib = 'E:\MS_Research_work\Matlab_work';
addpath([path2lib '\ST-VLAD-master']);
addpath([path2lib '\vtshitoyan-plotConfMat-1787253']);
addpath(genpath([path2lib '\ClassifierToolbox-master']));

disp('Loading Data...')
% MuHAVi-uncut-RGB
DS0 = load('G:\Hajra Naeem\C3D\C3D\Muhavi_Features_Muhavi_Finetune\FeaturesFinetuneWithoutCam8\features.mat');
LBDS0 = load('G:\Hajra Naeem\C3D\C3D\Muhavi_Features_Muhavi_Finetune\FeaturesFinetuneWithoutCam8\labels.mat');
Data1 = DS0.features;
DataNames1 = LBDS0.labels;

load('G:\Hajra Naeem\C3D\C3D\Muhavi_Features_Muhavi_Finetune\FeaturesFinetuneWithoutCam8\DataPerVideo.mat');
PosFeat = zeros(1,length(Data1));
LabelsPerVideo = cell(1,length(DataPerVideo));
idx = 1;
for i = 1:length(DataPerVideo)
    nFeat = DataPerVideo(i).NumFeat;
    LabelsPerVideo{i} = (DataPerVideo(i).Labels);
    PosFeat(idx:idx+nFeat-1) = 1:nFeat;
    idx = idx+nFeat;
end
clear('DS0','LBDS0','DataPerVideo');
%%
%model
SVM = templateSVM(...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', 0);
% KNN = templateKNN('NumNeighbors',1,'Standardize',1);
tic
disp('Reading Data....')
type = 'LOCO';%'LOAO'
path2read = 'F:\MuHAVi(RGB)_NewAnnotations';
setDir  = fullfile(path2read);
imgSets = imageSet(setDir, 'recursive');
toc
val1 = 1;
if strcmp(type,'LOCO')
    val2 = 8;
elseif strcmp(type,'LOAO')
    val2 = 7;
elseif strcmp(type,'LOSO')
    val2 = length(imgSets);
end

if ~(strcmp(type,'LOSO'))
data_names1 = DataNamesSplit(type,DataNames1);
LabelsPerVideo = DataNamesSplit(type,LabelsPerVideo);
end

predicted_label = [];
actual_label = [];

%%
% for 
k = 8;
    
%     CamS = 1;
%     CamT = 2;


%%%%%%%%%%%%%%%%% LOSO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(type,'LOSO')
   [TrainData1,TestData1,TestDataNames1,TrainDataNames1]...
            = LOSO(imgSets,Data1,DataNames1,k); 
else
    [TrainData1,TestData1,TestDataNames1,TrainDataNames1,TrainPos,TestPos,...
        nFeatTrain,nFeatTest]= DataSplit(type,k,data_names1,Data1,DataNames1,...
        PosFeat,LabelsPerVideo);
end

%     disp('PCA')
% load('E:\C3D-tensorflow-master\PCA\Test_Cam3.mat');
%     %apply PCA
%     
% %     [pc1,score,latent,tsquare] = pca(TrainData1);
%     
%     DimPCA = 128;
% 
%     TrainData1=TrainData1*(pc1(:,1:DimPCA)); % in this way reduced data will have size of 2000*k (k=15);
%     TestData1=TestData1*(pc1(:,1:DimPCA)); % in this way reduced data will have size of 2000*k (k=15);

%%%%%%%%%%%%%%%%%%%%%%Simple Average Pool%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    [Xs,train_label] = Pool(imgSets,TrainDataNames1,TrainData1,'max');

    [Xt,test_label] = Pool(imgSets,TestDataNames1,TestData1,'max');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  K-SVD  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%     DataClassNames = FileName2ClassName(TrainDataNames1);
% 
%     disp('K-SVD')
% %%%%%%%%%%%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Y11 = double(TrainData1);
%     class_names = unique(DataClassNames);
%     Y11_label = Names2Labels(TrainDataNames1, class_names);
%     Y11_mat = convert_labelvec_to_mat(Y11_label, length(Y11_label), length(class_names));
% 
%     params.num_classes         = length(class_names);
%     params.card                = 1; %sparsity(number of non-zero values in columns)
%     params.init_dic = 'random';
%     params.alg_type = 'KSVD';
%     params.num_runs = 3;
%     params.iter = 5;
%     params.dict_size = 12;
%     rng('default');
% 
% [dic_cell_11, Sparse_cell_1] = My_KSVD_trainer(params, double(Y11'), Y11_mat);
% D11 = [];
% 
% for i = 1:params.num_classes
%     D11 = [D11,dic_cell_11{i}];
% end
% lamda = 20;%round(size(D11,2)*0.1);%sparsity
% % D = [D11,D12];
% [Xs,train_label1] = OmpHistogram(imgSets,D11,TrainDataNames1,Y11,lamda);
% 
% [Xt,test_label1] = OmpHistogram(imgSets,D11,TestDataNames1,double(TestData1),lamda);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%    ST-VLAD

% disp('clustering...')
% %create the features vocabulary with the standard k-means, can be quite slow process
% k1=32; % the number of visual words for the vocabulary of the features
% tic
% [~, vocabFeatures]=kmeans(TrainData1, k1); 
% toc
% %create the vocabulary of the features position with the standard k-means
% k2=40;%floor(max(PosFeat)/2); % the number of visual words for the vocabulary of the features position
% [~, vocabPositions]=kmeans(TrainPos', k2); 
% 
% d = DimPCA;
% FeatDim = k1*d+k2*(d+k1);
% % FeatDim = k1*d;
% %obtain the final ST-VLAD encoding
% 
% disp('Encoding...')
% tic
% [Xs,train_label] = ST_VLAD_Encode(imgSets,TrainDataNames1,TrainData1,TrainPos,...
%     nFeatTrain,FeatDim,vocabFeatures,vocabPositions);
% 
% [Xt,test_label] = ST_VLAD_Encode(imgSets,TestDataNames1,TestData1,TestPos,...
%     nFeatTest,FeatDim,vocabFeatures,vocabPositions);
% toc
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear('TrainPos','TrainDataNames1','TrainData1','TestPos','TestDataNames1','TestData1');
%     %apply power normalization
    alpha=0.1;
    Xs = sparse(Xs);
    Xt = sparse(Xt);
    tic
    Xs(1:floor(end/2),:)=PowerNormalization(Xs(1:floor(end/2),:), alpha);
    Xt(1:floor(end/2),:)=PowerNormalization(Xt(1:floor(end/2),:), alpha);    
    Xs(floor(end/2):end,:)=PowerNormalization(Xs(floor(end/2):end,:), alpha);
    Xt(floor(end/2):end,:)=PowerNormalization(Xt(floor(end/2):end,:), alpha);    
    toc
%     Xs=NormalizeRowsUnit(Xs);
%     Xt=NormalizeRowsUnit(Xt);
    Xs = Xs';
    Xt = Xt';
    %%
% options = statset('UseParallel',true);
    disp('training')
    tic
    Mdl_SVM = fitcecoc(full(Xs),string(train_label),'Learners',SVM,'Coding',...
        'onevsall');
    CMdlSV = compact(Mdl_SVM);               % Discard training data
    CMdl = discardSupportVectors(CMdlSV);
%     Mdl_KNN = fitcecoc(Xs',train_label,'Learners',KNN);
    toc
    
    %predict label
    disp('predicting')
    [label] = predict(CMdl,full(Xt));
    predicted_label = [predicted_label label'];
    actual_label = [actual_label string(test_label)];
% end

conf_mat = confusionmat(actual_label, predicted_label,'Order',CMdl.ClassNames);%confusion matrix
tr = diag(conf_mat);%diagnol elements indicate number of sequences classified correctly
accuracy = (sum(tr)/sum(conf_mat(:)))*100;

plotConfMat(conf_mat, CMdl.ClassNames);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data_names = DataNamesSplit(type,DataNames)
if strcmp(type,'LOCO')
    idx = 3;
elseif strcmp(type,'LOAO')
    idx = 2;
end

data_names = cell(length(DataNames),1);
for i = 1:length(DataNames)
    s1 = strsplit(DataNames{i},'_');
    data_names{i} = s1{idx};
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [TrainData,TestData,TestDataNames,TrainDataNames,TrainPos,TestPos,...
    nFeatTrain,nFeatTest] = DataSplit(type,k,data_names,Data,DataNames,...
    nFeat,LabelsPerVideo)
if strcmp(type,'LOCO')
    fprintf('leave %d camera out\n',k);
    %split into training and test
    match1 = strcmp(sprintf('Camera%d',k),data_names);
    match2 = strcmp(sprintf('Camera%d',k),LabelsPerVideo);
elseif strcmp(type,'LOAO')
    fprintf('leave %d actor out\n',k);
    %split into training and test
    match1 = strcmp(sprintf('Person%d',k),data_names);
    match2 = strcmp(sprintf('Person%d',k),LabelsPerVideo);
end
    %split into training and test
    disp('spliting')
train_index = find(match1==0);
test_index = find(match1==1);
TrainData = Data(train_index,:);
TestData = Data(test_index,:);
TestDataNames = DataNames(test_index);
TrainDataNames = DataNames(train_index);
TrainPos = nFeat(train_index)./max(nFeat);
TestPos = nFeat(test_index)./max(nFeat);
nFeatTrain = sum(match2==0);
nFeatTest = sum(match2==1);
end

function [features,label] = ST_VLAD_Encode(imgSets,DataNames,Data,DataPos,...
    nFeat,FeatDim,vocabFeatures,vocabPositions)
label = cell(1,nFeat);
features = zeros(FeatDim,nFeat);
tic
idx = 1;
for j = 1:length(imgSets)

    name = imgSets(1,j).Description;
    s1 = strsplit(name,'_');
    class_labels = (s1(1));

    %train features
    m1 = strcmp(name, DataNames);
    f1 = Data((m1 == 1),:);
    f_pos = (DataPos(m1==1))';
    if(~isempty(f1))
        ST_VLAD_encoding=ST_VLAD(f1, vocabFeatures, f_pos, vocabPositions);
%         VLAD_length = length(ST_VLAD_encoding)-FeatDim;
%         features(:,idx) = ST_VLAD_encoding(VLAD_length+1:end)';
        features(:,idx) = ST_VLAD_encoding';

        label{idx} = (class_labels);
        idx = idx +1;
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [features,label] = Pool(imgSets,DataNames,Data,type)
label = [];
features = [];
tic
for j = 1:length(imgSets)

    name = imgSets(1,j).Description;
    s1 = strsplit(name,'_');
    class_labels = string(s1(1));

    %train features
    m1 = strcmp(name, DataNames);
    f1 = (Data((m1 == 1),:))';
    if(~isempty(f1))
        %%average pooling
        if(strcmp(type,'average'))
            H1 = sum(f1,2)/size(f1,2);
        elseif(strcmp(type,'sum'))
            H1 = sum(f1,2);
        elseif(strcmp(type,'max'))
            if(size(f1,2) > 1 )
                H1 = (max(f1'))';
                else
                    H1 = f1;
            end
        end
        features = [features,H1];

        label = [label,class_labels];
    end
end
toc
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ClassName = FileName2ClassName(DataNames)
ClassName = [];
    for j = 1:length(DataNames)
        s1 = strsplit(DataNames{j},'_');
%             class_labels = string(Labels_MAS_8(s1(1)));
%         ClassName = [ClassName,string(class_labels)];
                ClassName = [ClassName,string(s1(1))];
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function labels = Names2Labels(names, class_names)
labels = zeros(size(names));
for j = 1:length(names)
    s1 = strsplit(names{j},'_'); 
%     out = Labels_MAS_8(s1(1));
%     idx = strcmp(class_names,out);
    idx = strcmp(class_names,s1(1));

labels(j) = find(idx == 1);
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [features,label] = OmpHistogram(imgSets,codebook,DataNames,Data,lamda)
label = [];
features = [];
tic
for j = 1:length(imgSets)

    name = imgSets(1,j).Description;
    s1 = strsplit(name,'_');
    class_labels = string(s1(1));

    %train features
    m1 = strcmp(name, DataNames);
    f1 = (Data((m1 == 1),:))';
    if(~isempty(f1))
        GAMMA = omp(codebook'*f1,codebook'*codebook,lamda);
        %%sum pooling
%         H1 = sum(full(GAMMA),2);

        %%max pooling
%         if(size(f1,2) > 1 )
%         H1 = max(full(GAMMA'));
%         else
%             H1 = full(GAMMA');
%         end
        
        %%average pooling
        H1 = sum(full(GAMMA),2)/size(GAMMA,2);
        features = [features,H1];
        label = [label,class_labels];
    end
end
toc
end

