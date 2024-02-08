% Load the dataset
unzip('GTdb_crop.zip', 'cropped_faces');

% Create an imageDatastore and assign labels
imds = imageDatastore('cropped_faces');
imds.Labels = filenames2labels(imds.Files, "ExtractBetween", [1 3]);

% Define the range for training and testing subjects
trainSubjects = categorical(cellstr(strcat('s', num2str((1:40)', '%02d'))));
testSubjects = categorical(cellstr(strcat('s', num2str((41:50)', '%02d'))));

% Find indices for training and testing
trainIdx = find(ismember(imds.Labels, trainSubjects));
testIdx = find(ismember(imds.Labels, testSubjects));

% Create training and testing sets
imdsTrain = subset(imds, trainIdx);
imdsTest = subset(imds, testIdx);

% Check the number of files in each subset
disp(['Number of files in imdsTrain: ', num2str(numel(imdsTrain.Files))]);
disp(['Number of files in imdsTest: ', num2str(numel(imdsTest.Files))]);



net = vgg19;
inputSize = net.Layers(1).InputSize;
% Replace the classification head with a new one for your task
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));
layers = [layersTransfer;
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20);
    softmaxLayer;
    classificationLayer];

% Step 4: Set data augmentation and resizing parameters
pixelRange = [-20 20];
imageAugmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);

augimdsTest = augmentedImageDatastore(inputSize, imdsTest);

% Check the number of images in the training set
disp(['Number of training images: ', numel(imdsTrain.Files)])


options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ... % Reduced MiniBatchSize
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency', 3, ...
    'ValidationPatience', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Reset the datastore before training
reset(augimdsTrain);

% Train the network
netTransfer = trainNetwork(augimdsTrain, layers, options);

layer = 'fc7';
featuresTrain = activations(netTransfer,augimdsTrain,layer,'OutputAs','rows');
featuresValidation = activations(netTransfer,augimdsValidation,layer,'OutputAs','rows');


% Part 3 - Calculate cosine similarity scores

% Initialization
genuineScores = [];
impostorScores = [];
numSubjects = numel(unique(imdsValidation.Labels)); % Number of unique subjects


% Iterate over each subject
for subject = 1:numSubjects
    
    subjectCat = categorical({sprintf('s%02d', subject)});

    % Find indices of images for the current subject
    subjectIndices = find(imdsValidation.Labels == subjectCat);

    % Check if there are images for the current subject
    if isempty(subjectIndices)
        continue; % Skip to the next subject if no images are found
    end

    % Use the first image as the enrollment image
    enrollmentIdx = subjectIndices(1);
    enrollmentFeatures = featuresValidation(enrollmentIdx, :);

    % Genuine scores: Compare with remaining images of the same subject
    for i = 2:length(subjectIndices)
        genuineScore = 1 - pdist2(enrollmentFeatures, featuresValidation(subjectIndices(i), :), 'cosine');
        genuineScores = [genuineScores; genuineScore];
    end

    % Impostor scores: Compare with one image from each of the other subjects
    for otherSubject = 1:numSubjects
        if otherSubject ~= subject
            % Convert other subject index to categorical in the format 's01', 's02', etc.
            otherSubjectCat = categorical({sprintf('s%02d', otherSubject)});

            otherSubjectIdx = find(imdsValidation.Labels == otherSubjectCat, 1, 'first');
            if isempty(otherSubjectIdx)
                continue; % Skip if no image is found for the other subject
            end

            impostorScore = 1 - pdist2(enrollmentFeatures, featuresValidation(otherSubjectIdx, :), 'cosine');
            impostorScores = [impostorScores; impostorScore];
        end
    end
end

%Part 4 - Plot testing score distribution histograms
figure;
histogram(genuineScores, 'Normalization', 'probability', 'DisplayName', 'Genuine Scores');
hold on;
histogram(impostorScores, 'Normalization', 'probability', 'DisplayName', 'Impostor Scores');
xlabel('Cosine Similarity Scores');
ylabel('Probability');
title('Testing Score Distribution');
legend;

% Part 5 - Calculate ROC curve
labels = [ones(size(genuineScores)); zeros(size(impostorScores))];
scores = [genuineScores; impostorScores];
[~, ~, ~, auc] = perfcurve(labels, scores, 1);
[Xroc, Yroc, T, AUC] = perfcurve(labels, scores, 1);
figure; 
plot(Xroc, Yroc);
xlabel('False Positive Rate'); 
ylabel('True Positive Rate');
title(['ROC Curve, AUC = ' num2str(AUC)]);


% Part 6 - Calculate d' (d-prime)
meanGenuine = mean(genuineScores);
stdGenuine = std(genuineScores);
meanImpostor = mean(impostorScores);
stdImpostor = std(impostorScores);

dPrime = (meanGenuine - meanImpostor) / sqrt((stdGenuine^2 + stdImpostor^2) / 2);

% Display ROC AUC and d' (d-prime)
fprintf('ROC AUC: %.4f\n', auc);
fprintf('d'' (d-prime): %.4f\n', dPrime);