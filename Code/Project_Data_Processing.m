% *******************************************
% * (A) DATA PROCESSING & EIGEN FACE CALC. **
% *******************************************

% This script is used to load and process the data. The images are read
% from the corresponding folder, and grouped into train and test sets. The
% data is converted to double type to make the calculations more precise. 
% Neutral and smiling faces have separate sets (see below). The mean of 
% the datasets are calculated, and stored in dataset_mean. Finally
% eigenfaces are calculated according to the Neutral Training Set.
% This script also includes the solution to part A.

clc
clearvars -except S

% Variables below are created to find and store 4 datasets as follows:
% 1- Train dataset for neutral faces. FIrst 190 datapoints from neutral.
% 2- Test  dataset for neutral faces. Last   10 datapoints from neutral.
% 3- Train dataset for smiling faces. First 190 datapoints from smiling.
% 4- Test  dataset for smiling faces. Last   10 datapoints from smiling.

% *********** IMPORTANT NOTE *******************
% Notice I created my own folder structure to make reading data easier. 
Train_Neutral  = 'trainset\neutral\';
Train_Smiling  = 'trainset\smiling\';
Test_Neutral   = 'testset\neutral\';
Test_Smiling   = 'testset\smiling\';
dataset_names = {Train_Neutral, Train_Smiling, Test_Neutral,...
          Test_Smiling};
% *********** IMPORTANT NOTE *******************
     
% Loop below goes over each dataset and reads the images (in array form)
for i = 1:4
    current = dataset_names{i};
    jpgfiles = dir(fullfile(current,'\*.jpg*'));
    number_of_jpgs = numel(jpgfiles);
    current_dataset = [];
    for j = 1:number_of_jpgs
        im = jpgfiles(j).name;
        im1 = imread(fullfile(current, im));
        im1 = im2double(im1);
        [irow, icol] = size(im1);
%         temp = reshape(im1, irow * icol, 1);
        current_dataset(:,j) = im1(:);
%         figure()
%         imshow(im1);
    end
    datasets{i} = current_dataset;
end

% Dataset names are assigned here. Variable names explain themselves.
Train_Neutral = datasets{1};
Train_Smiling = datasets{2}; 
Test_Neutral  = datasets{3};
Test_Smiling  = datasets{4};

tra_neu_mean = mean(Train_Neutral, 2); 
tra_smi_mean = mean(Train_Smiling, 2); 
tes_neu_mean = mean(Test_Smiling,  2); 
tes_smi_mean = mean(Test_Smiling,  2); 

feat_len  = size(Train_Neutral, 1);
train_len = size(Train_Neutral, 2);
test_len  = size(Test_Neutral,  2);

% Normalizing the train sets by subtracting mean, storing it
Tra_Neu_Norm = Train_Neutral - repmat(tra_neu_mean, 1, train_len);
Tra_Smi_Norm = Train_Smiling - repmat(tra_smi_mean, 1, train_len);
Tes_Neu_Norm = Test_Neutral  - repmat(tes_neu_mean, 1, test_len);
Tes_Smi_Norm = Test_Smiling  - repmat(tes_smi_mean, 1, test_len);


% Matrix L constructed as outlined in the paper:
% "Eifenfaces for recognition" by Turk and Pentland
% Link: http://www.face-rec.org/algorithms/pca/jcn.pdf
% Getting the eigen values and eigenvectors of the matrix
L = Tra_Neu_Norm' * Tra_Neu_Norm; 
[V, D] = eig(L); 
D_vector = diag(D);
% Sorting the eigenvalues to get the best eigenvectors
% Best here means the ones that explain maximum variance. (page 5 of  the
% above paper, mentioned just before formula (6)
[values, indices] = sort(D_vector,'descend');
% Below we calculate the cumulative percent variance explained as we keep
% on adding more eigenvalues.
variance_explained = cumsum(values) / sum(values);
variance_explained = variance_explained >= 0.9;
% The cutoff below shows how many eigenfaces we need to get 90% variance
% explained. (In a traditional PCA sense, however here we won't be able to
% explain the full variance because our PCA method is not perfect when we
% use the matrix L).
cutoff = find(variance_explained, 1, 'first');
fprintf('%i eigen faces needed to explain 90 percent variance.\n', cutoff);

% Singular Value Decomposition of the original (not shifted) data matrix
% to plot the Singular Values as asked.
% I usually comment the below line as it takes a long time. I just keep the
% S variable in the workspace.
[~, S, ~] = svd(Tra_Neu_Norm);
S_diag = diag(S);

% Plotting Eigenvalues of matrix L
figure()
stem(1:train_len, values, 'o'); hold on;
xline(cutoff);
legend("90 percent variance explained cutoff.")
title('Eigenvalues of the Matrix L')
ylabel("Eigen value i's magnitude"); xlabel("Eigen value i"); hold off;

% Plotting singular values of the original data matrix (not shifted)
figure()
stem(1:train_len, S_diag, 'o');
title('Singular Values of Neutral Normalized Train Set')
ylabel("Singular value i's magnitude"); xlabel("Singular value i");

% Eigenfaces are calculated from the matrix product of the normalized
% dataset with each eigenvector. (formula (6) from the above paper)
eigenfaces_count = cutoff;
Eigenfaces_PCA = zeros(feat_len, cutoff);

for count = 1 : eigenfaces_count
    Eigenfaces_PCA(:,count) = Tra_Neu_Norm * V(:, indices(count));
    eigen_face_norm = norm(Eigenfaces_PCA(:,count));
    Eigenfaces_PCA(:,count) = Eigenfaces_PCA(:,count) / eigen_face_norm;
end

% *******************************************
% *********** PLOTTING EIGENFACES ***********
% *******************************************

% Plotting the first 9 Eigenfaces for reference, also as a sanity check.
% From the printed results, they seem as expected.
figure()
image_count = 9;
for count = 1:image_count
    eigenface = Eigenfaces_PCA(:, count);
    % Normalazing the eigenface to print it as grayscale image
    min_val = min(eigenface); max_val = max(eigenface);
    eigenface =((eigenface - min_val).*1)./(max_val - min_val);
    % Reshaping into original image dimensions    
    eigenface = reshape(eigenface, irow, icol);
    subplot(3, 3, count);
    imshow(eigenface)
    title(sprintf('Eigenface %d', count));
end
% sgtitle('Q5 a) The 9 highest variance eigenfaces derived by PCA.');

% *******************************************
% ********** PLOTTING RECONSTRUCTIONS *******
% *******************************************

% Plotting some of the reconstructions with all the 190 eigenfaces, as
% reference and also for sanity check
figure()
for face = 1:4
    face_id = face + 6;
    % reconstruct_face is my own function defined in its corresponding .m
    % file. It reconstructs a face from the dataset given the valid
    % parameters as below.
    reco = reconstruct_face(Eigenfaces_PCA, eigenfaces_count,...
                                 tra_neu_mean, Tra_Neu_Norm(:, face_id));
    reco = reshape(reco, irow, icol);
    subplot(4, 2, face * 2 - 1);
    % Original Image
    im = reshape(Train_Neutral(:, face_id), irow, icol);
    imshow((im));
    title('Original Image');
    % Reconstruction image
    subplot(4, 2, face * 2);
    imshow(reco);
    title(sprintf('Reconstruction, using %i eigen faces', cutoff));
end
% sgtitle('Q5 a) The 9 highest variance eigenfaces derived by PCA.');

% Loading a car image for future questions. Resizing the image to the same
% dimensions as the face images.
RGB = imread('car.jpg');
I = rgb2gray(RGB);
scaling = irow / size(I, 1);
J = imresize(I, scaling);
cropping = (size(J, 2) - icol) / 2;
car_image = J(:, cropping : size(J, 2) - cropping - 1);
car_image = im2double(car_image);
figure()
imshow(car_image)
title('Non face image');
car_image = reshape(car_image, irow * icol, 1);

% Resetting the cutoff value, I want to use all 190 eigenfaces for the next
% parts of the project. I am doing this manily to see if I will reach 0 MSE
% while working with train data.
cutoff = 190;
% Eigenfaces are calculated from the matrix product of the normalized
% dataset with each eigenvector. (formula (6) from the above paper)
eigenfaces_count = cutoff;
Eigenfaces_PCA = zeros(feat_len, cutoff);

for count = 1 : eigenfaces_count
    Eigenfaces_PCA(:,count) = Tra_Neu_Norm * V(:, indices(count));
    eigen_face_norm = norm(Eigenfaces_PCA(:,count));
    Eigenfaces_PCA(:,count) = Eigenfaces_PCA(:,count) / eigen_face_norm;
end


% Clearing unnecessary data -except option to keep relevant data
clearvars -except Train_Neutral Train_Smiling Test_Neutral Test_Smiling ...
          irow icol tra_neu_mean tra_smi_mean feat_len train_len ...
          Eigenfaces_PCA Tra_Neu_Norm Tra_Smi_Norm Tes_Neu_Norm ...
          Tes_Smi_Norm tes_neu_mean tes_smi_mean test_len car_image ...
          S cutoff

% Saving relevant data for future questions
save Project_Data