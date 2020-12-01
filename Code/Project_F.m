% RUN ME!!
% Project_Data_Processing
clc; clear all;
load Project_Data;

face_id = 18;
face = reshape(Train_Neutral(:, face_id), irow, icol);
degrees = linspace(0, 360, 9);
degrees = degrees(1:8);
eigenfaces_count = cutoff;

% All the rotation operations are counter-clockwise

index = 0;
for j = 1:4
    figure()
    for i = 1:2
        index = index + 1;
        degree = degrees(index);
        J = imrotate(face,degree,'bilinear','crop');

        cur_image = J(:) - tra_neu_mean;
        reco = reconstruct_face(Eigenfaces_PCA, eigenfaces_count, tra_neu_mean, cur_image);  
        MSE = sum((reco - J(:)).^2, 1) / size(reco, 1);
        fprintf("MSE for %i degrees", degree);
        MSE

        subplot(2, 2, i * 2 - 1);
        im = reshape(J, irow, icol);
        imshow((im));
        title(sprintf('Rotated Img, %i degrees', degree));
        % Reconstruction image
        subplot(2, 2, i * 2);
        reco = reshape(reco, irow, icol);
        imshow(reco);
        title('Reconstruction Image');
    end
end
