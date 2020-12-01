% RUN ME!!
% Project_Data_Processing
clc; clear all;
load Project_Data;

% ********************************************************************
% *************** (B) MSE CALC. FOR TRAIN NEUTRAL FACE ***************
% ********************************************************************

face_id = 18;
eigenfaces_count = cutoff;
MSEs = zeros(cutoff, 1);

figure();
plot_count = 1;
subplot(3,3,plot_count);
im = Train_Neutral(:, face_id);
im = reshape(im, irow, icol);
imshow(im);
title(sprintf('Neutral Tra Img'));
MSEs = zeros(cutoff, 1);
for count = 1:eigenfaces_count
    reco = reconstruct_face(Eigenfaces_PCA, count, tra_neu_mean, ...
                            Tra_Neu_Norm(:, face_id));  
    cur_image = Train_Neutral(:, face_id);
    MSEs(count) = sum((reco - cur_image).^2, 1) / size(reco, 1);
    
    if 0 == rem(count, 25) || count == 190
        plot_count = plot_count + 1;
        subplot(3,3,plot_count);
        % Reconstruction image
        reco = reshape(reco, irow, icol);
        imshow(reco);
       title(sprintf('Recon Img %i eigenfaces', count));
    end
end

figure();
stem([1:eigenfaces_count], MSEs, 'o');
title('MSEs plot, Neutral Train Image Reconstruction');

% ********************************************************************
% *************** (C) MSE CALC. FOR TRAIN SMILING FACE ***************
% ********************************************************************

figure();
plot_count = 1;
subplot(3,3,plot_count);
im = Train_Smiling(:, face_id);
im = reshape(im, irow, icol);
imshow(im);
title(sprintf('Smiling Tra Img'));
MSEs = zeros(cutoff, 1);
for count = 1:eigenfaces_count
    reco = reconstruct_face(Eigenfaces_PCA, count, tra_neu_mean, ...
                            Tra_Smi_Norm(:, face_id));  
    cur_image = Train_Smiling(:, face_id);
    MSEs(count) = sum((reco - cur_image).^2, 1) / size(reco, 1);
    
    if 0 == rem(count, 25) || count == 190
        plot_count = plot_count + 1;
        subplot(3,3,plot_count);
        % Reconstruction image
        reco = reshape(reco, irow, icol);
        imshow(reco);
       title(sprintf('Recon Img %i eigenfaces', count));
    end
end

figure();
stem([1:eigenfaces_count], MSEs, 'o');
yl = ylim; ylim([0, yl(2)]);
title('MSEs plot, Smiling Train Image Reconstruction');
xlabel('number of eigen faces used in reconstruction')
ylabel('MSE')

% ********************************************************************
% *************** (D) MSE CALC. FOR TEST NETURAL FACE ****************
% ********************************************************************

figure();
face_id = 8;
plot_count = 1;
subplot(3,3,plot_count);
im = Test_Neutral(:, face_id);
im = reshape(im, irow, icol);
imshow(im);
title(sprintf('Neutrak Test Img'));
MSEs = zeros(cutoff, 1);
for count = 1:eigenfaces_count
    reco = reconstruct_face(Eigenfaces_PCA, count, tra_neu_mean, ...
                            Tes_Neu_Norm(:, face_id));  
    cur_image = Test_Neutral(:, face_id);
    MSEs(count) = sum((reco - cur_image).^2, 1) / size(reco, 1);
    
    if 0 == rem(count, 25) || count == 190
        plot_count = plot_count + 1;
        subplot(3,3,plot_count);
        % Reconstruction image
        reco = reshape(reco, irow, icol);
        imshow(reco);
       title(sprintf('Recon Img %i eigenfaces', count));
    end
end

figure();
stem([1:eigenfaces_count], MSEs, 'o');
yl = ylim; ylim([0, yl(2)]);
title('MSEs plot, Neutral Test Image Reconstruction');
xlabel('number of eigen faces used in reconstruction')
ylabel('MSE')

% ********************************************************************
% ************** (E) RECONSTRUCTION FOR NON FACE IMAGE ***************
% ********************************************************************

figure();
plot_count = 1;
subplot(3,3,plot_count);
im = car_image;
im = reshape(im, irow, icol);
imshow(im);
title(sprintf('Non Face Image'));
MSEs = zeros(cutoff, 1);
for count = 1:eigenfaces_count-1
    reco = reconstruct_face(Eigenfaces_PCA, count, tra_neu_mean, ...
                            car_image);  
    cur_image = car_image;
    MSEs(count) = sum((reco - cur_image).^2, 1) / size(reco, 1);
    
    if 0 == rem(count, 25)
        plot_count = plot_count + 1;
        subplot(3,3,plot_count);
        % Reconstruction image
        reco = reshape(reco, irow, icol);
        imshow(reco);
       title(sprintf('Recon Img %i eigenfaces', count));
    end
end

figure()
stem([1:eigenfaces_count], MSEs, 'o')
yl = ylim; ylim([0, yl(2)]);
title('MSEs plot, Non Face Image Reconstruction')
xlabel('number of eigen faces used in reconstruction')
ylabel('MSE')
