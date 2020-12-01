function reco = reconstruct_face(Eigenfaces_PCA, eigenfaces_count,...
                                 tra_neu_mean, image)
    feat_len = size(Eigenfaces_PCA, 1);
    train_len = size(Eigenfaces_PCA, 2);
    weights = zeros(eigenfaces_count, 1);
    reco = zeros(feat_len, 1);
    % In the below loop we are finding the weights corresponding to each
    % eigen face. This is outlined in section 2.2 of the main paper
    for i = 1:eigenfaces_count
        weights(i) = Eigenfaces_PCA(:, i)' * image;
        % After gettting each weight the reconstruction is formed by a
        % weighted sum of the eigenfaces. This is Outlined in section 2.3
        % of the main paper.
        eigen_sum = weights(i) * Eigenfaces_PCA(:, i);
        reco = reco + eigen_sum;
    end
    % Finally, the reconstruction is still in shifted. We should shift it
    % by the train mean so that it is correct. As outlined in section 2.3 
    % and 2.4 o the main paper.
    reco = reco + tra_neu_mean;
end