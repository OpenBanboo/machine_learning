function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% Loop over the centroids
for i = 1:K

    %initialize the sum of i-th centroid
    centroid_i_sum = zeros(1, n);
    % initialize the number of the samples with the same centroid i
    C_k = 0;
    
    % Loop over all the samples (with the same centroid i) and compute the sum of axis (x,y)
    for j = 1:m
        
        if idx(j) == i
            centroid_i_sum = centroid_i_sum + X(j,:);
            C_k = C_k + 1;
        end
    end
    
    % Calculate the mean
    centroids(i, :) = centroid_i_sum/C_k;
    
end

% =============================================================


end

