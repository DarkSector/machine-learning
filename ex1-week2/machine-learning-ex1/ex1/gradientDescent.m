function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

normalizedLearningRate = alpha/m; % scalar

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % X is 97 x 2 & theta is 2 x 1
    % h is 97 x 1
    h = X * theta;

    % temp var = theta0 - (alpha/m) * (h {97 x 1} - y {97 x1})
    temp0 = theta(1) - normalizedLearningRate * sum(h - y); % {h-y is 97x1}
    temp1 = theta(2) - normalizedLearningRate * sum(X(:, 2)' * (h - y)); % X is 97 x 2 & h -y is 97 x 1

    theta = [temp0; temp1];

    
    cost = computeCost(X, y, theta)


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = cost;

end

end
