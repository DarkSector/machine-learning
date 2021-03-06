function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

normalizationParamater = 1/(2*m);
sum = 0;
for c = 1:m
	% Size of theta is 1 x 2
	% Size of X is 2 x 1
	h = theta' * X(c, :)';
	difference = (h - y(c, 1))^2;
	sum += difference;
end


J = normalizationParamater * sum;



% =========================================================================

end
