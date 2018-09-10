function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
input1 = X';
input1 = [ones(1,m);input1];
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

allA2 = sigmoid(Theta1 * input1);
allA2 = [ones(1,size(allA2,2));allA2];

finalOp = sigmoid(Theta2 * allA2);

[ele,i] = max(finalOp,[],1);

p = i';
for i = 1:size(p,1)
  if p(i) == 10
    p(i) = 0;
  endif
endfor




% =========================================================================


end
