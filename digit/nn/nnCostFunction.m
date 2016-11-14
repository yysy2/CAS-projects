function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%disp(y)

%Firstly, convert y into vector:
eye_matrix = eye(num_labels);
%disp(y)
y_matrix = eye_matrix(y,:);
%disp(y_matrix(1))
%disp(X(1,1))
a1 = [ones(m, 1), X];
%disp(size(Theta1))
%disp(a1)
z2 = sigmoid(Theta1 * a1');
%disp(size(z2))
a2 = [ones(size(z2, 2), 1), z2'];
%disp(size(a2))
%disp(size(Theta2))
a3 = sigmoid(Theta2 * a2');
h = a3;

%disp(size(y_matrix))
%disp(size(h(1,:)))

%disp(num_labels)

%disp(size(y_matrix(:,1)))
%disp(size(y_matrix(:,1).*log(h(1,:)')))
%disp(size(sum(-y_matrix(:,1).*log(h(1,:)')-(1-y_matrix(:,1)).*log(1-h(1,:)'))))
J = 0;
J_unreg  = 0;
J_unreg = (1/m)*sum(sum(-y_matrix(:,:).*log(h(:,:)')-(1-y_matrix(:,:)).*log(1-h(:,:)')));
%disp(size(Theta1))
J = J_unreg + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).*Theta1(:,2:end)))+sum(sum(Theta2(:,2:end).*Theta2(:,2:end))));
%disp(size(J));
%disp(size(a3))
%disp(size(y_matrix))
%disp(size(X))
%disp(size(y))
delta3 = a3' - y_matrix;
%size(Theta2(:,2:end))
%size(delta3)
%size(Theta1)
%size(a1)
%size(a2)
%sigmoidGradient(delta3);
delta2 = (delta3 * Theta2(:,2:end)) .* sigmoidGradient(Theta1 * a1')';
cdelta2 = (a2' * delta3)';
cdelta1 = (a1' * delta2)';
%size(X)
%size(delta2)
%size(delta3)
%size(cdelta1)
%size(cdelta2)

%size(a1)
%size(z2)
%size(a2)
%size(a3)
%size(delta3)
%size(delta2)
%size(Theta1)
%size(Theta2)
%size(cdelta1)
%size(cdelta2)
Theta1_grad = (1/m) * cdelta1; % + lambda*Theta1';
Theta2_grad = (1/m) * cdelta2; % + lambda*Theta2';

%Theta2_grad(2:end,:) = Theta2_grad(2:end,:) + lambda*Theta2(:,2:end)';
%Theta1_grad(2:end,:) = Theta1_grad(2:end,:) + lambda*Theta1(:,2:end)';

Theta1_hold = Theta1;
Theta2_hold = Theta2;
Theta1_hold(:,1) = 0;
Theta2_hold(:,1) = 0;
Theta1_grad = Theta1_grad + (lambda/m)*Theta1_hold;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2_hold;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
