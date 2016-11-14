%% Initialization
clear ; close all; clc
more off;

disp('Started running')

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 1800;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%Sort the data by num_labels, split into X and y
disp('Reading in data')
data = csvread('train.csv',1,0); % training data stored in arrays X, y
data_sort = sortrows(data,1);
clear data;
X = data_sort(:,2:end);
y = data_sort(:,1);
for i = 1:length(y);
	if y(i) == 0;
		y(i) = 10;
	end;
end;
clear data_sort;

%Set basic parameters
m = size(X, 1);
n = size(X, 2);
lambda = 2;

disp('Initializing weights')
%Randomly initalize weights for Theta
fprintf('\nLoading Saved Neural Network Parameters ...\n')
Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

%Randomly initalize weights for Theta_initial
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

disp('Doing fminunc')
%Doing fminunc (Training)
fprintf('\nTraining Neural Network... \n')
options = optimset('Display', 'iter' ,'MaxIter', 2000);
%options = optimset('GradObj','on','display','iter') 
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%[nn_params, cost] = fminunc(costFunction, initial_nn_params, options);
%[nn_params, cost] = fminbnd(costFunction, initial_nn_params, options);
%[nn_params, cost] = fminsearch(costFunction, initial_nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

disp('Doing predictions')
%Doing prediction
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%Sort the data by num_labels, split into X and y
disp('Reading in test data')
X_test = csvread('test.csv',1,0); % training data stored in arrays X, y
pred_test = predict(Theta1, Theta2, X_test);
for i = 1:length(pred_test);
  if pred_test(i) == 10;
    pred_test(i) = 0;
  end;
end;
imageid = [1:length(pred_test)]';
mysubmission = [imageid, pred_test];
csvwrite('thirdsubmission.csv',mysubmission);
