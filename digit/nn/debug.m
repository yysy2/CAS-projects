%% Initialization
clear ; close all; clc
more off;
%run 'news image'

disp('Started running')

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%Sort the data by num_labels, split into X and y
disp('Reading in data')
data = csvread('train.csv',1,0); % training data stored in arrays X, y
%data_sort = sortrows(data);
%disp(data_sort(1:10,1:10)i)
%return
%clear data;
X = data(:,2:end);
y = data(:,1);
for i = 1:length(y);
	if y(i) == 0;
		y(i) = 10;
	end;
end;
%clear data_sort;
%disp(X(27,70))
%disp(y(1:10))
%return

%Set basic parameters
m = size(X, 1);
n = size(X, 2);
lambda = 1;

%disp(m)
%disp(n)
%return

disp('Initializing weights')
%Randomly initalize weights for Theta
%fprintf('\nLoading Saved Neural Network Parameters ...\n')
%Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
%Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters 
%nn_params = [Theta1(:) ; Theta2(:)];

%Randomly initalize weights for Theta_initial
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = csvread('tt1.csv');
initial_Theta2 = csvread('tt2.csv');
%initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
%initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%disp(initial_Theta2(1:5,1:5))
%return
%disp('initial1')
%disp(initial_Theta1(:,26))
%disp('initial2')
%disp(initial_Theta2(:,26))
%return

J, grad = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

%disp(J)
return

%gtheta1 = reshape(grad(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
%gtheta2 = reshape(grad((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

%disp(gtheta1(:,25))
%disp(gtheta2(:,25))
%return

disp('Doing fminunc')
%Doing fminunc (Training)
fprintf('\nTraining Neural Network... \n')
options = optimset('Display', 'iter' ,'MaxIter', 100);
%options = optimset('GradObj','on','display','iter') 
%nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,num_labels, X, y, lambda);
%return
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%return
%[nn_params, cost] = fminunc(costFunction, initial_nn_params, options);
%[nn_params, cost] = fminbnd(costFunction, initial_nn_params, options);
%[nn_params, cost] = fminsearch(costFunction, initial_nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

disp('theta1')
disp(Theta1(:,26))
disp('theta2')
disp(Theta2(:,26))
return

%disp(Theta1(1:3,1:3))
%disp(size(Theta2))
%return

disp('Doing predictions')
%Doing prediction
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%disp(pred)
%disp(size(pred))
%disp(pred(1:5))
%disp(y(1:5))
%return

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
%csvwrite('thirdsubmission.csv',mysubmission);
