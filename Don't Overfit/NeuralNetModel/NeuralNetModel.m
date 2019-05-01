clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 5;  
hidden_layer_size = 3;   
num_labels = 1;
lambda = 0.03 ;

%File Paths
train_data_path = './../data/train.csv';
test_data_path  = './../data/test.csv';
preds_file_path = './../predictions/';

% read csv Data
train_data = csvread(train_data_path);

% get the training data
x_train = train_data(2:end, 3:end);
y_train = train_data(2:end, 2);

%Imp_features
#imp_features = [34,  57,  59,  91, 117, 118, 124, 132, 135, 142, 151, 218, 224, 269, 270];
imp_features = [ 34,  59, 118, 218, 269];
x_train = x_train(:, imp_features);

%fit to the model
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);


initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, x_train, y_train, lambda);
options = optimset('MaxIter', 100);                                   
                                   
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



y_train_pred = predict(Theta1, Theta2, x_train);       
sum(y_train_pred)    
% get the test data
test_data = csvread(test_data_path);
x_test = test_data(2:end, 2:end);
x_test = x_test(:,imp_features);
y_pred = predict(Theta1, Theta2, x_test);
sum(y_pred)
#csvwrite('imp_featuresHl10.csv', [test_data(2:end, 1) y_pred])




