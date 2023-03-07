% T = readtable("train.csv", "NumHeaderLines", 1);
% 
% network = BackpropNetwork(784, 600, 10);
% 
% 
% for i = 1:42000
% 
%     [network, ~] = network.networkForward((T{i, 3:end})');
%     network = network.networkSensitivity(transformDigit(T{i,2}));
%     network = network.networkUpdate();
% end


[network, temp] = network.networkForward((T{42001:end, 3:end})');
temp

test.labels(:, 1)