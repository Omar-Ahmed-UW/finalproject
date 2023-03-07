T = readtable("train.csv", "NumHeaderLines", 1);
network = BackpropNetwork(784, 100, 10);

sumW1 = zeros(100, 784);
sumB1 = zeros(100, 1);
sumW2 = zeros(10, 100);
sumB2 = zeros(10, 1);

for k = 1:10
    for i = 1:960
        for j = 1:50
            [network, ~] = network.networkForward((T{i*j, 3:end})');
            [network, temp1, temp2] = network.networkSensitivity(transformDigit(T{i*j, 2}));
            sumW1 = sumW1 + temp1*network.a0';
            sumB1 = sumB1 + temp1;
            sumW2 = sumW2 + temp2*network.a1';
            sumB2 = sumB2 + temp2;
        end
    
        network = network.batchUpdateNetwork(sumW1, sumB1, sumW2, sumB2);
        sumW1 = zeros(100, 784);
        sumB1 = zeros(100, 1);
        sumW2 = zeros(10, 100);
        sumB2 = zeros(10, 1);
    end
end

count = 0;
for m = 48001:60000
    [network, temp] = network.networkForward(T{m, 3:end});
    count = count + isequal(round(temp), transformDigit(T{m, 2}));
end

count / 100.00000
