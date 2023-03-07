function a = testPart2(epochs, hiddenSize, batchSize)
%TESTPART2 This function takes in some hyperparameters and returns a %
%accuracy for a network created with those parameters
    load('mnist.mat', 'test', 'training');
    network = BackpropNetwork(784, hiddenSize, 10);
    
    % hold the summed sensitivities for weights and biases for each layer
    sumW1 = zeros(hiddenSize, 784);
    sumB1 = zeros(hiddenSize, 1);
    sumW2 = zeros(10, hiddenSize);
    sumB2 = zeros(10, 1);

    numBatches = floor(60000/batchSize);
    numBatches = numBatches - 1;
    
    for k = 1:epochs
        for i = 0:numBatches
            for j = 1:batchSize
                [network, ~] = network.networkForward(training.images(:, i*batchSize+j));
                [network, sens1, sens2] = network.networkSensitivity(training.labels(:, i*batchSize+j));
                sumW1 = sumW1 + sens1*network.a0';
                sumB1 = sumB1 + sens1;
                sumW2 = sumW2 + sens2*network.a1';
                sumB2 = sumB2 + sens2;
            end
        
            network = network.batchUpdateNetwork(sumW1, sumB1, sumW2, sumB2);
            sumW1 = zeros(hiddenSize, 784);
            sumB1 = zeros(hiddenSize, 1);
            sumW2 = zeros(10, hiddenSize);
            sumB2 = zeros(10, 1);
        end
        % print k every other epoch
        if mod(k, 2) == 0
            disp(k);
        end
    end
    
    % now test how many the network correctly classifies
    count = 0;
    for m = 1:10000
        [network, temp] = network.networkForward(test.images(:, m));
        count = count + isequal(round(temp), test.labels(:, m));
    end

    a = count / 100.00000;
    % a is now a percent (instead of dividing by 10,000 and multiplying by
    % 100, I just divided by 10,000/100 = 100)
end