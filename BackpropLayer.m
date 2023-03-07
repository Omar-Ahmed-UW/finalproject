classdef BackpropLayer
    properties
        W
        b
        mostRecentN
    end

    methods
        function obj = BackpropLayer(first, second)
            % checks to generate weight vectors and bias vector or set to
            % passed in vectors.
            [fr,fc] = size(first);
            [sr,sc] = size(second);
            if fr == 1 && fc == 1 && sr == 1 && sc == 1
                obj.W = (-1 + (1+1).*rand(second, first));
                obj.b = (-1 + (1+1).*rand(second, 1));
            
            else
                obj.W = first;
                obj.b = second;
            end        
        end

        function [obj, a] = layerForward(obj, P)
            % uses vectorized approach to calculate inner product of all
            % weights with inputs then adds the correct biases.
            
            % TODO omar: Shouldn't this statement be obj.W' * P?
            n = (obj.W * P) + obj.b;
            a = logsig(n);
            obj.mostRecentN = n;
        end

        function [obj, output] = layerSensitivity(obj, x)
            % calculates the sensitivity for this layer
            % x represents the component from the following layer (either
            % -2*(t-a) or the sum of sensitivity*weight for each neuron)
            output = derivlogsig(obj.mostRecentN).*x;
        end
        
        function [obj] = updateLayer(obj, learningRate, sensitivity, prevLayerOutput)
            % updates the weights and bias of this layer using the
            % learning rate sepcified in the network, the senstivity
            % computed for this layer, and the layer output of the layer
            % before it.
            obj.W = obj.W - learningRate*sensitivity*(prevLayerOutput');
            obj.b = obj.b - learningRate*sensitivity;        
        end

        function obj = batchUpdateLayer(obj, sumW, sumB, learnRate)
            % updates the weights and biases of this layer after a batch of
            % inputs have been sent through
            obj.W = obj.W - learnRate * sumW;
            obj.b = obj.b - learnRate * sumB;
        end
    end
end