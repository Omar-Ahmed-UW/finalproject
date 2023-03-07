function a = logsig(n)
    % log-sigmoid function to process net-inputs.
    a = 1./(1+exp(-n));
end