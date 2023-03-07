function a = derivlogsig(x)
    % calculate derivative of log-sig at any point x
    fx = logsig(x);
    a = fx.*(1-fx);
end

