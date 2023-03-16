val = rosen(1,1);
minima = rosenGrad(1,1);
hessen = rosenHess(1,1);


function val = rosen(x, y)
 val = (1-x)^2 + (100 * (y - (x^2))^2);
end

function Df = rosenGrad(x, y)
 k = [x-1; y-(x^2)];
 Df = [2*k(1)-(400*x*k(2)); 200*k(2)];
end

function H = rosenHess(x, y)
 df2dx2 = 2 - 400*y + 1200*(x^2);
 df2dy2 = 200;
 df2dxdy = -400 * x;
 H = [df2dx2, df2dxdy; df2dxdy, df2dy2];
end