x = [1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]';
deads = [ 6, 13, 18, 28, 52, 53, 61, 60]';
total = [59, 60, 62, 56, 63, 59, 62, 60]';
y = deads ./ total;

polyfit(x, y, 1)
polyfit(x, y, 2)

b = glmfit(x, y,'binomial','link','logit')
yfit = glmval(b, x, 'logit')
plot(x, y,'o',x, yfit,'-','LineWidth',2)

glmfit(x, [deads total], 'binomial', 'logit')