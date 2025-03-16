# --------------------------------------------------------
# ST4061 / ST6041 / ST6042
# 2024-25
# Eric Wolsztynski
# ...
# Section 5: demo code for effect of kernel on SVM
# Here we simulate 2D data that have a circular spatial 
# distribution, to see the effect of the choice of kernel 
# shape on decision boundaries
# --------------------------------------------------------

rm(list=ls())
library(e1071)

# Simulate circular data...

# simulate 2-class circular data:
set.seed(4061)
n = 100
S1 = 15; S2 = 3
x1 = c(rnorm(60, m=0, sd=S1), rnorm(40, m=0, sd=S2))
x2 = c(rnorm(60, m=0, sd=S1), rnorm(40, m=0, sd=S2))
# corresponding 2D circular radii:
rads = sqrt(x1^2+x2^2)
# make up the 2 classes in terms of whether lower or 
# greater than median radius:
c1 = which(rads<median(rads))
c2 = c(1:n)[-c1]
# now we apply scaling factors to further separate the 
# 2 classes:
x1[c1] = x1[c1]/1.2
x2[c1] = x2[c1]/1.2
x1[c2] = x1[c2]*1.2
x2[c2] = x2[c2]*1.2
# label data according to class membership:
lab = rep(1,n)
lab[c2] = 2
par(mfrow=c(1,1))
plot(x1,x2,col=c(1,2)[lab], pch=c(15,20)[lab], cex=1.5)

# create final data frame:
x = data.frame(x1,x2)
y = as.factor(lab)
dat = cbind(x,y)

# apply SVMs with different choices of kernel shapes:
svmo.lin = svm(y~., data=dat, kernel='linear')
svmo.pol = svm(y~., data=dat, kernel='polynomial')
svmo.rad = svm(y~., data=dat, kernel='radial')
svmo.sig = svm(y~., data=dat, kernel='sigmoid')

plot(svmo.lin, dat, col=c("cyan","pink"), svSymbol=15)
plot(svmo.pol, dat, col=c("cyan","pink"), svSymbol=15)
plot(svmo.rad, dat, col=c("cyan","pink"), svSymbol=15)
plot(svmo.sig, dat, col=c("cyan","pink"), svSymbol=15)

#### NOTE: the code below is outside the scope of this course! 
#### It is used here for illustrations purposes only.

# this call format is easier when using predict():
svmo.lin = svm(x, y, kernel='linear', scale=F)
svmo.pol = svm(x, y, kernel='polynomial', scale=F)
svmo.rad = svm(x, y, kernel='radial', scale=F)
svmo.sig = svm(x, y, kernel='sigmoid', scale=F)

# evaluate the SVM boundaries on a regular 2D grid of points:
ng    = 50
xrg   = apply(x, 2, range)
x1g   = seq(xrg[1,1], xrg[2,1], length=ng)
x2g   = seq(xrg[1,2], xrg[2,2], length=ng)
xgrid = expand.grid(x1g, x2g)
plot(x, col=c(1,2)[y], pch=20)
abline(v=x1g, col=8, lty=1)
abline(h=x2g, col=8, lty=1)
#
ygrid.lin = predict(svmo.lin, xgrid)
ygrid.pol = predict(svmo.pol, xgrid)
ygrid.rad = predict(svmo.rad, xgrid)
ygrid.sig = predict(svmo.sig, xgrid)

par(mfrow=c(2,2), font.lab=2, font.axis=2)
CEX = .5
COLS = c(1,3)
DCOLS = c(2,4)
#
plot(xgrid, pch=20, col=COLS[as.numeric(ygrid.lin)], cex=CEX, main="Linear kernel")
points(x, col=DCOLS[as.numeric(y)], pch=20)
# points(x[svmo.lin$index,], pch=21, cex=2)
points(svmo.lin$SV, pch=21, cex=2) # same as previous line!
#
plot(xgrid, pch=20, col=COLS[as.numeric(ygrid.pol)], cex=CEX, main="Polynomial kernel")
points(x, col=DCOLS[as.numeric(y)], pch=20)
points(x[svmo.pol$index,], pch=21, cex=2)
#
plot(xgrid, pch=20, col=COLS[as.numeric(ygrid.rad)], cex=CEX, main="Radial kernel")
points(x, col=DCOLS[as.numeric(y)], pch=20)
points(x[svmo.rad$index,], pch=21, cex=2)
#
plot(xgrid, pch=20, col=COLS[as.numeric(ygrid.sig)], cex=CEX, main="Sigmoid kernel")
points(x, col=DCOLS[as.numeric(y)], pch=20)
points(x[svmo.sig$index,], pch=21, cex=2)

# Alternative plot:
par(mfrow=c(2,2), font.lab=2, font.axis=2)
CEX = .5
COLS = c(1,3)
DCOLS = c(2,4)
#
L1 = length(x1g)
L2 = length(x2g)
#
plot(xgrid, pch=20, col=COLS[as.numeric(ygrid.lin)], cex=CEX, main="Linear kernel")
bnds = attributes(predict(svmo.lin, xgrid, decision.values=TRUE))$decision
contour(x1g, x2g, matrix(bnds, L1, L2), level=0, add=TRUE, lwd=2)
#
plot(xgrid, pch=20, col=COLS[as.numeric(ygrid.pol)], cex=CEX, main="Polynomial kernel")
bnds = attributes(predict(svmo.pol, xgrid, decision.values=TRUE))$decision
contour(x1g, x2g, matrix(bnds, L1, L2), level=0, add=TRUE, lwd=2)
#
plot(xgrid, pch=20, col=COLS[as.numeric(ygrid.rad)], cex=CEX, main="Radial kernel")
bnds = attributes(predict(svmo.rad, xgrid, decision.values=TRUE))$decision
contour(x1g, x2g, matrix(bnds, L1, L2), level=0, add=TRUE, lwd=2)
#
plot(xgrid, pch=20, col=COLS[as.numeric(ygrid.sig)], cex=CEX, main="Sigmoid kernel")
bnds = attributes(predict(svmo.sig, xgrid, decision.values=TRUE))$decision
contour(x1g, x2g, matrix(bnds, L1, L2), level=0, add=TRUE, lwd=2)

# NB: naive Bayes decision boundary is obtained with
# contour(x1g, x2g, matrix(bnds, L1, L2), level=0.5, add=TRUE, col=4, lwd=2)
