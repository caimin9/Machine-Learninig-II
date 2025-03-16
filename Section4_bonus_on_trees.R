# Bonus on trees

# pick an example
library(ISLR)
dat = na.omit(Hitters)
y = dat$Salary
x = dat$Hits

y = na.omit(airquality)$Ozone
x = na.omit(airquality)$Wind

par(pch=20,lwd=4)
plot(x,y,pch=20)

o = tree(y~x)
o$frame$splits[,1]
cc = na.omit(unlist(strsplit(o$frame$splits[,1],"<")))
cuts = c(min(x), sort(as.numeric(cc)), max(x))
L = length(cuts)
yms = numeric(L)
for(i in 1:L){
	gp = which(x>=cuts[i] & x < cuts[(i+1)])
	yms[i] = mean(y[gp])
	segments(x0=cuts[i],x1=cuts[(i+1)],
				y0=yms[i],y1=yms[i],col=i)
}
# compare to regression line
abline(lm(y~x), col=8)

# what if... we fitted models instead?
dat = data.frame(x=x,y=y)
mms = matrix(NA,nrow=L,ncol=2)
plot(x,y,pch=20)
for(i in 1:L){
	gp = which(x>=cuts[i] & x < cuts[(i+1)])
	if(length(gp)){ 
		o = lm(y~.,data=dat[gp,]) 
		mms[i,] = coef(o)
		yp = predict(o,newdata=data.frame(x=c(cuts[i],cuts[(i+1)])))
		segments(x0=cuts[i],x1=cuts[(i+1)],
				y0=yp[1],y1=yp[2],col=i)
	}
}
