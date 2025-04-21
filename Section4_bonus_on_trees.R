# Bonus on trees - Demonstrating how tree models work with simple regression examples

# Load necessary library and prepare example datasets
library(ISLR)  

dat = na.omit(Hitters)  
y = dat$Salary  
x = dat$Hits    

# Example 2: Air quality dataset (commenting out but keeping for reference)
y = na.omit(airquality)$Ozone  # Response: Ozone level
x = na.omit(airquality)$Wind   # Predictor: Wind speed

# Create basic scatterplot of data
par(pch=20,lwd=4)  # Set plotting parameters: point style and line width
plot(x,y,pch=20)   # Plot Wind vs Ozone

# Fit a regression tree model
o = tree(y~x)  # Simple regression tree with one predictor

# Extract the split points from the tree
o$frame$splits[,1]  # Get the splits as character strings like "x<14.9"
cc = na.omit(unlist(strsplit(o$frame$splits[,1],"<")))  # Extract numeric values after "<"
cuts = c(min(x), sort(as.numeric(cc)), max(x))  # Create vector of cut points including min and max
L = length(cuts)  # Number of segments (number of cut points - 1)

# Initialize vector to store mean predictions for each segment
yms = numeric(L)  # Will store mean response for each segment

# For each segment between cut points
for(i in 1:L){
    # Find observations in current segment
    gp = which(x>=cuts[i] & x < cuts[(i+1)])
    
    # Calculate mean response in this segment
    yms[i] = mean(y[gp])
    
    # Draw horizontal line for this segment's prediction
    segments(x0=cuts[i],x1=cuts[(i+1)],
            y0=yms[i],y1=yms[i],col=i)  # Each segment gets different color
}

# Add linear regression line for comparison
abline(lm(y~x), col=8)  # Shows difference between tree model and linear model

# Advanced example: Fitting separate linear models in each segment
dat = data.frame(x=x,y=y)  # Create data frame for modeling
mms = matrix(NA,nrow=L,ncol=2)  # Matrix to store coefficients (intercept, slope)
plot(x,y,pch=20)  # New plot of the data

# For each segment defined by the tree
for(i in 1:L){
    # Find observations in current segment
    gp = which(x>=cuts[i] & x < cuts[(i+1)])
    
    # If segment contains data points
    if(length(gp)){ 
        # Fit linear model within this segment
        o = lm(y~.,data=dat[gp,]) 
        
        # Store coefficients
        mms[i,] = coef(o)
        
        # Predict at segment boundaries to draw line
        yp = predict(o,newdata=data.frame(x=c(cuts[i],cuts[(i+1)])))
        
        # Draw line segment for this local model
        segments(x0=cuts[i],x1=cuts[(i+1)],
                y0=yp[1],y1=yp[2],col=i)  # Each segment gets different color
    }
}
