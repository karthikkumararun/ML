# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
#library(caTools)
#set.seed(123)
#split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
#training_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting linear regression model to the dataset.
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

# Fititng polynomial regression model to the dataset.
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ . ,
              data = dataset)

# Visualizing the linear regression results.
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level,y = dataset$Salary), colour='Red') +
  geom_line(aes(x = dataset$Level ,y = predict(lin_reg , newdata = dataset) , colour = 'Blue')) +
  xlab('Position Level') +
  ylab('Salary') +
  ggtitle('Truth or Bluff (Linear Regression Results)')
  

# Visualizing the Polynomial regression results.

ggplot() +
  geom_point(aes(x = dataset$Level,y = dataset$Salary), colour='Red') +
  geom_line(aes(x = dataset$Level ,y = predict(poly_reg , newdata = dataset) , colour = 'Blue')) +
  xlab('Position Level') +
  ylab('Salary') +
  ggtitle('Truth or Bluff (Polynomial Regression Results)')

# Predicting a new result with Linear Regression
y_pred = predict(lin_reg , data.frame(Level = 6.5))

# Predicting a new result with Linear Regression
y_pred_Poly = predict(poly_reg , data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4))
