#############################################
#                                           #
# Author:     Sully Lynch                   #
# Date:       April 30, 2026                #
# Subject:    Final Course Project          #
# Class:      DSCI 512                      #
# Section:    01W                           #         
# Instructor: David Munoz                   #
# File Name:  DSCI512_Final_Lynch_Sully.R   #
#                                           # 
#############################################


########################
# 1.  Data Preparation #
########################

#     a.  Load the dataset insurance.csv into memory.

insurance <- read.csv("C:/Users/sulli/Downloads/insurance.csv")
View(insurance)

#     b.  In the data frame, transform the variable charges by seting
#         insurance$charges = log(insurance$charges). Do not transform
#         it outside of the data frame.

insurance$charges = log(insurance$charges)
insurance$sex = as.factor(insurance$sex)
insurance$smoker = as.factor(insurance$smoker)
insurance$region = as.factor(insurance$region)

#     c.  Using the data set from 1.b, use the model.matrix() function
#         to create another data set that uses dummy variables in place
#         of categorical variables. Verify that the first column only has
#         ones (1) as values, and then discard the column only after
#         verifying it has only ones as values.

mm <- model.matrix(~ ., data = insurance)
mm <- as.data.frame(mm)
View(mm)
mm$`(Intercept)` <- NULL

#     d.  Use the sample() function with set.seed equal to 1 to generate
#         row indexes for your training and tests sets, with 2/3 of the
#         row indexes for your training set and 1/3 for your test set. Do
#         not use any method other than the sample() function for
#         splitting your data.

set.seed(321)
index <- sample(1:nrow(insurance), (2/3)*nrow(insurance))

#     e.  Create a training and test data set from the data set created in
#         1.b using the training and test row indexes created in 1.d.
#         Unless otherwise stated, only use the training and test
#         data sets created in this step.

train <- insurance[index,]
test <- insurance[-index,]

#     f.  Create a training and test data set from data set created in 1.c
#         using the training and test row indexes created in 1.d

trainmm <- mm[index,]
testmm <- mm[-index,]

#################################################
# 2.  Build a multiple linear regression model. #
#################################################

#     a.  Perform multiple linear regression with charges as the
#         response and the predictors are age, sex, bmi, children,
#         smoker, and region. Print out the results using the
#         summary() function. Use the training data set created in
#         step 1.e to train your model.

lm.result <- lm(charges ~ ., data = train)
summary(lm.result)

#     b.  Is there a relationship between the predictors and the
#         response?

      # Yes. With an adjusted R-squared of 0.7846, there is a significant
      # correlation between the predictors and the response.

#     c.  Does sex have a statistically significant relationship to the
#         response?

      # At the 0.05 significance level, yes. But not any lower.

#     d.  Perform best subset selection using the stepAIC() function
#         from the MASS library, choose best model based on AIC. For
#         the "direction" parameter in the stepAIC() method, set
#         direction="backward"

library(MASS)
full = lm(charges ~ ., data = train)
lm.bwd = stepAIC(full, direction = "backward")
print(lm.bwd)

#     e.  Compute the test error of the best model in #3d based on AIC
#         using LOOCV using trainControl() and train() from the caret
#         library. Report the MSE by squaring the reported RMSE.

library(caret)
train_control = trainControl(method = "LOOCV")
model <- train(charges ~ age + sex + bmi + children + smoker + region, data = train, trControl = train_control, method = "lm")
print(model)
0.426326 ^ 2

#     f.  Calculate the test error of the best model in #3d based on AIC
#         using 10-fold Cross-Validation. Use train and trainControl
#         from the caret library. Refer to model selected in #3d based
#         on AIC. Report the MSE.

train_control = trainControl(method = "CV", number = 10)
model <- train(charges ~ age + sex + bmi + children + smoker + region, data = train, trControl = train_control, method = "lm")
print(model)
0.4231055 ^ 2

#     g.  Calculate and report the test MSE using the best model from 
#         2.d and the test data set from step 1.e.

predictions = predict(lm.bwd, newdata = test)
mean((predictions - test$charges)^2)

#     h.  Compare the test MSE calculated in step 2.f using 10-fold
#         cross-validation with the test MSE calculated in step 2.g.
#         How similar are they?

      # They are similar, but the test MSE calculated in step 2.g is higher,
      # meaning the best model was less precise with the test set than with the
      # training set. Also, 10-fold cross-validation produced the lowest MSE,
      # meaning it produced the best model.

######################################
# 3.  Build a regression tree model. #
######################################

#     a.  Build a regression tree model using function tree(), where
#         charges is the response and the predictors are age, sex, bmi,
#         children, smoker, and region.

library(tree)
tree.insurance = tree(charges ~ ., data = train)
summary(tree.insurance)

#     b.  Find the optimal tree by using cross-validation and display
#         the results in a graphic. Report the best size.

cv.train = cv.tree(tree.insurance)
plot(cv.train$size, cv.train$dev, type = 'b')
# The best size is 3.

#     c.  Justify the number you picked for the optimal tree with
#         regard to the principle of variance-bias trade-off.

      # 3 nodes is the optimal tree size for it produces good enough results
    # while maintaining simplicity and reducing the possibility of overfitting.
      # As you can see in the chart, at size 3 the variance takes a steep drop,
      # but diminishes quickly thereafter. This is the sweet spot in terms of 
      # the variance-bias tradeoff, making it the optimal size for the tree.

#     d.  Prune the tree using the optimal size found in 3.b

prune.train = prune.tree(tree.insurance, best = 3)

#     e.  Plot the best tree model and give labels.

plot(prune.train)
text(prune.train, pretty = 0)

#     f.  Calculate the test MSE for the best model.

yhat = predict(prune.train, newdata = test)
mean((yhat - test$charges)^2)

####################################
# 4.  Build a random forest model. #
####################################

#     a.  Build a random forest model using function randomForest(),
#         where charges is the response and the predictors are age, sex,
#         bmi, children, smoker, and region.

library(randomForest)
rf.train = randomForest(charges ~ ., data = train, importance = TRUE)

#     b.  Compute the test error using the test data set.

yhat.rf = predict(rf.train, newdata = test)
mean((yhat.rf - test$charges)^2)

#     c.  Extract variable importance measure using the importance()
#         function.

importance(rf.train)

#     d.  Plot the variable importance using the function, varImpPlot().
#         Which are the top 3 important predictors in this model?

varImpPlot(rf.train)

# The top 3 important predictors are smoker, age, and children.

############################################
# 5.  Build a support vector machine model #
############################################

#     a.  The response is charges and the predictors are age, sex, bmi,
#         children, smoker, and region. Please use the svm() function
#         with radial kernel and gamma=5 and cost = 50.

library(e1071)
svm.fit = svm(charges ~ ., data = train, kernel = "radial", gamma = 5, cost = 50)
summary(svm.fit)

#     b.  Perform a grid search to find the best model with potential
#         cost: 1, 10, 50, 100 and potential gamma: 1,3 and 5 and
#         potential kernel: "linear","radial" and
#         "sigmoid". And use the training set created in step 1.e.

tune.outlinear = tune(svm, charges ~ ., data = train, kernel = "linear", 
                      ranges = list(cost = c(1, 10, 50, 100), gamma = c(1, 3, 5)))
tune.outradial = tune(svm, charges ~ ., data = train, kernel = "radial",
                      ranges = list(cost = c(1, 10, 50, 100), gamma = c(1, 3, 5)))
tune.outsigmoid = tune(svm, charges ~ ., data = train, kernel = "sigmoid",
                       ranges = list(cost = c(1, 10, 50, 100), gamma = c(1, 3, 5)))

#     c.  Print out the model results. What are the best model
#         parameters?

summary(tune.outlinear)
summary(tune.outradial)
summary(tune.outsigmoid)

#     d.  Forecast charges using the test dataset and the best model
#         found in c).

pred = predict(tune.outradial$best.model, newdata = test)

#     e.  Compute the MSE (Mean Squared Error) on the test data.

mean((pred - test$charges)^2)

#############################################
# 6.  Perform the k-means cluster analysis. #
#############################################

#     a.  Use the training data set created in step 1.f and standardize
#         the inputs using the scale() function.

scaled_trainmm <- scale(trainmm)

#     b.  Convert the standardized inputs to a data frame using the
#         as.data.frame() function.

scaled_trainmm <- as.data.frame(scaled_trainmm)

#     c.  Determine the optimal number of clusters, and use the
#         gap_stat method and set iter.max=20. Justify your answer.
#         It may take longer running time since it uses a large dataset.

library(cluster)
library(factoextra)
set.seed(321)
fviz_nbclust(scaled_trainmm, kmeans, method = "gap_stat")

#     d.  Perform k-means clustering using the optimal number of
#         clusters found in step 6.c. Set parameter nstart = 25

km.res <- kmeans(scaled_trainmm, 3, nstart = 25)

#     e.  Visualize the clusters in different colors, setting parameter
#         geom="point"

fviz_cluster(km.res, data = scaled_trainmm)

######################################
# 7.  Build a neural networks model. #
######################################

#     a.  Using the training data set created in step 1.f, create a 
#         neural network model where the response is charges and the
#         predictors are age, sexmale, bmi, children, smokeryes, 
#         regionnorthwest, regionsoutheast, and regionsouthwest.
#         Please use 1 hidden layer with 1 neuron. Do not scale
#         the data.

library(neuralnet)
nn.model <- neuralnet(charges ~ ., data = trainmm, hidden = 1)

#     b.  Plot the neural network.

plot(nn.model)

#     c.  Forecast the charges in the test dataset.

predict.nn = compute(nn.model, testmm[, 1:8])

#     d.  Compute test error (MSE).

mean((predict.nn$net.result - testmm$charges)^2)

################################
# 8.  Putting it all together. #
################################

#     a.  For predicting insurance charges, your supervisor asks you to
#         choose the best model among the multiple regression,
#         regression tree, random forest, support vector machine, and
#         neural network models. Compare the test MSEs of the models
#         generated in steps 2.g, 3.f, 4.b, 5.e, and 7.d. Display the names
#         for these types of these models, using these labels:
#         "Multiple Linear Regression", "Regression Tree", "Random Forest", 
#         "Support Vector Machine", and "Neural Network" and their
#         corresponding test MSEs in a data.frame. Label the column in your
#         data frame with the labels as "Model.Type", and label the column
#         with the test MSEs as "Test.MSE" and round the data in this
#         column to 4 decimal places. Present the formatted data to your
#         supervisor and recommend which model is best and why.

results.df <- data.frame(
  Model.Type = c("Multiple Linear Regression", "Regression Tree", "Random Forest", 
                 "Support Vector Machine", "Neural Network"),
  Test.MSE = c(round(mean((predictions - test$charges)^2), 4), round(mean((yhat - test$charges)^2), 4),
               round(mean((yhat.rf - test$charges)^2), 4), round(mean((pred - test$charges)^2), 4),
               round(mean((predict.nn$net.result - testmm$charges)^2), 4))
)
print(results.df)

#     b.  Another supervisor from the sales department has requested
#         your help to create a predictive model that his sales
#         representatives can use to explain to clients what the potential
#         costs could be for different kinds of customers, and they need
#         an easy and visual way of explaining it. What model would
#         you recommend, and what are the benefits and disadvantages
#         of your recommended model compared to other models?

      # The model I'd recommend is the regression tree for its simple visuals and
      # and strong results. Although it is not the best model according to MSE,
      # the regression tree would be an effective way of explaining potential costs
      # to clients by showing the tree and asking if they smoke, then ask for their
      # age, and then get a ballpark estimate of what their charge will be. This
      # simple flow from predictor to response gets the point of the model across
      # quickly, while still giving strong predictions.

#     c.  The supervisor from the sales department likes your regression
#         tree model. But she says that the sales people say the numbers
#         in it are way too low and suggests that maybe the numbers
#         on the leaf nodes predicting charges are log transformations
#         of the actual charges. You realize that in step 1.b of this
#         project that you had indeed transformed charges using the log
#         function. And now you realize that you need to reverse the
#         transformation in your final output. The solution you have
#         is to reverse the log transformation of the variables in 
#         the regression tree model you created and redisplay the result.
#         Follow these steps:
#
#         i.   Copy your pruned tree model to a new variable.

copy_pruned_tree = prune.train

#         ii.  In your new variable, find the data.frame named
#              "frame" and reverse the log transformation on the
#              data.frame column yval using the exp() function.
#              (If the copy of your pruned tree model is named 
#              copy_of_my_pruned_tree, then the data frame is
#              accessed as copy_of_my_pruned_tree$frame, and it
#              works just like a normal data frame.).

copy_pruned_tree$frame$yval <- exp(copy_pruned_tree$frame$yval)

#         iii. After you reverse the log transform on the yval
#              column, then replot the tree with labels.

plot(copy_pruned_tree)
text(copy_pruned_tree, pretty = 0)
