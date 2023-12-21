
#Allstate Severity Claims Final Project

#--------------------------------Packages------------------
library(recipes)
library(tidyverse)
library(vroom)
library(DataExplorer)
library(GGally)
library(rpart)
library(ranger)
library(lubridate)
library(poissonreg)
library(rsem)
library(ggplot2)
library(modeltime)
library(vroom)
library(embed)
library(kknn)
library(kernlab)
library(rpart)
library(ranger)
library(tidymodels)
library(skimr)
library(stacks)

#---------------------------Data Load In----------------
AllTrain <- vroom("C:/Users/isaac/Documents/Fall 2023 Real/train.csv")
AllTest <- vroom("C:/Users/isaac/Documents/Fall 2023 Real/test.csv")

train <- AllTrain[sample(nrow(AllTrain), 500), ]
test  <- AllTest[sample(nrow(AllTest), 100), ]

#-------------------------Collinearity Visuals-------------
skimr::skim(train)
### EDA #### 
hist(train$loss)
### Some variables are highly correlated to each other 
plot_correlation(train, type = 'continuous')


###------------------------MY_RECIPE and LOG-----------------------###
train$loss <- log(train$loss)


my_recipe <- recipe(loss ~ ., train) %>% 
  update_role(id, new_role = 'ID') %>%
  step_scale(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = .6) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  prep()

prep <- prep(recipe)
baked <- bake(prep, new_data = train)










### ---------------------------Initial_Split Boost Tree--------------- ####


## Model
boost <- boost_tree(mode = 'regression', 
                    learn_rate = tune(),
                    loss_reduction = tune(),
                    tree_depth = tune(),
                    min_n = tune()
) %>%
  set_engine('xgboost', objective = 'reg:absoluteerror')

## Workflow
boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost)


## Tuning Grid
tuning_grid <- grid_regular(learn_rate(),
                            loss_reduction(),
                            tree_depth(),
                            min_n(),
                            levels = 2)

## Set up K-fold CV
folds <- vfold_cv(train, v = 3, repeats=2)


## Find Parameters
boost_fit <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse))

bestTune <- boost_fit %>%
  select_best("rmse")


## Finalize workflow and predict
final_boost_wf <-
  boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_boost_pred <- predict(final_boost_wf, new_data = AllTest)

## Format Data
My_Boost_Pred <- data.frame(AllTest$id, final_boost_pred)

colnames(My_Boost_Pred) <- c("id", "loss")

My_Boost_Pred$loss <- exp(My_Boost_Pred$loss)

vroom_write(My_Boost_Pred, file="AllStateBoost.csv", delim=",")


########################### WIth Fixed Parameters ######################
AllTrain <- vroom("C:/Users/isaac/Documents/Fall 2023 Real/train.csv")
AllTest <- vroom("C:/Users/isaac/Documents/Fall 2023 Real/test.csv")

BigTrain <- AllTrain

#BigTrain <- AllTrain[sample(nrow(AllTrain), 1000), ]

BigTrain$loss <- log(BigTrain$loss)

my_all_recipe <- recipe(loss ~ ., BigTrain) %>% 
  update_role(id, new_role = 'ID') %>%
  step_scale(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = .6) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  prep()

pre_boost <- boost_tree(mode = 'regression', 
                    learn_rate = 0.1,
                    loss_reduction = 1e-10,
                    tree_depth = 15,
                    min_n = 40
) %>%
  set_engine('xgboost', objective = 'reg:absoluteerror')

pre_boost_wf <- workflow() %>%
  add_recipe(my_all_recipe) %>%
  add_model(pre_boost)

prefit_boost_wf <- pre_boost_wf %>%
  fit(data=BigTrain)

Preboost_Pred <- predict(prefit_boost_wf, 
                         new_data=AllTest)

Preboost_Pred <- data.frame(AllTest$id, Preboost_Pred)

colnames(Preboost_Pred) <- c("id", "loss")

Preboost_Pred$loss <- exp(Preboost_Pred$loss)

vroom_write(Preboost_Pred, file="AllStatePreBoost.csv", delim=",")







###------------Stacked Models-------------###























###-------------------------------Decision Trees------------###

## My Recipe
# my_recipe <- recipe(loss ~ ., data = AllTrain) %>%
#   update_role(id, new_role = 'ID') %>%
#   step_scale(all_numeric_predictors()) %>%
#   step_corr(all_numeric_predictors(), threshold = 0.6) %>%
#   step_novel(all_nominal_predictors()) %>%
#   step_unknown(all_nominal_predictors()) %>%
#   step_dummy(all_nominal_predictors()) %>%
#   step_pca(all_predictors(), threshold = 0.87, num_comp = 30)

# prep_recipe <- prep(recipe, training = AllTest)
# 
# baked_test <- bake(prep_recipe, new_data = AllTest)

## Tree Model
tree_mod <- decision_tree(tree_depth = 25,
                          cost_complexity = tune(),
                          min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use
  set_mode("regression")

## Workflow
tree_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(tree_mod)

## Set up grid of tuning value
tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 3)

## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=3)


## Find best tuning parameters
CV_results <- tree_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse))
            
bestTune <- CV_results %>%
  select_best("rmse")


## Finalize workflow and predict
final_tree_wf <-
  tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_tree_pred <- predict(final_tree_wf, new_data = AllTest)


## Format Data
My_Tree_Pred <- data.frame(AllTest$id, final_tree_pred)

colnames(My_Tree_Pred) <- c("id", "loss")

vroom_write(My_Tree_Pred, file="AllStateBoost.csv", delim=",")













#----------------------LINEAR REGRESSION
linear_mod <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# Create a workflow with model & recipe
linear_wf <- workflow() %>%
  add_recipe(my_all_recipe) %>%
  add_model(linear_mod)

# Finalize workflow and predict
finals_wf <- linear_wf %>%
  fit(data = BigTrain)

Lin_Pred <- predict(finals_wf, 
                         new_data=AllTest)

Lin_Pred <- data.frame(AllTest$id, Lin_Pred)

colnames(Lin_Pred) <- c("id", "loss")

Lin_Pred$loss <- exp(Lin_Pred$loss)

vroom_write(Lin_Pred, file="AllStateLin.csv", delim=",")






##############-------Stacked Models------####################
## Create a control grid
untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a model

linreg_fit <- linear_wf %>%
  fit_resamples(resamples=folds,
                control=tunedModel)

boosted_fit <- pre_boost_wf %>%
  fit_resamples(resamples=folds,
                control=tunedModel)



my_stack <- stacks() %>%
  add_candidates(linreg_fit) %>%
  add_candidates(boosted_fit)

stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset

stack_final <- predict(stack_mod,new_data=AllTest)

Stack_Pred <- data.frame(AllTest$id, stack_final)

colnames(Stack_Pred) <- c("id", "loss")

Stack_Pred$loss <- exp(Stack_Pred$loss)

vroom_write(Stack_Pred, file="AllStateStack.csv", delim=",")








#---------------------------------Penalized Regression--------------
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV19
folds <- vfold_cv(train, v = 2, repeats=2)

## Run the CV
CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse))

bestTune <- CV_results %>%
  select_best("rmse")

## Finalize the Workflow & fit it
final_wf <-
  preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

## Predict
final_pred <- predict(finals_wf, new_data = AllTest)

final_pred[final_pred<0]=0
My_Pred <- data.frame(AllTest$id, final_pred)
colnames(My_Pred) <- c("id", "loss")

vroom_write(My_Pred, file="AllState.csv", delim=",")



