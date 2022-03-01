# Ailerons-of-the-aircraft
The goal is to predict the control action on the ailerons of the aircraft.
we will use optimal feature selection on the Ailerons dataset, which addresses a control problem, namely flying a F16 aircraft. The attributes describe the status of the aeroplane, while the goal is to predict the control action on the ailerons of the aircraft.

The model selected a sparsity of 9 as the best parameter, but we observe that the validation scores are close for many of the parameters. We can use the results of the grid search to explore the tradeoff between the complexity of the regression and the quality of predictions:

We see that the quality of the model quickly increases with additional terms until we reach 4, and then only has small increases afterwards. Depending on the application, we might decide to choose a lower sparsity for the final model than the value chosen by the grid search.

We can evaluate the quality of the model using score with any of the supported loss functions. For example, the R^2
  on the training set
Proper EDA and Statistical analysis of data followed by Developing a model, to predict the column “Goal”.
Loss function is “RMSE”
Use of proper feature selection process and hyperparameter tuning 
