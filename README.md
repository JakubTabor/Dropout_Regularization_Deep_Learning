# Dropout_Regularization_Deep_Learning
# I import "warnings" to "ignore" the  "warnings" and load my "df" without "header"
# I check the shape of my "df" "df.shape", I also check if there is any "non value column" "df.isna().sum()"
# I chcek balance of my "target column" """df[60].value_counts()""" and my "y" will be my "target column"  "y = df[60]" and "X" the rest
# Now I need to change my target column into numerical values, so I use method "get_dummies" """y = pd.get_dummies(y, drop_first=True)"""
# Next i can import "train_test_split" and get "train" and "test" set and check "shape" "X_train.shape, X_test.shape"
# Then i import important libraries "tensorflow" and "keras" and create my "model" using "keras.Sequential"
# The shape of my layer is "input_dim=60" and first layer gonna have (60) neurons and standard "activation" "relu"
# Then I have 2 hidden layers, and output layer with "activation='sigmoid'" 
# I "compile" my model with "loss" as "binary_crossentropy", "optimizer" as "adam" and "metrics" "accuracy"
# Next i "fit" my model with "X_train and y_train", set "epochs" at 100 and "batch_size" at 8 so my "gradient descent"
# Then I "evaluate" my model with "X_test and y_test" and I prepare "y_pred" "model.predict(X_test).reshape(-1)" with reshape to 1 dimension
# I "round" my "y_pred" "np.round(y_pred)", "print" first 10 predictions and compare with my "y_test"
# Next I import "confusion_matrix" and "classification_report" and I print it "print(classification_report(y_test, y_pred))"
# Now I will test my model with "Dropout" method, so after every layer it will drop half of neurons to prevent "overtraining"
# My first layer have shape "input_dim=60" 60 neurons and "ctivation relu", then i drop half of neurons "keras.layers.Dropout(0.5)"
