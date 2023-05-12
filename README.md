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
