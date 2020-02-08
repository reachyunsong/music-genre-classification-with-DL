# Music Genres Classification

A Parallel CNN-RNN model for Music Genres Classification



## Files Included

`best_model_prediction.py`

> The code for running the model to perdicte the labels for test data

`confusion_matrix.eps`

> The confusion matrix for the final result of test data perdiction

`data`

> A folder included training, validation and test data/numpy array for training and evaluation

`requirements.txt`

> the dependency install

`weights.best.h5`

> our best model

`training_log`

> contain four `.ipynb` files: *data_pre_img*, *data_pre_label*, *model_Adam* and *model_RMSProp*

`team5_report.pdf`

> our report



## Get Test Result

##### load the best model, and use test data to perdict the labels

Use `requirements.txt` to install dependency

```
pip install -r requirements.txt
```

run `best_model_prediction.py` inside our folder

```
python best_model_prediction.py
```

After run the `best_model_prediction.py`, it will print out the `Sample prediction(head 20)`, `Classification Report`, `Test Accuracy` and get the `confusion_matrix.eps` in the folder. 





