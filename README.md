# PLAN-BERT-TKDE
The source code and data to reproduce the reported results in PLAN-BERT's TKDE paper.

# File Descriptions
|               Files              | Description                                                                                                                                                                                            |
|:--------------------------------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| datasets                         | The path contains the preprocessing code, processed data, and the like of the origin of these data, including Check-in Tokyo, Amazon Beauty, Taobao, and Uk-retail 2009 datasets. Need to be unzipped. |
| datasets/*/preprocessing.ipynb   | The code the to convert the original dataset of .csv format into pickle dictionary.                                                                                                                    |
| */PLANBERT.ipynb                 | Train the complete PLAN-BERT model. Test it with both time information mode and wishlist mode.                                                                                                         |
| */BiLSTM.ipynb                   | Train and test the BiLSTM model.                                                                                                                                                                       |
| */KNN.ipynb                      | Employ UserKNN to provide recommendation.                                                                                                                                                              |
| */PLANBERT-auto-regressive.ipynb | Train a uni-directional Transformer to provide recommendation without future reference items.                                                                                                          |
| */LSTM.ipynb                     | Train a uni-directional LSTM to provide recommendation without future reference items.                                                                                                                 |
| checkpoint                       | Path to save trained models and recall of different h and r.                                                                                                                                           |
