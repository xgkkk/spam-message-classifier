# Spam message classifier 

## Requirement

- numpy
- sklearn
- jieba

To install the required libraries, you can run the following command:

```
pip3 install numpy sklearn jieba
```

## Note
- `spam_email_classifier.py` can train the models without any necessary hardware requirement.
- `spam_email_classifier_multicores.py` will parallelize the training process with multiple cores. So it requires that your computer must have multiple cores.
- `带标签短信.txt` is the data file. The first column of it is the labels (0 is not spam, 1 is spam) and the second column is the messages.

