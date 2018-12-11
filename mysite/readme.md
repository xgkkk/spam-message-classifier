# Online demo system of the spam message classifier

## Requirement

- django 2.1.3
- jieba

To install the required libraries, please run the following command:

```
pip3 install django==2.1.3 jieba
```

## Note
To verify it's working, run the following command:

```
cd mysite/
python3 manage.py runserver
```

- xx.py can train the models in any computer.
- xx.py will parallelize the training process with multiple cores. So it requires your computer must have multiple cores.
