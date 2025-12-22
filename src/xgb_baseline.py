import xgboost as xgb

def get_xgb_classifier(num_class=None):
    params = {
        'n_estimators': 400,
        'max_depth': 7,
        'learning_rate': 0.07,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',  
    }
    if num_class is None or num_class == 2:
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
        return xgb.XGBClassifier(**params)
    else:
        params['objective'] = 'multi:softprob'
        params['eval_metric'] = 'mlogloss'
        params['num_class'] = num_class
        return xgb.XGBClassifier(**params)
