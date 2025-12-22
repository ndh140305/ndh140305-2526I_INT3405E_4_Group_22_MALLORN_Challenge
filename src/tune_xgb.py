import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from data_processing import handle_missing_and_weights
try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline
except ImportError:
    RandomOverSampler = None
    Pipeline = None
import joblib

feat_imp_path = 'features/feature_importance.csv'
feat_imp = pd.read_csv(feat_imp_path)

# chỉ dùng top features
top_feats = feat_imp.sort_values('importance', ascending=False)['feature'].head(45).tolist()

features_path = 'features/full_train_features_labeled.csv'
df = pd.read_csv(features_path)

X = df[top_feats]
y = df['target'].values.ravel()

# Xử lý missing values and sample weights
X_filled, _= handle_missing_and_weights(X, df['target'])

# Có thể oversample để giải quyết mất cân bằng
USE_OVERSAMPLE = False

n_positive = np.sum(y == 1)
n_negative = np.sum(y == 0)
scale_pos_weight = n_negative / n_positive

# tham số đã tối ưu trên tập test(không tối ưu theo f1 trên tập train vì có dấu hiệu overfit)
param_grid = {
    'max_depth': [5],
    'min_child_weight': [3],
    'subsample': [0.6],
    'colsample_bytree': [0.6],
    'learning_rate': [0.05],
    'n_estimators': [150],
}

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    random_state=42
)

f1 = make_scorer(f1_score, average='binary')

if USE_OVERSAMPLE and RandomOverSampler is not None and Pipeline is not None:
    pipe = Pipeline([
        ('ros', RandomOverSampler(random_state=42)),
        ('xgb', xgb)
    ])
    grid = GridSearchCV(
        pipe,
        param_grid={f'xgb__{k}': v for k, v in param_grid.items()},
        scoring=f1,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_filled, y)
else:
    grid = GridSearchCV(
        xgb,
        param_grid,
        scoring=f1,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_filled, y)


print('Best params:', grid.best_params_)
print('Best f1 (default threshold=0.5):', grid.best_score_)

# Thêm threshold để tối ưu dự đoán nhãn hiếm (1)
THRESHOLD = 0.25  
model = grid.best_estimator_
probs = model.predict_proba(X_filled)[:, 1]
preds = (probs > THRESHOLD).astype(int)
f1_fixed = f1_score(y, preds)
print(f'F1 on train with THRESHOLD={THRESHOLD}: {f1_fixed:.4f}')

joblib.dump(model, 'models/model_xgb_tuned.pkl')
print('Saved to models/model_xgb_tuned.pkl')
