from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy="constant")

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
# Bundle preprocessing for numerical and categorical data
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('numerical', numerical_transformer, numerical_cols),
#         ('categorical', categorical_transformer, categorical_cols)
#     ])


class Preprocessor:
    def __init__(self, numerical_cols, categorical_cols):
        self.pipeline = ColumnTransformer(transformers=[
            ('numerical', numerical_transformer, numerical_cols),
            ('categorical', categorical_transformer, categorical_cols)]
        )

    def __getattr__(self, item):
        self.preprocessor.__getattribute__(item)

    def __repr__(self):
        return f"Preprocessor with several steps" # TODO WHAT


class ModelPipeline(Preprocessor):
    def __init__(self, numerocal_cols, categorical_cols, model, numerical_cols, **kwargs):
        super().__init__(numerical_cols, categorical_cols)
        self.pipeline = Pipeline(steps=[
            ("preprocessor", super(ModelPipeline, self).pipeline),
            ("model", model(**kwargs))
        ])