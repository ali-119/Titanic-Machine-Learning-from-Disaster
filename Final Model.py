import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer


def get_common_features(df):
    
    # Cleaning
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    df['FareGroup'] = pd.cut(df['Fare'], bins=[-1, 7.91, 14.45, 31, 500], 
                             labels=['Low', 'Medium', 'High', 'Very High'])
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    return df


def titanic_data():
    data = pd.read_csv(r"F:\download\File\ML\titanic\train.csv")
    
    # Cleaning
    data = data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)
    data['Embarked'] = data["Embarked"].fillna('S')
    data['Gender'] = data['Sex']
    data = data.drop('Sex', axis=1)

    data = get_common_features(data)
    return data


def titanic_test_data():
    test_data = pd.read_csv(r"F:\download\File\ML\titanic\test.csv")
    
    # Cleaning
    test_data = test_data.drop(['Cabin', 'Ticket', 'Name'], axis=1)
    test_data['Embarked'] = test_data["Embarked"].fillna('S')
    test_data['Gender'] = test_data['Sex']
    test_data = test_data.drop('Sex', axis=1)
    
    test_data = get_common_features(test_data)
    return test_data



def titanic_gender_submission():
    
    gender_submission_data = pd.read_csv(r"F:\download\File\ML\titanic\gender_submission.csv")
    
    return gender_submission_data

titanic_data()
titanic_test_data()
titanic_gender_submission()


# Importing and selecting training data columns
data = titanic_data()

X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
       'Gender', 'FareGroup', 'AgeGroup', 'FamilySize', 'FarePerPerson', 'IsAlone']]
y = data['Survived']


# Column Classification
numeric_features = [
    'Pclass',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'FamilySize',
    'FarePerPerson',
    'IsAlone'
]

categorical_features = [
    'Embarked',
    'Gender',
    'FareGroup',
    'AgeGroup'
]


# Scale and Encode
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_features),
    ('categorical', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    ), categorical_features)
])


# Making pipeline of Model
pipe_line = Pipeline([
    ('preprocessor', preprocessor),
    ('decisiontree', DecisionTreeClassifier(criterion='entropy', splitter='best',
                                            max_depth=10, min_samples_split=2,
                                            min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                            min_impurity_decrease=0.0, ccp_alpha=0.01))
    ], verbose=True)
pipe_line.fit(X, y)


# Importing and selecting testing and gender submission data columns
test_data = titanic_test_data()
gender_submission_data = titanic_gender_submission()

X_test = test_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
       'Gender', 'FareGroup', 'AgeGroup', 'FamilySize', 'FarePerPerson', 'IsAlone']]
y_test = gender_submission_data['Survived']


# Results
test_accuracy = pipe_line.score(X_test, y_test)
pred = pipe_line.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, pred), display_labels=pipe_line.classes_).plot()
plt.show()
