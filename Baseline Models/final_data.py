import pandas as pd
from sklearn.impute import SimpleImputer

def get_common_features(df):

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    df['FareGroup'] = pd.cut(df['Fare'], bins=[-1, 7.91, 14.45, 31, 600], 
                             labels=['Low', 'Medium', 'High', 'Very High'])
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    return df


def titanic_data():
    data = pd.read_csv(r"F:\download\File\ML\titanic\train.csv")
    
    # Cleaning
    data = data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)
    data['Age'] = SimpleImputer(strategy='mean').fit_transform(data[['Age']])
    data['Embarked'] = data["Embarked"].fillna('S')
    data['Gender'] = data['Sex']
    data = data.drop('Sex', axis=1)

    data = data[(data['SibSp'] != 8) & (data['SibSp'] != 5)]
    data = data[(data['Parch'] != 5) & (data['Parch'] != 3) & (data['Parch'] != 4) & (data['Parch'] != 6)]
    
    
    data = get_common_features(data)
    return data


def titanic_test_data():
    test_data = pd.read_csv(r"F:\download\File\ML\titanic\test.csv")
    
    # Cleaning
    test_data = test_data.drop(['Cabin', 'Ticket', 'Name'], axis=1)
    test_data['Embarked'] = test_data["Embarked"].fillna('S')
    test_data['Fare'] = test_data["Fare"].fillna(test_data["Fare"].mean())
    test_data['Age'] = SimpleImputer(strategy='mean').fit_transform(test_data[['Age']])
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
