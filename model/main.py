import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 2
def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # Scale Data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    
    # train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # test model
    y_pred = model.predict(X_test)
    print("Accuracy Of Our Model: ", accuracy_score(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))
    
    return model, scaler

# 1
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    print(data.head())
    
    data = data.drop(['Unnamed: 32', 'id'], axis=1) 
    print(data.head())
    
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    print(data.head())
    
    return data

def main():
    data = get_clean_data()
    
    # train model and test model
    model, scaler = create_model(data)
    
    
#3
    with open('model/model.pkl','wb') as f:
        pickle.dump(model,f)
        
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)

if __name__ == '__main__':
    main()
