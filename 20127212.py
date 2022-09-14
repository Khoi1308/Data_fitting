import numpy as np
import pandas as pd

df = pd.read_csv("D:/wine.csv", sep=';')

headers = list(df.columns.values)
#Translate datastructure
data = np.array(df)
#print model 
def print_y(theta_hat):
  print("Model: ")
  print("y =", end=" ")
  for i in range(len(theta_hat)):
    print(f"+ {theta_hat[i]}.x{i+1}", end = " ")


def theta(data):
    X = data[:,:-1]
    Y = data[:,-1:].reshape(len(data),)
    return np.linalg.inv(X.T @ X) @ X.T @ Y
  
print("a) ", "\n")
theta_hat = theta(data)
print_y(theta_hat)
print("\n")

print("b) ", "\n")

#Calculator r base on Cross Validation method
def calculator_R(train, test):
    theta_hat = theta(test)
    r = 0
    
    for i in range(len(train)):
        temp = 0
        for j in range(len(theta_hat)):
            temp += theta_hat[j] * train[i][j]
        r += (temp - train[i][-1])**2
    res=np.sqrt(r)
    return res
  
#Average r of one dataset base on Cross Validation method
def avg_r(data):
    n = int(np.round(len(data) / 4))
    models = [data[i*n:i*n+n, :] for i in range(4-1)]
    models.append(data[(4-1)*n:, :])

    residuals = []
    for i in range(4):
        train = np.concatenate(models[:-1], axis=0)
        test = models[-1]
        residuals.append(calculator_R(train, test))
        
        models[-1], models[i] = models[i], models[-1]
    
    return np.average(residuals), residuals
#Average r of each features in database
def feature_r(data):
    avg = []
    r = []
    
    for i in range(len(data[0]) - 1):
        feature = np.concatenate((data[:, i:i+1], data[:, -1:]), axis=1)
        feature_avg, feature_residuals = avg_r(feature)
        avg.append(feature_avg)
        r.append(feature_residuals)
    
    return avg, r
  
#Ranking features
def Ranking(data, headers):
    avg_r, r = feature_r(data)
    indices = np.argsort(avg_r)
    
    ranking_data = []
    for i in range(len(indices)):
        tables = []
        tables.append(headers[indices[i]])
        tables.append(r[indices[i]][0])
        tables.append(r[indices[i]][1])
        tables.append(r[indices[i]][2])
        tables.append(r[indices[i]][3])
        tables.append(avg_r[indices[i]])
        tables.append(i + 1)
        
        ranking_data.append(tables)
        
    return pd.DataFrame(ranking_data, columns=["Features", "r1", "r2", "r3", "r4", "r4", "Rank"])

# Filter the best feature 
def Filter(data):
    r, _ = feature_r(data)
    best_index = np.argsort(r)[0]
    
    new_data = np.concatenate((data[:, best_index:best_index+1], data[:,-1:]), axis=1)
    
    return new_data


print(Ranking(data, headers))

new_data = Filter(data)
theta_x = theta(new_data)
print_y(theta_x)