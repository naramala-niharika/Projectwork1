## Title of the Project
HVAC-Occupancy-Detection-with-ML-and-DL-Methods

## About
<!--Detailed Description about the project-->
For this kernel, I'm using a dataset for predicting room occupancy with using environmental observations such as temperature, humidity and CO2 level. This predictions might help Heating, Ventilating and Air Conditioning (HVAC) sector. For instance, we are using sensors like thermostats to get information about the environment and with that info our system decides to heat or not situation. But if the thermostat set manually by a occupant before and there is no more occupants in the environment, what then? The system won't shutdown until it gets set values, and this situation will lead high energy consumption.
## Features
<!--List the features of the project as shown below-->
The project on HVAC occupancy detection using AI, ML, and DL techniques includes several notable features aimed at improving accuracy, efficiency, and adaptability. 
Here are some of its key features:
- Enhanced Occupancy Detection.
- Energy Management Optimization.
- Predictive Maintenance Capabilities.
- Flexible and Adaptive Models.
- Computational Efficiency.
- Real-Time Data Processing and Adaptive Control.
- Sustainable and Comfortable Built Environments.
- Scalability for Smart Building Technology

## Requirements
<!--List the requirements of the project as shown below-->
* To implement a project on HVAC occupancy detection using AI, ML, and DL techniques, certain requirements must be met across hardware, software, data, and operational needs. Here’s an outline of the key requirements:

### 1. **Hardware Requirements**
   - **HVAC System**: A compatible HVAC system with programmable control and connectivity options.
   - **Sensors**:
     - **Occupancy Sensors**: Motion, CO2, and temperature sensors for occupancy and environment monitoring.
     - **Environmental Sensors**: Temperature, humidity, and air quality sensors to provide data for model training and real-time control.
   - **Computing Hardware**:
     - **Edge Device or Local Server**: For real-time processing (Raspberry Pi, NVIDIA Jetson, or similar) to handle ML/DL computations near the source.
     - **High-Performance Server**: For model training and storage, especially when working with DL models that require significant processing power.
   - **Networking Components**: Routers and connectivity modules (e.g., Wi-Fi, Ethernet) to facilitate data communication between sensors, controllers, and servers.

### 2. **Software Requirements**
   - **Operating System**: Linux (Ubuntu, CentOS) or Windows, depending on compatibility with deployed devices and processing units.
   - **Programming Languages**: Primarily Python for ML/DL model development, with possible support for JavaScript (for IoT interface) or C++ for edge computing optimizations.
   - **Machine Learning and Deep Learning Libraries**:
     - **TensorFlow/PyTorch**: For model training and neural network design.
     - **Scikit-learn**: For preprocessing and simpler machine learning algorithms.
     - **OpenCV**: For image or motion analysis, if visual occupancy detection is integrated.
   - **Data Processing and Database Tools**:
     - **SQL/NoSQL Databases** (e.g., PostgreSQL, MongoDB) for storing and retrieving historical and real-time data.
     - **Data Preprocessing Libraries**: Pandas, Numpy, for data manipulation.
   - **Communication Protocols**: MQTT, HTTP, or other protocols for data transfer and communication with edge devices.

### 3. **Data Requirements**
   - **Historical Data**: Past data on occupancy patterns, environmental factors, and HVAC usage to train predictive models. Data can be sourced from real sensors or simulated in early stages.
   - **Real-Time Data**: Data feeds from occupancy and environmental sensors for real-time adjustments.
   - **Labeled Data for Model Training**: For supervised learning tasks, a labeled dataset with occupancy data correlated with sensor inputs (e.g., temperature, CO2 levels) to develop accurate models.
   - **Maintenance Data**: Historical data on HVAC component performance, repairs, and faults for predictive maintenance modeling.

### 4. **Algorithm and Model Requirements**
   - **Occupancy Detection Models**:
     - **Supervised Learning Models**: Decision trees, SVMs, or neural networks trained on labeled occupancy data.
     - **Deep Learning Models**: CNNs or RNNs if image/video data or time-series predictions are used.
   - **Anomaly Detection and Predictive Maintenance Models**:
     - **Unsupervised/Semi-Supervised Learning Models**: Clustering techniques (e.g., K-means) or autoencoders for anomaly detection in sensor readings.
     - **Predictive Models**: Algorithms like LSTM or reinforcement learning models for predicting equipment failure or maintenance needs.
   - **Energy Optimization Models**:
     - **Reinforcement Learning Algorithms**: For adaptive control of HVAC settings based on real-time occupancy and environmental conditions.

### 5. **Functional Requirements**
   - **Sensor Data Integration**: Ability to collect and process real-time data from multiple sensors.
   - **Model Training Pipeline**: End-to-end pipeline for model training, including data preprocessing, feature engineering, and model validation.
   - **Real-Time Decision Making**: Systems for real-time HVAC adjustments based on occupancy and environmental data.
   - **Predictive Maintenance Notifications**: System to notify when maintenance is due based on predictive analysis.
   - **User Interface for Monitoring**: Dashboard for users to monitor occupancy, energy consumption, and maintenance needs, with options to adjust HVAC settings.

### 6. **Security and Privacy Requirements**
   - **Data Security**: Encryption of data in transit and at rest to protect sensitive building information and user privacy.
   - **Access Control**: Role-based access to ensure only authorized personnel can access system controls and data.
   - **Compliance**: Adherence to privacy standards and regulations (e.g., GDPR, CCPA) if personal or identifiable data is collected.

### 7. **Testing and Validation Requirements**
   - **Model Validation**: Cross-validation, accuracy measurement, and robustness testing for the models across different building types and occupancy patterns.
   - **Performance Testing**: Evaluation of system response times and real-time data processing under various load conditions.
   - **Hardware Compatibility Testing**: Ensuring that sensor data is reliably transmitted to and from the HVAC system.

### 8. **Deployment and Maintenance Requirements**
   - **Edge or Cloud Deployment**: Infrastructure for deploying models on local edge devices or cloud servers depending on computational needs.
   - **Maintenance of Models**: Continuous monitoring and retraining of ML models to adapt to evolving occupancy patterns or environmental changes.
   - **System Monitoring and Updates**: Regular software and hardware updates to ensure reliability and security.

These requirements are essential for developing an efficient and scalable HVAC occupancy detection system, supporting real-time operation, predictive maintenance, and adaptive energy management for smart buildings.

#program
```
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Importing necessary libraries for this notebook.
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2, l1
from keras.metrics import BinaryAccuracy
/kaggle/input/occupancy-detection-data-set-uci/datatraining.txt
/kaggle/input/occupancy-detection-data-set-uci/datatest.txt
/kaggle/input/occupancy-detection-data-set-uci/datatest2.txt


Using TensorFlow backend.
datatest = pd.read_csv("/kaggle/input/occupancy-detection-data-set-uci/datatest.txt")
datatest2 = pd.read_csv("/kaggle/input/occupancy-detection-data-set-uci/datatest2.txt")
datatraining = pd.read_csv("/kaggle/input/occupancy-detection-data-set-uci/datatraining.txt")
```
# Exploratory Data Analysis¶
```
print(datatest.info())
datatest.head()


print(datatest2.info())
datatest2.head()



print(datatraining.info())
datatraining.head()




datatest['date'] = pd.to_datetime(datatest['date'])
datatest2['date'] = pd.to_datetime(datatest2['date'])
datatraining['date'] = pd.to_datetime(datatraining['date'])
datatest.reset_index(drop=True, inplace=True)
datatest2.reset_index(drop=True, inplace=True)
datatraining.reset_index(drop=True, inplace=True)




datatraining.describe()


scaler = MinMaxScaler()
columns = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
scaler.fit(np.array(datatraining[columns]))
datatest[columns] = scaler.transform(np.array(datatest[columns]))
datatest2[columns] = scaler.transform(np.array(datatest2[columns]))
datatraining[columns] = scaler.transform(np.array(datatraining[columns]))


plt.figure(figsize=(10,10))
plt.title('Box Plot for Features', fontdict={'fontsize':18})
ax = sns.boxplot(data=datatraining.drop(['date', 'Occupancy'],axis=1), orient="h", palette="Set2")
print(datatraining.drop(['date', 'Occupancy'],axis=1).describe())


plt.figure(figsize=(10,8))
plt.title('Correlation Table for Features', fontdict={'fontsize':18})
ax = sns.heatmap(datatraining.corr(), annot=True, linewidths=.2)


data = datatraining.copy()
data.Occupancy = data.Occupancy.astype(str)
fig = px.scatter_3d(data, x='Temperature', y='Humidity', z='CO2', size='Light', color='Occupancy', color_discrete_map={'1':'red', '0':'blue'})
fig.update_layout(scene_zaxis_type="log", title={'text': "Features and Occupancy",
                                                'y':0.9,
                                                'x':0.5,
                                                'xanchor': 'center',
                                                'yanchor': 'top'})
iplot(fig)



sns.set(style="darkgrid")
plt.title("Occupancy Distribution", fontdict={'fontsize':18})
ax = sns.countplot(x="Occupancy", data=datatraining)


hours_1 = []
hours_0 = []
for date in datatraining[datatraining['Occupancy'] == 1]['date']:
    hours_1.append(date.hour)
for date in datatraining[datatraining['Occupancy'] == 0]['date']:
    hours_0.append(date.hour)



plt.figure(figsize=(8,8))
ax = sns.distplot(hours_1)
ax = sns.distplot(hours_0)




datatest['period_of_day'] = [1 if (i.hour >= 7 and i.hour <= 17) else 0 for i in datatest['date']]
datatest2['period_of_day'] = [1 if (i.hour >= 7 and i.hour <= 17) else 0 for i in datatest2['date']]
datatraining['period_of_day'] = [1 if (i.hour >= 7 and i.hour <= 17) else 0 for i in datatraining['date']]
datatraining.sample(10)
```

# Classification with Machine Learning Methods
```
X_train = datatraining.drop(columns=['date', 'Occupancy'], axis=1)
y_train = datatraining['Occupancy']
X_validation = datatest.drop(columns=['date', 'Occupancy'], axis=1)
y_validation = datatest['Occupancy']
X_test = datatest2.drop(columns=['date', 'Occupancy'], axis=1)
y_test = datatest2['Occupancy']
```
# KNN (K-Nearest Neighbors)
```
# parameter-tuning for knn
n_neighbors_list = [7,15,45,135]
weights_list = ['uniform', 'distance']
metric_list = ['euclidean', 'manhattan']
accuracies = {}
for n in n_neighbors_list:
    for weight in weights_list:
        for metric in metric_list:
            knn_model = KNeighborsClassifier(n_neighbors=n, weights=weight, metric=metric)
            knn_model.fit(X_train, y_train)
            accuracy = knn_model.score(X_validation, y_validation)
            accuracies[str(n)+"/"+weight+"/"+metric] = accuracy


plotdata = pd.DataFrame()
plotdata['Parameters'] = accuracies.keys()
plotdata['Accuracy'] = accuracies.values()
fig = px.line(plotdata, x="Parameters", y="Accuracy")
fig.update_layout(title={'text': "Accuracies for Different Hyper-Parameters",
                                                'x':0.5,
                                                'xanchor': 'center',
                                                'yanchor': 'top'})
iplot(fig)



knn_model = KNeighborsClassifier(n_neighbors=135)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_validation)
plt.title("KNN Confusion Matrix for Validation Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_validation, y_pred), annot=True, fmt="d")
```
#SVM (Support-Vector Machine)
```

svm_model = SVC()
svm_model.fit(X_train, y_train)
print("Accuracy for SVM on validation data: {}%".format(round((svm_model.score(X_validation, y_validation)*100),2)))

y_pred = svm_model.predict(X_validation)
plt.title("SVM Confusion Matrix for Validation Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_validation, y_pred), annot=True, fmt="d")

```
#Classification with Neural Networks
```
# NN without regularization
model1 = Sequential()
model1.add(Dense(32, activation='relu', input_dim=6))
model1.add(Dense(16, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history1 = model1.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation))




# NN with 0.2 dropout ratio before the hidden layer.
model2 = Sequential()
model2.add(Dense(32, activation='relu', input_dim=6))
model2.add(Dropout(0.2))
model2.add(Dense(16, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history2 = model2.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation))




# NN with L1(Lasso) regularization
model3 = Sequential()
model3.add(Dense(32, activation='relu', input_dim=6, kernel_regularizer=l1(l=0.01)))
model3.add(Dense(16, activation='relu', kernel_regularizer=l1(l=0.01)))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history3 = model3.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation))




# NN with L2(Ridge) Regularization
model4 = Sequential()
model4.add(Dense(32, activation='relu', input_dim=6, kernel_regularizer=l2(l=0.01)))
model4.add(Dense(16, activation='relu', kernel_regularizer=l2(l=0.01)))
model4.add(Dense(1, activation='sigmoid'))
model4.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history4 = model4.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation))



loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']
loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']
loss3 = history3.history['loss']
val_loss3 = history3.history['val_loss']
loss4 = history4.history['loss']
val_loss4 = history4.history['val_loss']


fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(loss1)), y=loss1,
                    name='Training Loss without Regularization', line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=np.arange(len(val_loss1)), y=val_loss1,
                    name='Validation Loss without Regularization', line = dict(color='firebrick')))

fig.add_trace(go.Scatter(x=np.arange(len(loss2)), y=loss2,
                    name='Training Loss with Dropout', line=dict(color='royalblue', dash='dash')))
fig.add_trace(go.Scatter(x=np.arange(len(val_loss2)), y=val_loss2,
                    name='Validation Loss with Dropout', line = dict(color='firebrick', dash='dash')))

fig.add_trace(go.Scatter(x=np.arange(len(loss3)), y=loss3,
                    name='Training Loss with L1 Regularization', line=dict(color='royalblue', dash='dot')))
fig.add_trace(go.Scatter(x=np.arange(len(val_loss3)), y=val_loss3,
                    name='Validation Loss with L1 Regularization', line = dict(color='firebrick', dash='dot')))

fig.add_trace(go.Scatter(x=np.arange(len(loss4)), y=loss4,
                    name='Training Loss with L2 Regularization', line=dict(color='royalblue', dash='longdashdot')))
fig.add_trace(go.Scatter(x=np.arange(len(val_loss4)), y=val_loss4,
                    name='Validation Loss with L2 Regularization', line = dict(color='firebrick', dash='longdashdot')))


fig.update_layout(xaxis_title='Epochs',
                  yaxis_title='Loss',
                  title={'text': "Training and Validation Losses for Different Models",
                                                'x':0.5,
                                                'xanchor': 'center',
                                                'yanchor': 'top'})
iplot(fig)



model = Sequential()
model.add(Dense(32, activation='relu', input_dim=6, kernel_regularizer=l2(l=0.01)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(l=0.01)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32)

```

#Comparing Performances of SVM and Neural Network
```
print("Accuracy for SVM on test data: {}%\n".format(round((svm_model.score(X_test, y_test)*100),2)))
print("Accuracy for Neural Network model on test data: {}%".format(round((model.evaluate(X_test, y_test)[1]*100),2)))


y_pred = svm_model.predict(X_test)
plt.title("SVM Confusion Matrix for Test Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")


y_pred = model.predict(X_test)
threshold = 0.6
y_pred = [1 if i >= threshold else 0 for i in y_pred]
plt.title("Neural Network Confusion Matrix for Test Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

```



## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
## Fig 01- BOX PLOT FOR FEATURES : 
![__results___12_1](https://github.com/user-attachments/assets/d6751398-5a21-43ad-8d3c-e907eb6fba6c)
## Fig 02 - CORRELATION TABLE FOR FEATURES :
![__results___13_0](https://github.com/user-attachments/assets/663da6dd-9dad-42d5-8349-31f8c45c0f60)
## Fig 03 - OCCUPANCY RATE
![newplot](https://github.com/user-attachments/assets/6dccb365-b9c6-4bd1-824a-ceded761ad31)
## Fig - 04 - OCCUPANCY DISTRIBUTION
![__results___17_0](https://github.com/user-attachments/assets/c5a52eca-477c-4e0b-afb2-bda54ed78cdb)
## Fig - 05 - GRAPHICAL REPRESENTATION 
![__results___20_0](https://github.com/user-attachments/assets/b0383002-30b0-48df-ad81-c80c097f6f4b)
## Fig - 06 - ACCUARCY
![newplot (1)](https://github.com/user-attachments/assets/916d8e33-57d5-4892-91e3-a20fbeaac700)
## Fig - 07 - KNN CONFUSION MATRIX 

![__results___29_0](https://github.com/user-attachments/assets/a0cbd0f1-3b6f-47d9-9fcc-57b7d510fa1e)
## Fig - 08 - SVM CONFUSION MATRIX

![__results___32_0](https://github.com/user-attachments/assets/8a790f80-eaf1-466a-8e5f-15d6f381d563)

## Fig - 09 - TRAINING AND VALIDATION 

![newplot (2)](https://github.com/user-attachments/assets/2aafbf24-0643-4536-96a0-d5ac118613fe)
## Fig - 10 - CONFUSION MATRIX FOR TEST DATA
![__results___45_0](https://github.com/user-attachments/assets/77ebd4be-a5fe-4696-9718-418520b79b7c)
## Fig - 11 - CONFUSION MATRIX FOR TEST DATA
![__results___46_0](https://github.com/user-attachments/assets/0e31328e-f9c3-4fd7-9ffb-71a6de162483)




## Results and Impact
<!--Give the results and impact as shown below-->
Both of the models did great job when predicting occupancy. Our accuracy is nearly 98%. So what do you think, which method (ML or DL) is suitable for this dataset and problem?

Before answer that, look at the confusion matrix which are created when evaluating models with the test data. SVM model looks like biased toward occupied class. But we don't have that problem with neural network. So we can say that, we could use neural network for more stable and accurate results without significant errors.


## Articles published / References
~~~
1. Liu, D., et al. "Smart HVAC systems: Demand-based energy consumption for improved efficiency using ML." Energy Reports.


2. Wang, S., et al. "Real-time occupancy detection using CNNs for HVAC control." IEEE Access.


3. Mirakhorli, A., et al. "Machine learning models for occupant behavior in building energy systems." Energy and Buildings.


4. Liu, L., et al. "Predictive models for HVAC occupancy using environmental sensors." Building Simulation.


5. Yang, R., and Wang, L. "Occupancy detection for smart buildings using machine learning." MDPI Sensors.


6. Jain, R.K., et al. "Modeling occupant behavior for energy efficiency using machine learning." Energy and Buildings.


7. Daoud, K., et al. "A review of ML for energy efficiency in HVAC systems." Renewable and Sustainable Energy Reviews.


8. Yang, X., et al. "Improving building energy performance with occupancy prediction models." Building and Environment.


9. Ali, M., et al. "Deep learning for smart HVAC systems: A review." IEEE Internet of Things Journal.


10. Chen, Z., et al. "ML models for predicting HVAC energy consumption." Applied Energy.


11. Das, S., et al. "Occupant-centric control using deep learning methods for smart HVAC systems." Automation in Construction.


12. Hong, T., et al. "Occupancy-driven HVAC controls with machine learning algorithms." Building Research & Information.


13. Caldas, L.G., et al. "Occupancy prediction in smart buildings with machine learning." Energy and AI.


14. Ma, Y., et al. "Smart thermostats and occupant behavior prediction using deep learning." IEEE Transactions on Smart Grid.


15. Kim, M., et al. "HVAC load forecasting using machine learning." Energy.


16. Yu, S., et al. "A novel hybrid ML model for real-time HVAC control." IEEE Transactions on Cybernetics.


17. Lee, E., et al. "Deep learning-based HVAC control in commercial buildings." Journal of Building Performance Simulation.


18. Zhu, X., et al. "Multi-agent reinforcement learning for building energy optimization." Applied Energy.


19. Liu, Y., et al. "End-to-end deep learning for HVAC systems optimization." Energy and Buildings.


20. Zhang, Y., et al. "Using artificial intelligence for HVAC energy savings." IEEE Access.


21. Ahmad, T., et al. "A machine learning-based framework for energy management in smart homes." Sustainable Cities and Society.


22. Xie, J., et al. "Occupant comfort and energy efficiency trade-offs in HVAC systems." Building and Environment.


23. Schiavon, S., et al. "Occupancy-driven HVAC control: A deep learning approach." Energy Efficiency.


24. Luo, X., et al. "ML for predicting indoor air quality and occupancy patterns." Building Simulation.


25. Ding, Y., et al. "Occupancy detection using Wi-Fi signals for smart HVAC systems." IEEE Sensors Journal.


26. Chan, H., et al. "Deep learning for HVAC demand response in smart buildings." IEEE Transactions on Smart Grid.


27. Liang, J., et al. "Data-driven approaches for smart HVAC systems." Energy and AI.


28. Fadlullah, Z.M., et al. "AI-enhanced HVAC system control for energy efficiency." IEEE Communications Magazine.


29. Wan, K., et al. "Machine learning methods for HVAC energy consumption prediction." Energy and Buildings.


30. Liu, W., et al. "Building energy performance optimization with deep learning." Energy Conversion and Management.


31. Zhong, X., et al. "Deep neural networks for HVAC systems energy efficiency." Journal of Cleaner Production.


32. Zhu, Q., et al. "A review of AI applications in HVAC systems." IEEE Access.


33. Agarwal, Y., et al. "Occupancy sensing and HVAC optimization using ML algorithms." Sensors.


34. Kang, D., et al. "Smart HVAC systems with occupancy detection using deep learning." IEEE Transactions on Industrial Electronics.


35. Lopez, C., et al. "Artificial neural networks for occupancy detection in HVAC systems." Energy Reports.


36. Ma, Z., et al. "Deep reinforcement learning for HVAC system control." Energy Reports.


37. Nguyen, T., et al. "A hybrid machine learning approach for HVAC systems optimization." Building and Environment.


38. Wei, C., et al. "Occupant behavior prediction in smart buildings using deep learning." IEEE Internet of Things Journal.


39. Gupta, A., et al. "Energy efficiency in HVAC systems using machine learning." Sustainable Energy Technologies and Assessments.


40. Zhao, Y., et al. "Data-driven methods for energy-efficient HVAC control." Renewable Energy.
~~~


