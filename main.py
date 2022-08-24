# Libraries
# import scikit library for regression model AI
from sklearn import linear_model
from sklearn import metrics
# import numpy for convert python array to vertical matrix
import numpy as np
# import xlrd for read Excel files
import xlrd


# Functions
# functions for read Excel files
def read_excel_files(file_loc):
    file = xlrd.open_workbook(file_loc)
    return file.sheet_by_index(0)


# get dataset information from Excel file and create list from them
dataset_loc = "Iran_Population.xls"
dataset_info = read_excel_files(dataset_loc)
dataset = []
for i in range(0, dataset_info.nrows):
    data_items = [dataset_info.cell_value(i, 0), dataset_info.cell_value(i, 1)]
    dataset.append(data_items)

# create x train and y train by vertical format
year_train = []
population_train = []
for data_item in dataset:
    year_train.append(data_item[0])
    population_train.append(data_item[1])

x_train = np.array(year_train).reshape(-1, 1)
y_train = np.array(population_train).reshape(-1, 1)

# create vertical matrix for test
test_array = [1398, 1399, 1400, 1401]
test_matrix = np.array(test_array).reshape(-1, 1)

regression_obj = linear_model.LinearRegression()
regression_obj.fit(x_train, y_train)
predict_list = regression_obj.predict(test_matrix)

print(test_array)
print(predict_list)
