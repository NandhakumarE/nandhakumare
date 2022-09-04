# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 15:35:22 2022

@author: Nk
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets, ensemble
from sklearn import neighbors
from sklearn import linear_model
import glob, os
import time
import pickle
import numpy as np

from skimage.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def text_to_csv(folder_path, file_name, column_drop):
    # Start Time
    starttime = time.time()
    print(f"Process started : {starttime}")
    file_list = glob.glob(folder_path + "/*.txt")
    file_list.sort()
    dataset1 = pd.read_fwf(file_list[0], sep=" ", header=None)
    # Transfor Rows to Column and Column to Rows
    data = dataset1.T.drop(columns=[column_drop])
    pm = data.apply(sorted, key=pd.isnull)
    main_dataframe = pm[~pd.isnull(pm).all(1)].fillna("")
    count = 0
    for i in range(1, len(file_list)):
        count += 1
        print(f"Processed files count : {count}")
        dataset = pd.read_fwf(file_list[i], sep=" ", header=None)
        # Transfor Rows to Column and Column to Rows
        data = dataset.T.drop(columns=[column_drop])
        # Drop Column Header
        data = data.drop(index=0)
        pm = data.apply(sorted, key=pd.isnull)
        new = pm[~pd.isnull(pm).all(1)].fillna("")
        main_dataframe = pd.concat([main_dataframe, new], axis=0)
    main_dataframe.to_csv(file_name, index=False, header=None)
    print("CSV File Saved Successfully")
    # End Time
    endtime = time.time()

    print(f"Total time to process {(endtime - starttime)/60}")


def preprocess_data(file_path, file_name):
    # label_encoder object knows how to understand word labels.
    # label_encoder = preprocessing.LabelEncoder()
    # Encode labels in column 'Country
    df = pd.read_csv(file_path)
    print(f"Shape of DataFrame : {df.shape}")
    df = pd.get_dummies(
        df, columns=["Weather conditions", "Road_traffic_density", "Festival", "City"]
    )

    # Label Encoding
    # df["Weather conditions"] = label_encoder.fit_transform(df["Weather conditions"])
    # df["Road_traffic_density"] = label_encoder.fit_transform(df["Road_traffic_density"])
    # df["Festival"] = label_encoder.fit_transform(df["Festival"])
    # df["City"] = label_encoder.fit_transform(df["City"])
    print("Check Null Values")
    a = df.isnull().sum()

    b = a[a < (0.05 * len(a))]
    print(f"Null Values Column Greater than 0.05 ratio : {b}")
    print(f"Shape after null : {b.shape}")

    # Drop null valus rows
    df = df.dropna(how="any")
    print(f"After dropping null values : {df.shape}")
    # Droping unwanted Column
    # df = df.drop(["ID","Delivery_person_ID","Order_Date",])

    # Droping Column with corr
    target = "Time_taken (min)"
    x = df.select_dtypes(include=["integer", "float"]).corr()[target].abs()
    x = x.round(2)
    print(f"Corrlation between target vs features : {x}")
    print(x.shape)
    cor = 0.2
    df = df.drop(x[x < cor].index, axis=1)
    print(f"Dropping correlation less than {cor} values :{df.shape}")
    df = df.drop(
        [
            "ID",
            "Delivery_person_ID",
            "Type_of_order",
            "Order_Date",
            "Time_Orderd",
            "Time_Order_picked",
            "Delivery_person_ID",
            "Type_of_vehicle",
        ],
        axis=1,
    )
    print(df.shape)
    df.to_csv(file_name, index=None)
    return df


def pre_process_input(df):
    df = df.copy()

    # Drop Unwanted columns
    print("Shape befor preprocess : {df.info()}")
    column = [
        "ID",
        "Delivery_person_ID",
    ]
    df = df.drop(column, axis=1)

    # Fillna Values with 0
    df["Time_Orderd"] = pd.to_datetime(
        df["Time_Orderd"], format="%H:%M", errors="coerce"
    ).fillna("00:00")
    df["Time_Orderd_hr"] = df["Time_Orderd"].dt.hour
    df["Time_Orderd_min"] = df["Time_Orderd"].dt.minute
    df["Time_Order_picked"] = pd.to_datetime(
        df["Time_Order_picked"], format="%H:%M", errors="coerce"
    ).fillna("00:00")
    df["Time_Order_picked_hr"] = df["Time_Order_picked"].dt.hour
    df["Time_Order_picked_min"] = df["Time_Order_picked"].dt.minute
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"], format="%YY-%MM-%DD", errors="coerce"
    ).fillna("1000-00-00")
    df["Order_Date_year"] = df["Order_Date"].dt.year
    df["Order_Date_month"] = df["Order_Date"].dt.month
    df["Order_Date_day"] = df["Order_Date"].dt.day

    df = df.drop(["Time_Orderd", "Time_Order_picked", "Order_Date"], axis=1)

    # Checking Uniques Values Object
    columns_object = {
        column: len(df[column].unique())
        for column in df.select_dtypes(["object"]).columns
    }

    print(f"Unique Values in Object {columns_object}")

    # Binary Encoding or Label Encoding
    label_encoder = preprocessing.LabelEncoder()
    df["Festival"] = label_encoder.fit_transform(df["Festival"])

    # One Hot Encoding
    columns_name = [
        "Weather conditions",
        "Road_traffic_density",
        "City",
        "Type_of_order",
        "Type_of_vehicle",
    ]
    df = pd.get_dummies(df, columns=columns_name)
    print(f"After Shape Encoding Dataframe shape : {df.shape}")

    df = df.fillna(0, axis=1)
    print(f"After Shape Encoding Dataframe shape : {df.shape}")

    # Cor checking btw Target vs feature
    target = "Time_taken (min)"
    x = df.select_dtypes(include=["integer", "float"]).corr()[target].abs()
    print(f"Checking correlation : {x}")

    # Scaling data's
    scaler = StandardScaler()
    y = df["Time_taken (min)"]
    x = df.drop("Time_taken (min)", axis=1)

    scaler.fit(x)
    df_x = pd.DataFrame(scaler.transform(x), index=x.index, columns=x.columns)
    print(df_x)
    df = pd.concat([df_x, y], axis=1)

    return df


def pre_proc_inp_test(df):
    df = df.copy()

    # Drop Unwanted columns
    print(df.info())

    column = [
        "ID",
        "Delivery_person_ID",
    ]
    df = df.drop(column, axis=1)

    # Convert the datatypes object to datetime format for parsing date values
    df["Time_Orderd"] = pd.to_datetime(
        df["Time_Orderd"], format="%H:%M", errors="coerce"
    ).fillna("00:00")
    df["Time_Orderd_hr"] = df["Time_Orderd"].dt.hour
    df["Time_Orderd_min"] = df["Time_Orderd"].dt.minute

    df["Time_Order_picked"] = pd.to_datetime(
        df["Time_Order_picked"], format="%H:%M", errors="coerce"
    ).fillna("00:00")

    df["Time_Order_picked_hr"] = df["Time_Order_picked"].dt.hour
    df["Time_Order_picked_min"] = df["Time_Order_picked"].dt.minute
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"], format="%YY-%MM-%DD", errors="coerce"
    ).fillna("1000-00-00")
    df["Order_Date_year"] = df["Order_Date"].dt.year
    df["Order_Date_month"] = df["Order_Date"].dt.month
    df["Order_Date_day"] = df["Order_Date"].dt.day

    # Droping Because not required we converted the dates
    df = df.drop(["Time_Orderd", "Time_Order_picked", "Order_Date"], axis=1)

    # Checking Uniques Values Object
    columns_object = {
        column: len(df[column].unique())
        for column in df.select_dtypes(["object"]).columns
    }

    print(f"Unique Values in Object {columns_object}")

    # Binary Encoding or Label Encoding
    label_encoder = preprocessing.LabelEncoder()
    df["Festival"] = label_encoder.fit_transform(df["Festival"])

    # One Hot Encoding
    columns_name = [
        "Weather conditions",
        "Road_traffic_density",
        "City",
        "Type_of_order",
        "Type_of_vehicle",
    ]
    df = pd.get_dummies(df, columns=columns_name)
    print(f"After Shape Encoding Dataframe shape : {df.shape}")

    # Filling 0 for Nan Values in Data Frame
    df = df.fillna(0, axis=1)
    print(df.info())

    # Converting to Standard all data's
    scaler = StandardScaler()
    x = df
    scaler.fit(x)
    df_x = pd.DataFrame(scaler.transform(x), index=x.index, columns=x.columns)
    print(f"After Converting to StndardScaler : {df_x}")

    return df


def train_model(df_x_train):

    x = df_x_train[list(df_x_train.columns)[:-1]]
    y = df_x_train["Time_taken (min)"]
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    params = {
        "n_estimators": 1000,
        "max_depth": 4,
        "min_samples_split": 10,
        "learning_rate": 0.1,
        "loss": "squared_error",
    }

    reg = ensemble.GradientBoostingRegressor(**params)

    fit_model = reg.fit(x_train, y_train)

    mse = mean_squared_error(y_test, reg.predict(x_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    y_pre = reg.predict(x_test)
    print(f"Y Pred Value :{y_pre}")

    print(f"Regression Score : {reg.score(x_test, y_test)}")

    reg_score = reg.score(x_test, y_test)

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pre)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=4)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()

    return fit_model, reg_score


if __name__ == "__main__":
    pass

    cur_w_d = os.getcwd()
    print(f"Current working directory : {cur_w_d}")

    # folder_name = ["pre_processed_data", "saved_model", "result"]
    # for folder in folder_name:
    #     os.makedirs(folder, exist_ok=True)
    # # Text to CSV
    # text_to_csv(
    #     f"{cur_w_d}\\dataset\\train\\",
    #     f"{cur_w_d}\\pre_processed_data\\Combined_train_data.csv",
    #     20,
    # )
    # text_to_csv(
    #     f"{cur_w_d}\\dataset\\test\\",
    #     f"{cur_w_d}\\pre_processed_data\\Combined_test_data.csv",
    #     19,
    # )

    # # PreProcess the Test and Training data

    data = pd.read_csv(
        f"{cur_w_d}\\pre_processed_data\\Combined_train_data.csv",
        parse_dates=["Order_Date"],
    )
    data_test = pd.read_csv(
        f"{cur_w_d}\\pre_processed_data\\Combined_test_data.csv",
        parse_dates=["Order_Date"],
    )

    df_train = pre_process_input(data)

    # Train the Model
    reg, score = train_model(df_train)

    print(f"Model Score :{100*score}")
    ##Saving the Prediction Model
    Pkl_Filename = f"{cur_w_d}\\saved_model\\Pickle_Gb_Model.pkl"
    with open(Pkl_Filename, "wb") as file:
        pickle.dump(reg, file)
    # Saved Model
    loaded_model = pickle.load(
        open(f"{cur_w_d}\\saved_model\\Pickle_Gb_Model.pkl", "rb")
    )

    # Predict the Model
    # PreProcess Input Test data
    df_test = pre_proc_inp_test(data_test)

    result = loaded_model.predict(df_test)
    result = pd.DataFrame({"Time_taken (min)": result})
    result["Time_taken (min)"] = result["Time_taken (min)"].apply(np.int64)
    df = pd.concat([data_test["ID"], result], axis=1)
    print(f"Submission Result : {df}")

    # Saved Submission result to CSV
    df.to_csv(f"{cur_w_d}\\result\\result_submission.csv", index=None)
