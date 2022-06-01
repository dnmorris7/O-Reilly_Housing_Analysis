from asyncio import sleep
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT= "https://github.com/ageron/handson-ml/tree/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        print("Downloading...")
    # if data is not already downloaded, download it
    if not os.path.isfile(os.path.join(housing_path, "housing.tgz")):
        urllib.request.urlretrieve(housing_url, os.path.join(housing_path, "housing.tgz"))
        print("Download complete")
    tgz_path = os.path.join(housing_path, "housing.tgz")
    print("Target Path: " + tgz_path)
    
    print("Extracting...")
    sleep(10000)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    



import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data(HOUSING_URL, HOUSING_PATH)
df = load_housing_data(HOUSING_PATH)
print(df.head())
df.info()
print(df['ocean_proximity'].value_counts())