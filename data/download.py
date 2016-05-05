import urllib.request
from zipfile import ZipFile

kaggle_url = "https://www.kaggle.com/c/sf-crime/download/train.csv.zip"
f = urllib.request.urlretrieve(kaggle_url)
data_zip = ZipFile(f)
data_zip.extractall()
