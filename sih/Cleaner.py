"""
This Python file consists of function to perform basic data cleaning/data preprocessing operation on most dataset.
Functions includes, Removal of Unique COlumns,High Null value ratio, Missing Value Handling, String Categorical feature Handling .
"""
import io
import re 
import urllib
import os.path
#os.chdir('<path URL>')
import requests
import httplib2
import pandas as pd
from requests.models import HTTPError 
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tarfile import is_tarfile
import os,tarfile,requests,warnings
from zipfile import ZipFile, is_zipfile
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from scipy.stats import kruskal

global dic
dic = {}
def logs(dic, key, value):
    dic[key]=value

#from statsmodels.tsa.stattools import kpss,adfuller
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["PYTHONWARNINGS"] = "ignore"
    #from statsmodels.tsa.stattools import kpss,adfuller
    
def dataCleaner(df,features,target):
    updateddf=RemoveRowsWithHighNans(df)
    updateddf=RemoveHighNullValues(updateddf)
    updateddf=dropUniqueColumn(updateddf,target)
    if updateddf.isnull().values.any(): 
        cols=updateddf.columns[updateddf.isnull().any()].tolist()
        for i in cols:
            Cleaner(updateddf,i)

    X_values,Y_value=updateddf.drop(target,axis=1),updateddf[target]
    if target in updateddf.columns.to_list():EncoderResult=Encoder(X_values,Y_value,target)
    else:EncoderResult=Encoder(X_values,None,target)

    return EncoderResult

def dropUniqueColumn(X_values,target):  
    initial = X_values
    row_counts = len(X_values)
    for i in X_values.columns.to_list():
        if len(X_values[i].unique())==row_counts and i!=target:
            X_values.drop(i, axis=1, inplace=True)
    
    return X_values

def RemoveHighNullValues(dataframe):

    thresh = len(dataframe) * .5
    dataframe.dropna(thresh = thresh, axis = 1, inplace = True)
    return dataframe

def Cleaner(df,i):
    if(df[i].dtype in ["float","int"]):
        if(len(np.unique(df[i]))<=3):
            df[i].fillna(df[i].mode()[0],inplace=True)
            print("missingdict:-mode")
        else:
            df[i].fillna(df[i].mean(),inplace=True)   
            print("missingdict:-mean")
    elif(df[i].dtype=="object"):
        df[i].fillna(df[i].mode()[0],inplace=True)
        print("missingdict:-mode")

def Encoder(X,Y=None,target=""): 
    encode=dict()
    print("hey this is encoder :",X.info())
    if("object" in X.dtypes.to_list() or Y.dtype=="object"):
        if("object" in X.dtypes.to_list()):
            #objectTypes(X,DictionaryClass)
            X=pd.get_dummies(X)
            encode['X']='OneHotEncode' 
        dataframe=X.copy(deep=True)
        if(Y.dtype=="object"):
            encode['Y']='LabelEncoder' 
            original_labels=np.sort(pd.unique(Y), axis=-1, kind='mergesort')
            Y=LabelEncoder().fit_transform(Y)
            encoded_label=[xi for xi in range(len(original_labels))]
            encodes={encoded_label[i]:original_labels[i] for i in range(len(original_labels))}
        dataframe[target]=Y
        logs(dic, "Encoded", dataframe.columns)
        return dataframe
    else:
        dataframe=X.copy(deep=True)
        dataframe[target]=Y
        return dataframe

def objectTypes(X,DictionaryClass):
    """
    param1: pandas.dataframe
    param2: class object

    Function indentifies existence of String Categorical features.
    If String Categorical Feature exist record the list of features with string data in Class List Variable,
    and set boolean flag for existence to True else False
    """

    g = X.columns.to_series().groupby(X.dtypes).groups
    gd={k.name: v for k, v in g.items()}
    if 'object' in gd.keys():
        if DictionaryClass!=None:
            DictionaryClass.ObjectExist=True
            DictionaryClass.ObjectList= gd['object'].to_list()  
    else:
        if DictionaryClass!=None:DictionaryClass.ObjectExist= False

def RemoveRowsWithHighNans(dataframe):

    percent = 80.0
    min_count = int(((100-percent)/100)*dataframe.shape[1] + 1)
    dataframe = dataframe.dropna( axis=0, 
                    thresh=min_count)
    logs(dic, "High Nan Dropping Threshold", percent)
    return dataframe

def scaling_data(dataframe,update=False):
    
    scaler=MinMaxScaler() 
    X=scaler.fit_transform(dataframe) 
    if isinstance(dataframe,pd.DataFrame):X=pd.DataFrame(data = X,columns = dataframe.columns)
    logs(dic, "Scaling", "MinMaxScaler")
    return X

def uncompress_file(file):

    if os.path.isfile(file):
        return decompress(file)
    else:
        raise FileNotFoundError(f"provided path {file} does not exist")
    
def decompress(file):
    try:
        ogpath=os.path.splitext(file)
        extract_dir="./"+os.path.basename(ogpath[0])
        if is_zipfile(file):
            with ZipFile(file,"r") as zip_ref:
                members=zip_ref.namelist()
                for file in members:
                    zip_ref.extract(member=file,path=extract_dir)
        elif is_tarfile(file):
            tar = tarfile.open(file, mode="r:gz")
            members=tar.getmembers()
            for member in members:
                tar.extract(member=member,path=extract_dir)
        print(f"file has been decompressed to folder {extract_dir}")
    except Exception as e:print(e)
    return extract_dir

def file_from_url(url):
    try:
        ogpath=os.path.splitext(url)
        download_path="./"+os.path.basename(ogpath[0])+ogpath[-1]
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with tqdm.wrapattr(open(download_path, "wb"), "write", miniters=1,total=total,desc="Downloading :") as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)
        print("file_from_url :",download_path)
        return download_path
    except Exception as e: print(e)

def check_subfolder_data(file):
    """
    param1:string
    param2: Class object
    return: Tuple(string,list)
    """
    print("file path for subfolder :",file)
    targets = os.listdir(file)
    #targets=file
    print(f"identified target are :{targets}")
    check_status=True
    for category in targets:
        path=os.path.join(file, category)
        if not os.path.isfile(path):
            for img in os.listdir(path):
                try:
                    extension = os.path.splitext(img)[1]
                    print("checking for the extension",extension)
                    check_status= check_status if extension in ['.png',".PNG",".jpg",".jpeg",'.JPEG',''] else False 
                    if not check_status:break
                except Exception as e:print(e)
        else: check_status=False
        if not check_status:break
    #DictionaryClass.addKeyValue('features',{'Y_values':targets})
    if not check_status: raise TypeError("some files have different formats")
    return (file,targets)

def quick_image_processing(path,size):
    data = cv2.imread(path)
    
    data=cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
    img_resize=cv2.resize(data,(size,size))
    img_data=[img_resize.flatten()]
    logs(dic, "Image Shape", (size, size))
    return (img_data,data)


def timeseries_cleaner(X,date,target,samplingtype):
    X=X.loc[0:X.shape[0],[date,target]].copy(deep=True) 
    updateddf=RemoveRowsWithHighNans(X)
    updateddf=RemoveHighNullValues(X)
    updateddf=parsetime(X,date)
    updateddf,time_frequency=FrequencyChecker(updateddf,date)
    #X = updateddf[target].values
    #s = StationarityTest()
    #s.results(X)
    #updateddf=frequencysampling(updateddf,date,time_frequency,samplingtype) if samplingtype!=None else updateddf
    #updateddf=RemoveHighNullValues(updateddf)
    logs(dic, "Testing", "Frquency Sampling")
    return updateddf,dic


def parsetime(df,date):
    try:
        df[date]= pd.to_datetime(df[date])
        return df
    except:
        try:
            df[date] = pd.to_datetime(df[date],format="%d.%m.%Y")

        except:
            raise TypeError("Unsupported Date Format")
        
def FrequencyChecker(df,date):
    df_copy=df.copy(deep=True)
    df['Month']=df[date].dt.month
    df['Year']=df[date].dt.year
    df['Hour']=df[date].dt.hour
    df['Days']=df[date].dt.day_name()
    d=df.Days.nunique()
    h=df.Hour.nunique()
    m=df.Month.nunique()
    if(h>1 and h<=24):
        time_frequency="H"
        
    elif(d>1 and d<=7):
        time_frequency="D"
        
    elif(m>1 and m<=12):
        time_frequency="M"
        
    else:
        time_frequency="Y"
    df_copy=df_copy.set_index(date)
    return df_copy,time_frequency
   
class StationarityTest:
    def __init__(self, SignificanceLevel=.05,test=[]):
        self.SignificanceLevel = SignificanceLevel 
        self.test=test
    def ADF_Stationarity_Test(self, timeseries):
        adfTest = adfuller(timeseries, autolag='AIC')
        self.pValue = adfTest[1]
        if (self.pValue<self.SignificanceLevel):
            self.test.append(True)
        else:
            self.test.append(False)
            
    def kpss_test(self,timeseries):
        statistic, p_value, n_lags, critical_values = kpss(timeseries)
        if (p_value < self.SignificanceLevel):
            self.test.append(False)
        else:
            self.test.append(True)
    def seasonality_test(self,timeseries):
        seasoanl = False
        idx = np.arange(len(timeseries)) % 12
        H_statistic, p_value = kruskal(timeseries, idx)
        if p_value <= self.SignificanceLevel:
            seasonal = True
        self.test.append(seasonal)
    def results(self,timeseries):
        StationarityTest.ADF_Stationarity_Test(self, timeseries)
        StationarityTest.kpss_test(self,timeseries)
        StationarityTest.seasonality_test(self,timeseries)
        
        result=max(self.test, key=self.test.count)
       
        return result

def frequencysampling(df,date,time_frequency,samplingtype):
    downsample=""
    #df=df.set_index(date)
    if (samplingtype=="day" and time_frequency=="H"):
        dff=df.resample('H').mean()
        downsample="day"
        print("downsample:day")
        
    elif(samplingtype=="week" and time_frequency in ["H","D"]):
        dff=df.resample("D").mean()
        downsample="week"
        print("downsample:week")

    elif (samplingtype=="month" and time_frequency in ["H","D"]):
        dff=df.resample('M').mean()
        downsample="month"
        print("downsample:month")
        
    elif (samplingtype=="quarterly" and time_frequency in ["H","D","M"]):
        dff=df.resample("Q").mean()
        downsample="quarterly"
        print("downsample:quarterly")
        
    elif (samplingtype=="year" and time_frequency in ["H","D","M"]):
        different_locale=df.resample('M').mean()
        downsample="year"
        print("downsample:year")

    else: #(samplingtype not in ["year","quaterly"," month","week","day",None])
        raise ValueError(f"{samplingtype} is not a valid option, valid options are ['year','quaterly','month','week','day',None]")
    #if downsample not in [None,""]:
        #dictclass.addKeyValue("cleaning",{"downsample":downsample})
    

    return dff

def spliter(df):
    size=df.shape[0]
    trainsize=int(np.round(90*size/100))
    train_data=df.iloc[0:trainsize,:].squeeze()
    test_data=df.iloc[trainsize:,:].squeeze()
    return train_data, test_data 



def validate_url(url: str):
    """
    param1: string
    
    return: boolean
    Function perform a symantic check/regular expression check whether the provided string is in format of URL Types
    """
    DOMAIN_FORMAT = re.compile(
        r"(?:^(\w{1,255}):(.{1,255})@|^)" # http basic authentication [optional]
        r"(?:(?:(?=\S{0,253}(?:$|:))" # check full domain length to be less than or equal to 253 (starting after http basic auth, stopping before port)
        r"((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+" # check for at least one subdomain (maximum length per subdomain: 63 characters), dashes in between allowed
        r"(?:[a-z0-9]{1,63})))" # check for top level domain, no dashes allowed
        r"|localhost)" # accept also "localhost" only
        r"(:\d{1,5})?", # port [optional]
        re.IGNORECASE
    )
    SCHEME_FORMAT = re.compile(
        r"^(http|hxxp|ftp|fxp)s?$", # scheme: http(s) or ftp(s)
        re.IGNORECASE
    )
    url = url.strip()
    try:
        if not url:raise Exception("No URL specified")
        result = urllib.parse.urlparse(url)
        scheme = result.scheme
        domain = result.netloc
        if not scheme:raise Exception("No URL scheme specified")
        if not re.fullmatch(SCHEME_FORMAT, scheme):raise Exception("URL scheme must either be http(s) or ftp(s) (given scheme={})".format(scheme))
        if not domain:raise Exception("No URL domain specified")
        if not re.fullmatch(DOMAIN_FORMAT, domain):raise Exception("URL domain malformed (domain={})".format(domain))
        return check_url_existence(url)
    except Exception:return False

def check_url_existence(url):
    """
    param1: string
    return: boolean
    Function check whether the provided string url exists over internet or not.
    """
    h = httplib2.Http()
    resp = h.request(url, 'HEAD')
    if int(resp[0]['status']) < 400:
        return True
    else: raise HTTPError(f"{url} does not exist")

def val(a):
    print(a)
    print(dic)
