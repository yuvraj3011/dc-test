#from msilib.schema import Feature
#from pyexpat import features
from django.shortcuts import render
import mimetypes
from urllib.error import HTTPError
#from sih.AutoFeatureSelection import FeatureSelection
from sih.AutoFeatureSelection import *
from django.core.files.storage import FileSystemStorage
#import AutoFeatureSelection
from sih.models import File_Data
import smtplib
from email.message import EmailMessage
#from django.core.handlers import FilePathField

from django.shortcuts import redirect  

import librosa
#! pip install librosa
import pandas as pd
import os
import librosa
import tqdm
#import IPython.display as ipd
#import matplotlib.pyplot as plt
import numpy as np
#import sklearn
#import tensorflow as tf
import scipy
import gc


#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler
#from sklearn.model_selection import GridSearchCV

from warnings import simplefilter

from scipy.io import wavfile
from scipy.stats import skew
from tqdm import tqdm, tqdm_pandas


from sih.Cleaner import *
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
import os
#from somewhere import handle_uploaded_file

#Katherine
from django.utils.datastructures import MultiValueDictKeyError

# Create your views here.
from django.http import HttpResponse
import pandas as pd

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk

#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
from tqdm.notebook import tqdm
tqdm.pandas()

def dataframe(path, delimiter=","):
    extension = path.split(".")[1]


    if extension == "txt":
        df = pd.read_csv(path, delimiter=delimiter)
    elif extension == "csv":
        df = pd.read_csv(path, delimiter=delimiter)
    elif extension == "tsv":
        df = pd.read_csv(path, delimiter=delimiter)

    return df


def get_dataframe_type(file_path):
    #extension = os.path.splitext(file_path)[1]
    extension=file_path.split('.')[1]
    #global df

    try:
        if (extension == "csv"):
            Types = "csv"
            df = pd.read_csv(file_path)
        elif extension == "xlsx":
            Types = "xlsx"
            df = pd.read_excel(file_path)
        elif extension == "parquet":
            df = pd.read_parquet(file_path)
        elif extension == "json":
            Types = "JSON"
            df = pd.read_json(file_path)
        elif extension == "pkl":
            df = pd.read_pickle(file_path)
            Types = "Pickle"

    except HTTPError:
        response = requests.get(file_path)
        file_object = io.StringIO(response.content.decode('utf-8'))
        if (extension == "csv"):
            Types = "csv"
            df = pd.read_csv(file_object)
        elif extension == "xlsx":
            Types = "xlsx"
            df = pd.read_xlsx(file_object)
        elif extension == "excel":
            df = pd.read_excel(file_object)
        elif extension == ".parquet":
            df = pd.read_parquet(file_object)
        elif extension == "json":
            Types = "JSON"
            df = pd.read_json(file_object)
        elif extension == "pkl":
            df = pd.read_pickle(file_object)
            Types = "Pickle"

    # if dc!=None: dc.addKeyValue('data_read',{"type":Types,"file":file_path,"class":"df"})
    return df


#def index(request):
 #   return HttpResponse("Hello, SIH")
def index(request):

    return render(request,'index.html')

def Save_File(request):
    if request.method=='POST':
        a=request.FILES['file']
        print(a)
        df = pd.read_csv(a)
        print("row data :", df.head())
        print("row data shape :", df.shape)
        target = "Loan_Status"
        date = "date"
        frequency_sampling_type = None
        X, Y = AutoFeatureSelection.FeatureSelection(df, target)  #
        # res=dataCleaner(df,features,target,DictionaryClass=None)
        print("final df Shape :", X.shape)

        print("Finale dataframe :", X.head())

        my_data=File_Data.objects.create(files=X)
        my_data.save()
        return HttpResponse("Form Submitted")


def classi(request):
    return render(request,'classi.html')

def handle_uploaded_file(f):  
    with open('sih/static/upload/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)  

def classifile(request):
    if request.method=='POST':
        a=request.FILES['classi']
        tr=request.POST['tar']
        handle_uploaded_file(a)
        fileToUse = 'sih/static/upload/'+a.name
        #df = pd.read_csv(a)\
        df = get_dataframe_type(fileToUse)
        print(tr)

        print("row data :", df.head())
        print("row data shape :", df.shape)
        target = tr
        date = "date"
        frequency_sampling_type = None
        X, Y = AutoFeatureSelection.FeatureSelection(df, target)  #
        # res=dataCleaner(df,features,target,DictionaryClass=None)
        X["target"]=Y
        X.to_csv("sih/static/media/train.csv",index=False)
        print("final df Shape :", X.shape)
        log=AutoFeatureSelection.val("LoGS:")
        print(log)

        with open("sih/static/media/"+a.name+"_log.txt", 'w+') as f:
            f.write(str(log))

        print("Finale dataframe :", X.head())

        #my_data=File_Data.objects.create(files=media/train.csv)
        #my_data.save()

        #print(X.name)

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Define the full file path
        #filepath = BASE_DIR + '/media/' + 'train.csv'
        # Open the file for reading content
        #path = open(filepath, 'rb')
        # Set the mime type
        #mime_type, _ = mimetypes.guess_type(filepath)
        # Set the return value of the HttpResponse
        #response = HttpResponse(path, content_type=mime_type)
        # Set the HTTP header for sending to browser
        # response['Content-Disposition'] = "attachment; filename=%s" % X.name
        #response['Content-Disposition'] = "attachment; filename=train.csv"
        # Return the response value
        #return response
        return redirect("/fileDownload?file1=/static/media/train.csv&file2=/static/media/"+a.name+"_log.txt")
    else:
        # Load the template
        return render(request, 'index.html')

def nlp(request):
    return render(request,'nlp.html')


def preprocessing(df, textField, target):
    data = df.iloc[:, textField]
    tar = df[df.columns[target]]
    # print(len(tar))
    # print(data[0])
    preprocessed_data = []
    for i in range(len(data)):
        d1 = re.sub('[^a-zA-z]', ' ', data[i])
        d2 = d1.lower()
        tokens = d2.split()
        stemming = PorterStemmer()
        stemmed = [stemming.stem(word) for word in tokens if word not in set(stopwords.words('english'))]
        preprocessed_data.append(" ".join(stemmed))
    dataframe = pd.DataFrame({"Preprocessed": preprocessed_data, "Label": tar})

    # dataframe.to_csv("result.csv", index=False)
    path = os.getcwd()
    # print(f"Result saved to {path}")

    return preprocessed_data, dataframe

def bagOfWords(data, textField, target):
  data = data.iloc[:, textField]
  v = CountVectorizer()
  df_cv = v.fit_transform(data)
  df = pd.DataFrame(df_cv.toarray())
  df.columns=v.get_feature_names()
  df["Label"] = df.iloc[:, target]
  df.to_csv("media/train.csv", index=False)
  path = os.getcwd()
  print(f"Result saved to {path}")
  dic = {
        "Shape":df.shape,
        "Features":("Vector count","Regular Expression","Tokenization","Stemming","Stop words removal"),
        "Encoded column number":textField,
        "Path":path
    }
  return df_cv.toarray(), dic

#df = dataframe("/content/file.tsv", delimiter="\t")
#df_, log = bagOfWords(df, textField=0, target=1)


def nlpfile(request):
    if request.method=='POST':
        a=request.FILES['nlp']
        nlptarget=request.POST['nlptarget']
        nlpdelimitor=request.POST['nlpdelimitor']
        nlpfield=request.POST['nlpfield']
        print(a)
        #df = pd.read_csv(a)\
        handle_uploaded_file(a)
        #path="media/"+a.name
        path = 'sih/static/upload/' + a.name
        dff = dataframe(path,nlpdelimitor)
        _, log = bagOfWords(dff,int(nlpfield),int(nlptarget))





        #X, Y = AutoFeatureSelection.FeatureSelection(df, target)  #
        # res=dataCleaner(df,features,target,DictionaryClass=None)
        #print("final df Shape :", X.shape)

        #print("Finale dataframe :", X.head())
        with open("logs/"+ a.name +".txt", "w+") as f:
            f.write(str(log))
        
        
        print(log)
        #my_data=File_Data.objects.create(files=X)
        #my_data.save()

        print(a.name)
        print("Success")

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Define the full file path
        filepath = BASE_DIR + '/media/' +'train.csv'
        # Open the file for reading content
        path = open(filepath, 'rb')
        # Set the mime type
        mime_type, _ = mimetypes.guess_type(filepath)
        # Set the return value of the HttpResponse
        response = HttpResponse(path, content_type=mime_type)
        # Set the HTTP header for sending to browser
        #response['Content-Disposition'] = "attachment; filename=%s" % X.name
        response['Content-Disposition'] = "attachment; filename=train.csv"
        # Return the response value
        return response
    else:
        # Load the template
        return render(request, 'index.html')


#    return HttpResponse("Form Submitted")




#    return HttpResponse("Form Submitted")

def ts(request):
    return render(request,'ts.html')


def tsfile(request):
    if request.method=='POST':
        a=request.FILES['ts']
        tstarget=request.POST['tstarget']
        tsdate=request.POST['tsdate']
        tsfeature=request.POST['tsfeature']
        handle_uploaded_file(a)
        path = 'sih/static/upload/'+a.name
        df = get_dataframe_type(path)
        target = tstarget
        date = tsdate
        samplingtype = tsfeature
        my_data=File_Data.objects.create(files=a)
        my_data.save()
        print(a.name)

        dff, log =timeseries_cleaner(df,date,target,samplingtype=None)
        print((dff))
        val("LOGS:")
        print(str(log))
        with open('sih/static/media/'+a.name+'_log.txt', 'w+') as f:
            f.write(str(log))
        dff.to_csv("sih/static/media/train.csv")
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Define the full file path
        filepath = BASE_DIR + '/sih/static/media/' + 'train.csv'
        # Open the file for reading content
        #path = open(filepath, 'rb')
        # Set the mime type
        #mime_type, _ = mimetypes.guess_type(filepath)
        # Set the return value of the HttpResponse
        #response = HttpResponse(path, content_type=mime_type)
        # Set the HTTP header for sending to browser
        #response['Content-Disposition'] = "attachment; filename=train.csv"
        # Return the response value
        #return response
        return redirect('/fileDownload?file1=/static/media/train.csv&file2=/static/media/'+a.name+'_log.txt')
        
    else:
        # Load the template
        return render(request, 'index.html')


#    return HttpResponse("Form Submitted")

def downloadFile(request):
    return render(request, 'fileDownload.html')

def vid(request):
    return render(request,'vid.html')


def trainImage(file=None, target=None,resize=50):
    #data read
    l=[resize]
    if file!=None:
        root, ext = os.path.splitext(file)
        print("File type : ",ext)
        compress_list=[".zip",".tar",".gz",'.tar.gz','.bz2']
        if not ext and ext not in compress_list and target==None:
            print("hy first block is excuting")
            if validate_url(file): 
                file=file_from_url(file) 
            data,target=check_subfolder_data(file)
        elif ext in compress_list:
            if validate_url(file): 
                file=file_from_url(file) 
            file=uncompress_file(file)
            data,target=check_subfolder_data(file)
        else:
            data,target=check_subfolder_data(file)
        data=AutoFeatureSelection.image_processing(data,target,resize)

        return data,data.to_csv("data.csv")
    else:
        raise ValueError("{file} can't be null or empty")


def vidfile(request):
    if request.method=='POST':
        a=request.FILES['vid']
        target=None
        handle_uploaded_file(a)
        path='C:\\Users\\yuvra\\Downloads\\DATACHEF_CodeAssasins_31473\\SIH_Finale-master\\sih\\static\\upload\\'+a.name

        #path = 'sih/static/upload/'+a.name

        #data, target, log = img_train(path, target, resize=50)
        data, target, log = trainImage(path, target, resize=50)
        #print(data)
        data.to_csv("media/train.csv")

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Define the full file path
        #filepath = BASE_DIR + '/media/' + a.name
        filepath = BASE_DIR + '/' + path
        # Open the file for reading content
        path = open(filepath, 'rb')
        # Set the mime type
        mime_type,_ = mimetypes.guess_type(filepath)
        # Set the return value of the HttpResponse
        response = HttpResponse(path, content_type=mime_type)
        # Set the HTTP header for sending to browser
        response['Content-Disposition'] = "attachment; filename=%s" % a.name
        # Return the response value
        return response,log
    else:
        # Load the template
        return render(request, 'index.html')
#    return HttpResponse("Form Submitted")

def img_train(file=None, target=None,resize=50):
    #data read
    if file!=None:
        root, ext = os.path.splitext(file)
        print(ext)
        compress_list=[".zip",".tar",".gz",'.tar.gz','.bz2']
        if not ext and ext not in compress_list and target==None:
            print("hy first block is excuting")
            if validate_url(file):
                file=file_from_url(file)
            data,target=check_subfolder_data(file)

        elif ext in compress_list:
            print("hy 2nd block is excuting")
            if validate_url(file):
                file=file_from_url(file)
            file=uncompress_file(file)
            data,target=check_subfolder_data(file)
        else:
            data,target=check_subfolder_data(file)
        data=AutoFeatureSelection.image_processing(data,target,resize)
        log = {
            "Features":["Resize", "Rescale", "Reshape"],
            "Shape":data.shape,
            "Resize":resize,
            "Transformation":[target]
        }
        return data,target,log
    else:
        raise ValueError("{file} can't be null or empty")


def aud(request):
    return render(request,'audio.html')


def extract_features(audio):
  aud, rate = librosa.load(audio, res_type="kaiser_best")
  mfcc_feature = librosa.feature.mfcc(y=aud, sr=rate, n_mfcc=100)
  return np.mean(mfcc_feature.T, axis=0)

# os.listdir("/content/dataset/negatives")

def extract(path, folder_name):
  with ZipFile(path, "r") as obj:
    obj.extractall(path.split(".")[0])
  folder_name = "sih/static/upload/positives/"+folder_name+"/"
  print(folder_name)
  extracted_features=[]
  for i in os.listdir(folder_name):
    extracted_features.append([extract_features(os.path.join(folder_name, i)), str(i)])
  df1 = pd.DataFrame(extracted_features, columns=["Processed", "filename"])
  df1.to_csv("sih/static/media/result.csv")
  return df1
# df = extract("/content/drive/MyDrive/voice_classification.zip", "/content/dataset/negatives")



def audfile(request):
    if request.method=='POST':
        # positive = request.POST['audPositive']
        # negative = request.POST['audNegative']
        dataset = request.FILES['audios']
        folder = request.POST['folder']
        handle_uploaded_file(dataset)

    
        positive = 'sih/static/upload/'+dataset.name

        print(positive)
        df1 = extract(positive, folder)

        #a=request.FILES['aud']
        #alabel=request.FILES['label']
        #targetaud=request.POST['targetaud']
        #print(a)
        #print(a.name)
        #simplefilter(action='ignore', category=FutureWarning)
        #train_path="media/audio_test/"+a.name
        #label_path="media/"+alabel.name

        #print(train_path)
        #print(label_path)
        #train_df=driver(train_path,label_path)
        # print(train_df)
        # train_df.to_csv("media//train.csv")
        # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Define the full file path
        # filepath = BASE_DIR + '//media//' + 'train.csv'
        # # Open the file for reading content
        # path = open(filepath, 'rb')
        # # Set the mime type
        # mime_type, _ = mimetypes.guess_type(filepath)
        # # Set the return value of the HttpResponse
        # response = HttpResponse(path, content_type=mime_type)
        # # Set the HTTP header for sending to browser
        # response['Content-Disposition'] = "attachment; filename=train.csv"
        # # Return the response value
        # return response
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Define the full file path
        filepath = BASE_DIR + '/sih/static/media/result.csv'
        # Open the file for reading content
        path = open(filepath, 'rb')
        # Set the mime type
        mime_type,_ = mimetypes.guess_type(filepath)
        # Set the return value of the HttpResponse
        response = HttpResponse(path, content_type=mime_type)
        # Set the HTTP header for sending to browser
        response['Content-Disposition'] = "attachment; filename=result.csv"
        # Return the response value
        return response
    else:
        # Load the template
        return render(request, 'index.html')












def audio_magic(fileName, path):
    data, _ = librosa.core.load(path + fileName, sr = 44100)
    try:
        mfcc = librosa.feature.mfcc(data, sr = 44100, n_mfcc=20)
        rms = librosa.feature.rms(data)[0]
        chroma_stft = librosa.feature.chroma_stft(data)[0]
        zcr = librosa.feature.zero_crossing_rate(data)[0]
        rolloff = librosa.feature.spectral_rolloff(data)[0]
        spectral_centroid = librosa.feature.spectral_centroid(data)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(data)[0]
        mfcc_trunc = np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1), skew(mfcc, axis = 1), np.max(mfcc, axis = 1), np.median(mfcc, axis = 1), np.min(mfcc, axis = 1)))
        return pd.Series(np.hstack((mfcc_trunc, rms, chroma_stft, zcr, rolloff, spectral_centroid, spectral_bandwidth)))
    except:
        print('Error with audio')
        return pd.Series([0]*20)


def driver(train_path,label_path):
    train_df = pd.DataFrame()
    l_df=pd.read_excel(label_path)
    print(l_df)
    print("inside the driver",train_path)
    train_df['fname']=l_df['fname']
    train_df = train_df['fname'].progress_apply(audio_magic, path=train_path)
    train_df['label']=l_df['label']
    return train_df


def check(request):
    name=request.POST['name']
    email=request.POST['email']
    subject=request.POST['subject']
    message=request.POST['message']
    global e
    e=email
    global a
    a=0
    if(a==0):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('invention.helpdesk@gmail.com', 'umair8999')
        email = EmailMessage()
        email['From'] = 'invention.helpdesk@gmail.com'
        email['To'] = 'hacksspyder@gmail.com'
        email['subject'] = subject
        email.set_content('Name : ' + name + 'Sender email ' + e + 'Query message ' + message)
        server.send_message(email)
        print("yo1")
        a=1
    return HttpResponse ("Thanks Your mail has been sent")


