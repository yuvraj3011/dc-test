import os,cv2
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold,SelectKBest,f_regression,f_classif
from sklearn.preprocessing import  MinMaxScaler
from statistics import mean
#from . import  dataCleaner
from sih.Cleaner import dataCleaner
global dic
dic = {}

def logs(dic, key, value):
    dic[key]=value

class AutoFeatureSelection: 
    
    def dropHighCorrelationFeatures(X):
        cor_matrix = X.corr()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        if to_drop!=[]:
            logs(dic, "High Correlation Features", to_drop)
            return X.drop(to_drop, axis=1)
        else: return X
            
    def dropConstantFeatures(X):
        cols=X.columns
        constant_filter = VarianceThreshold(threshold=0).fit(X)
        constcols=[col for col in cols if col not in cols[constant_filter.get_support()]]
        if(constcols!=[]): X.drop(constcols,axis=1,inplace=True)
        logs(dic, "Contant Features", X.columns)
        return X

    def GetAbsoluteList(resdic,dataframe,impmain,dict_class):
        keylist=[]
        imp_dict={}
        for key, value in resdic.items():
            if value < 0.01: 
                for key_2 in impmain.keys():
                    if key in key_2:
                        keylist.append(key_2)
            else: imp_dict[key]=value
            
        result_df=dataframe.drop(keylist,axis=1)
        logs(dic, "Dropped Column", keylist)
        dict_class.feature_importance=imp_dict
        return result_df

    def FeatureSelection(dataframe,target):
        df=dataCleaner(dataframe,dataframe.drop(target,axis=1).columns.to_list(),target)
        #score_func=f_classif if(dict_class.getdict()['problem']["type"]=='Classification') else f_regression
        X=df.drop(target,axis=1)
        Y=df[target]
        X=AutoFeatureSelection.dropConstantFeatures(X)
        X=AutoFeatureSelection.dropHighCorrelationFeatures(X) #if not disable_colinearity else X
        #X=AutoFeatureSelection.get_feature_importance(X,Y,score_func,dict_class)
        #featureList=AutoFeatureSelection.getOriginalFeatures(X.columns.to_list(),dict_class)
        #dict_class.addKeyValue('features',{'X_values':featureList,'Y_values':target})
        return X,Y

    def getOriginalFeatures(featureList,dict_class):
        if(dict_class.ObjectExist):
            res,res2= [],[]#List
            for val in featureList: #filter for String categorical field existence.
                if not any(ele+"_" in val for ele in dict_class.ObjectList): res.append(val)
            res=res+dict_class.ObjectList
            for v in res:# filter for Selected String categorical
                if not any (v in ele for ele in featureList): res2.append(v)
            # filter to consider the features
            res3=[i for i in res if i not in res2]
            return res3 
        else: return featureList

    def image_processing(data,targets,resize):
        training_data,label_mapping=AutoFeatureSelection.create_training_data(data,targets,resize)
        original_label=label_mapping
        original_shape=[len(training_data)]
        original_shape.extend(list(training_data[0][0].shape))
        original_shape=original_shape
        df = pd.DataFrame(training_data, columns=['image', 'label'])
        logs(dic, "Image Shape", (resize, resize))
        return df

    def create_training_data(data,target,resize):
        training_data=[]
        label_mapping={}
        for category in target:
            path=os.path.join(data, category)
            class_num=target.index(category)
            label_mapping[class_num]=category
            for img in os.listdir(path):
                try:
                    img_array=cv2.imread(os.path.join(path,img))
                    img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
                    new_array=cv2.resize(img_array,(resize,resize))
                    training_data.append([new_array,class_num])
                except Exception as e:
                    pass
        return (training_data,label_mapping)

    def get_reshaped_image(training_data):

        lenofimage = len(training_data)
        X, y = [], []
        for categories, label in training_data:
            X.append(categories)
            y.append(label)
        X = np.array(X).reshape(lenofimage,-1)
        y = np.array(y)
        return (X,y)

    def val(a):
        print(a)
        print(dic)
        return dic
