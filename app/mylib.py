import pandas as pd
import pickle
import streamlit as st

@st.cache_data 
def load_dataset(path:str):
    dataset = pd.read_csv(path)
    return dataset

@st.cache_data 
def load_object(path:str):
    with open(path, 'rb') as f:
        return pickle.load(f)

DF = load_dataset('d:\\BankProject1\\data\\train_test\\train.csv')
sel_columns=['job','housing','loan','education','marital','default']
all_columns=sel_columns+['age']

categs={}
unknown_index={}
for col in sel_columns:
    categs[col]=sorted(DF[col].unique())
    unknown_index[col]=categs[col].index('unknown')

main_columns=['loan','housing','job']
numeric_subset=list(DF.select_dtypes(['float64','int64']).columns)
categoric_subset=list(DF.select_dtypes('object').columns)
encoder=load_object('d:\\BankProject1\\app\\encoder.pkl')
scaler=load_object('d:\\BankProject1\\app\\scaler.pkl')
knn_imputer=load_object('d:\\BankProject1\\app\\knn_imputer.pkl')
pipe=load_object('d:\\BankProject1\\app\\knn_preproc8.pkl')
model=load_object('d:\\BankProject1\\models\\model.pkl')


def get_label_sym(label):
    syms=['1️⃣','2️⃣','3️⃣','4️⃣']
    for i in range(4):
        if label==i:return syms[i]


def check_main_na(client:pd.DataFrame):

    # приведение данных к шаблону на котором проводилось заполнение пропусков
    templ=pd.DataFrame(columns=numeric_subset+categoric_subset)
    temleted_client=[]
    for col in templ.columns:
        if col in client.columns:
            temleted_client.append(client[col].iloc[0])
        else:
            temleted_client.append(None)
    templ.loc[0]=temleted_client

    # определение пропусков при вводе
    bl=False
    for col in main_columns:
        if client[col].iloc[0]=='unknown':
            bl=True
            break
    if bl:
        # заполнение пропусков

        # алгоритм, который исспользовался в ноутбуке 2
        for col in categoric_subset:
            templ[col]=templ[col].replace('unknown',None)

        templ_num=templ[numeric_subset]
        templ_cat=templ[categoric_subset]

        templ_num_scaled=pd.DataFrame(scaler.transform(templ_num))
        templ_cat_encoded=pd.DataFrame(encoder.transform(templ_cat),columns=categoric_subset)

        # определение значений, которыми закодировался None
        none_kodes={}
        for col in categoric_subset:
            idx=list(encoder.feature_names_in_).index(col)
            none_kodes[col]=len(encoder.categories_[idx])-1

            # в этих колонках нет None
            if col in ['poutcome','subscribed']:
                none_kodes[col]=-1
        for col in categoric_subset:
            templ_cat_encoded[col]=templ_cat_encoded[col].replace(none_kodes[col],None)

        # сбор предобработанных данных
        temple_encode=pd.concat([templ_num_scaled,templ_cat_encoded],axis=1,ignore_index=True)
        temple_encode=pd.DataFrame(temple_encode)
        temple_encode.columns=numeric_subset+categoric_subset

        # заполенение пропусков
        templ_imputed=pd.DataFrame(knn_imputer.transform(temple_encode),columns=temple_encode.columns)

        # раскодировка катeгориальных и числовых признаков
        client_num_imputed=pd.DataFrame(scaler.inverse_transform(templ_imputed[numeric_subset]))
        client_cat_imputed=pd.DataFrame(encoder.inverse_transform(templ_imputed[categoric_subset]))
        client_imputed=pd.concat([client_num_imputed,client_cat_imputed],axis=1,ignore_index=True)
        client_imputed=pd.DataFrame(client_imputed)
        client_imputed.columns=numeric_subset+categoric_subset

        # выборка нужных для классификации признаков
        client_main_imputed=client_imputed[main_columns]

        # форматирование признаков для модели
        client_data=pd.DataFrame(pipe.transform(client_imputed),columns=numeric_subset+categoric_subset)[main_columns]
        client_info=client_main_imputed
        mess="Недостающие важные данные заполнены методом ближайших пяти соседей."
    else:
        # форматирование признаков для модели
        client_data=pd.DataFrame(pipe.transform(templ),columns=numeric_subset+categoric_subset)[main_columns]

        # выборка нужных для классификации признаков
        client_info=client[main_columns]
        mess="Все важные данные введены."
    
    client_data.index=['Клиент']
    client_info.index=['Клиент']

    return mess,client_data,client_info