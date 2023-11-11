import pandas as pd
import streamlit as st
from mylib import *



# оформление страницы
st.title("ML Application 🤖")

st.header('Классфификация клиента')
st.markdown('Введите данные клиента, обязательно заполните 2 поля с 🚨 - они очень важны:')

# секция ввода данных
job = st.selectbox("***🚨Работа***",categs['job'] ,index=unknown_index['job'])
housing = st.selectbox("***🚨Есть ли кредит на жильё***",categs['housing'],index=unknown_index['housing'])
loan = st.selectbox("***🚨Есть ли непопогашенный кредит (не на жильё)***", categs['loan'],index=unknown_index['loan'])
education = st.selectbox("Образование", categs['education'],index=unknown_index['education'])
marital = st.selectbox("Семейное положение", categs['marital'],index=unknown_index['marital'])
default = st.selectbox("Был ли отказ от выплаты кредита", categs['default'],index=unknown_index['default'])
age = st.number_input(label="Возраст",min_value=18,max_value=100)

# вывод введённых данных
st.markdown('Введённые данные:')
client=pd.DataFrame(columns=all_columns)
client.loc['Клиент']=[job,housing,loan,education,marital,default,age]
st.dataframe(client,width=1500)

# обработка нажатия на кнопку, классификация
if st.button('Классифицировать'):
    if (job=='unknown')+(housing=='unknown')+(loan=='unknown')>1:
        st.markdown('Введите минимум два поля с 🚨')
    else:
        # запуск алгоритма обработки пропусков и предобработки
        mess, client_data, client_info=check_main_na(client)

        # вывод предобработанных данных
        st.markdown('Данные для модели:')
        st.markdown(mess)
        st.dataframe(client_info,width=1500)
        st.markdown('Данные, с которыми непосредственно работает модель:')
        st.dataframe(client_data,width=1500)

        # классфикация
        label=model.predict(client_data[main_columns])[0]
        st.markdown('')
        st.markdown('')
        st.markdown(f'***Модель опредилила данного клиента к классу {get_label_sym(label)}***')
        st.markdown('Откройте раздел *Описание кластеров* для ознакомеления с описанием данного класса.')

else:
    st.markdown('Заполните поля и нажмите на кнопку для выполения классификации.')


# описание классов и данных
st.header('Отображение разделов')
clusters_info = st.checkbox('Описание кластеров')
model_info = st.checkbox('Описание модели')
work_info = st.checkbox('Описание проделанной работы')
data_info = st.checkbox('Описание исходных данных')

if clusters_info:
    st.header('Описание классов клиентов')
    st.markdown("""          
    Модель распеделяет клиентов банка по четырем классам (также указаны возможые продукты): 
                    
    - ***Класс 1***: клиенты имеют кредиты на жильё и в основном имеют работу admin, blue-color, реже management, housemaid и entrepreneur;
                     
                Страхование жилья, Программы по погашению кредитов на жильё, Ипотечные программы с фиксированными процентными ставками
    - ***Класс 2***: клиенты имеют непогашенные кредиты;
                     
                Программы консолидации долгов, Уведомления о платежах, Планы отсрочки платежей, Программы рефинансирования
    - ***Класс 3***: клиенты не имеют ни непогашенные кредиты, ни кредитов на жильё;
                     
                Бесплатные счета, Программы по сбережению, Кэшбэк и награды, Инвестиционные продукты, Цифровые финансовые инструменты
    - ***Класс 4***: клиенты имеют кредиты на жильё и в основном имеют работу technician, services, реже self-employed, retired, unemployed и student ; 
                     
                Программы для безработных и студентов, Программы для пенсионеров, Ипотечные страховки, Ипотечные программы
                
    Эти калассы были получены при помощи кластеризации методом K-means. Было проведено исследование, в котором были определены три признака, по которым клиенты отлично делятся на 4 кластера, которые хорошо интерпретируются как целевые группы для определённого банковского продукта.            
    """)

    st.markdown("""
                
    Ниже показано как классы клиентов расположены в итоговм трёхмерном пространстве, можно легко их разделить визуально. 
    """)
    st.write(load_object('d:\\BankProject1\\app\\scatter3d.pkl'))
    st.markdown("""
                
    Ниже представлена радиальная диаграмма средних значений для каждого признака внутри каждого кластера. Она помогает при анализе качества кластеризации и интерпритации полученных кластеров. 
    """)
    st.write(load_object('d:\\BankProject1\\app\\final_polar_plot.pkl'))

if model_info:
    st.header('Описание модели')
    st.markdown(""" 
    Ниже отрисована структура Решающего дерева, отвечающего за классификацию клиентов по выявленным классам. 
    """)
    st.write(load_object('d:\\BankProject1\\app\\model_structure.pkl'))

if work_info:
    st.header('Описание проделанной работы')
    with open('d:\\BankProject1\\app\\summary.txt', encoding="utf8") as f:
        st.markdown(f.read())

if data_info:
    st.header('Описание исходных данных')
    with open('d:\\BankProject1\\app\\raw_data_description.txt', encoding="utf8") as f:
        st.markdown(f.read())


st.header('Ссылки')
st.markdown("""
- *GitHub*: https://github.com/batchatilionus
- *Telegram*: https://t.me/batchatilion
- *LinkedIn*: https://www.linkedin.com/in/nikita-kuchko-249a4a290
""")