## ML_OPS-Banking-Сlustering-and-Classification

***conda env create -f environment.yml не находит библиотеки на Linux - проблемы с GitLab***

***Задача***: кластеризация клиентов банка на 3-4 кластера и 
        создание приложения с моделью классификации клиентов по 
        выявленным кластерам.


***Описание содержимого проекта и хода решения***:

1. BI. APACHE-SUPERSET

   Можно выделить несколько этапов:
    
         - 1 Установка и настройка Superset для Windows:
                с помощью Docker были запущены контейнеры с Superset и PostgresSQL в WSL. В Superset была
                подключена база данных PostgresSQL и активирована возможность загрузки .csv

                docker, WSL, PostgresSQL

        - 2 Создание Dashboard:
                были загружены данные в формате .csv из каталога data/. На основе загруженных таблиц
                были построены графики отражающие особенности и зависимости в данных. В случаях, когда
                данных исходных тблиц не хватало, были реализованы SQL-запросы, которые помогли получить
                недостающую информацию

                charts (linechart, barchart, piechart, radarchart), SQL (case, window functions)

                

2. ИССЛЕДОВАНИЕ ДАННЫХ, КЛАСТЕРИЗАЦИЯ, КЛАССИФИКАЦИЯ

    Этапы данного раздела описаны в 5-ти ноутбуках в каталоге notebooks/

        - 1_data_analysis:
                визуализация данных c data/raw, создание признаков, формулировка предположений о
                важности признаков и о целесообразности их исспользования, выявление ненужных признаков

                plotly

        - 2_unknown_values_processing:
                определение пропущенных значений и формирование датасетов с разными
                способами заполнения пропусков. Данные хранятся в data/train_test

                KNNImputer, IterativeImputer ( BayesianRidge )

        - 3_preprocessing:
                определение 8 конвееров предобработки (масштабирование, кодирование)
                данных: 2 способа масштабирования числовых признаков (MinMax, Z-scaling),
                2 способа кодировки категориальных признаков (Original, OneHot),
                2 способа масштабирования закодированных категориальных признаков (MinMax, Z-scaling).
                Всего получено 24 набора данных (по 8 для каждого из 3-ех способов заполнения пропусков),
                в каждом тренироваочный и тестовый набор - итого 48 файлов. Файлы хранятся в data/preprocessed

                Pipeline, ColumnTransformer

        - 4_clustering:
                для каждого из полученных ранее наборов данных была произведена кластеризация методом K-means
                на 2,3,4,...,15 кластеров для всего набора признаков и без признаков социально-экономической
                обстановки. Для каждого случая были отрисованы и сохранены сборные графики с основными метриками
                клстеризациии - всего 48 сборных графиков, они сохранены в каталоги images/num_of_clusters_with_soc_econ_factors
                и images/num_of_clusters_without_soc_econ_factors соответственно. После анализа полученных графиков, было решено,
                что лучшая кластеризация при KNN-методе обработки пропксков и 8-ом конвеере предобработки (MinMax, Original, MinMax).
                После этого призведён анализ признаков полученных кластеров при кластеризации с использованием признаков социально-
                экономической обстановки и без них. В ходе анализа было решено отбросить признаки социально-экономической
                обстановки и ряд других признаков, так так они совпадали по значенияям во всех кластерах, т.е. являлись ненужными.
                В итоге разбил на 4 кластера по 3-ём признакам с Silhouette score равным 0.63, вместо изначального 0.28, при
                кластеризации на всех признаках. Некоторые график сохранены в /images/clusters

                TSNE, KMeans, silhouette_score, Elbow method, create_dendrogram, Scatterpolar, scatter_3d, make_subplots, fig.write

        - 5_classification_model:
                на этапе кластеризации стало понятно, что классификация будет максимально простой, так так данные визуально 
                полностью разделялись в итоговом 3-ёх мерном пространстве job, housing, loan. Были построены модели Логистической 
                регресии для несбалансированных классов и Решающее дерево глубиной 3. Решающее дерево отлично формализовало
                алгоритм классификации и ,исходя из природы ограниченности итоговых признаков, оно идеально подходит для 
                классификации в данном проекте. Сохранённая модель и изображение дерева сохранены в /models

                accuracy_score, confusion_matrix, ConfusionMatrixDisplay, plot_tree, pickle


        - интерпритация полученных кластеров:
                - Кластер 1: клиенты имеют кредиты на жильё и в основном имеют работу admin, blue-color, реже management и т.д.;
                        кредиты на жильё их наврядли интересуют, можно предлагать обычный кредит в соответствии с моделью кредитного скоринга

                - Кластер 2: клиенты имеют непогашенные кредиты;
                        можно предлагать выкуп их кредита или новый кредит в соответствии с моделью кредитного скоринга

                - Кластер 3: клиенты не имеют ни непогашенные кредиты, ни кредитов на жильё;
                        можно предлагать любые продукты

                - Кластер 4: клиенты имеют кредиты на жильё и в основном имеют работу technician, services, реже self-employed и т.д.;
                        кредиты на жильё их наврядли интересуют, можно предлагать обычный кредит в соответствии с моделью кредитного скоринга

                Размеры кластеров:
                    Кластер 1:  0.26
                    Кластер 2:  0.15
                    Кластер 3:  0.43
                    Кластер 4:  0.16

                Можно сильно улучшить функциональность модели, если узнать об уровнях дохода клиентов.

            

3. СОЗДАНИЕ ПРИЛОЖЕНИЯ
    
    Данный раздел можно разбить на несколько этапов

        - 1 Выбор библиотеки для реализации:
                было принято использовать библиотеку streamlit, так как она предоставляет возможность
                быстрого и качественного создания data-app

                streamlit as st

        - 2 Создание интерфейса для работы с приложением:
                были созданы поля для ввода необходимой информации разных типов, созданы разделы, 
                описывающие кластеры, модель, ход работы и исходные данные. Были интегрированы изображения
                plotly, matplotlib и текстовые файлы, некоторые файлы продублированы и сохранены в каталог /app

                pickle, st.number_input, st.selectbox, st.header, st.markdown, st.checkbox, st.write, st.button

        - 3 Предобработка введённых данных:
               изначально требуется ввести не менее 2-ух признаков, из которых 2 должны быть и важных (job, housing, loan)
               сложность реализации заключалась в том, что вводятся не все признаки, которые учавствовали при
               предобработке данных в процессе создания модели, поэтому пришлось приводить введённые данные к 
               нужному для предобработки формату, для предобработки использовались scaler, encoder, knn_imputer 
               из ноутбука 2, а так же piprline preproc8 из ноутбука 3 - они были сохранены в виде .pkl файлов 
               и использованы в финальном приложении непосредственно

               pickle, pandas

        -4 Интеграция модели:
                модель хоть и простая, но её интеграция от этого легче не становится, за счёт сложности с 
                обработкой введённых данных. Сама модель была так же восстановлена из pickle файла

                pickle, predict



***Итог***: проведено BI исследование данных, на основе которого было создано приложение,
      ноутбуки с исследованием и само приложение собрано в проект, произведена 
      кластеризация на 4 кластера и построена модель классификации для распределения
      клиентов по кластерам, всё это завёрнуто в приложение.

## Про папки проекта:

- BI - хранятся результаты BI исследования
- app - хранятся ВСЕ файлы необходимые для работы ПРИЛОЖЕНИЯ
- data - хранятся исходные данные и промежуточные результаты предобработок
- images - хранятся графики полученные в ходе исследования
- models - хранится модель и её ОПИСАНИЕ
- notebooks - хрянятся ноутбуки описывающие ход исследования
- pipelines - хранятся предобработчики данных для разных случаев


