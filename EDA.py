from asyncio.windows_utils import pipe
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from yellowbrick.classifier import confusion_matrix
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from lime import lime_text
from sklearn.preprocessing import FunctionTransformer


# modulos propios
from clean.clean_text import TicketMessage, stop_words
from train_model import SelectModel


train = pd.read_csv("data/customer-issues-train.csv",
        usecols=["date-received", "product", "sub-product", "issue","sub-issue",
                "consumer-message", "state","complaint-id"],
        sep=",",parse_dates=["date-received"])

test = pd.read_csv("data/customer-issues-test.csv", sep=",",
                    usecols=['date-received', 'consumer-message',
                            'state','complaint-id'],
                    parse_dates=["date-received"])


# agregar id_dataset
train["type"] = "train"
test["type"] = "test"

# número de registros en train y test
print(f"registros en train {train.shape[0]:,}" )
print(f"registros en test {test.shape[0]:,}" )
print(f"total de registros {test.shape[0] + train.shape[0]:,}" )


# unir ambos datasets para hacer más fácil su analisis
merged_data = pd.concat([train[["date-received", "consumer-message", "state", "complaint-id", "type","product"]],
                        test[["date-received", "consumer-message", "state","complaint-id", "type"]]])

merged_data.reset_index(drop=True, inplace=True)
# validar que solo ids unicos
merged_data["complaint-id"].shape[0] == len(merged_data["complaint-id"].unique())


# distribucion en el tiempo de los tickets
tickets_by_date = merged_data.groupby(pd.Grouper(key='date-received', freq='1M'))["complaint-id"].count()
fig = px.line(x=tickets_by_date.index, y=tickets_by_date.values)
fig.show()

# tenemos datos desde 2015 al 2017


# distrubucion de tickets por estado
tickets_by_state = merged_data.groupby(["state"], as_index=False)\
        .agg(tickets_number=("complaint-id",  "count"))

tickets_by_state = tickets_by_state.sort_values(by="tickets_number", ascending=False)

category_fig = px.bar(tickets_by_state,
                x='state', y='tickets_number', color='tickets_number')

category_fig.show()

# la mayoria de lo tickets se concentra en California

# limpiar textos
# para la limpieza del texto se creo la clase TicketMessage
# con el obejtivo de usarla mas adelante
# usar pandarrel dado que el apply solo usa 1 nucleo es muy lento
#pandarallel.initialize(progress_bar=True)
merged_data["clean_message"] = merged_data["consumer-message"].apply(TicketMessage.parse_text)
merged_data["len_clean_message"] = merged_data["clean_message"].str.len()
#merged_data[["complaint-id", "clean_message"]].to_csv("review.csv")

# conteo de palabras en texto libre
merged_data["clean_words_number"] = merged_data["clean_message"].apply(lambda x: len(x.split()))
merged_data["raw_words_number"] = merged_data["consumer-message"].apply(lambda x: len(x.split()))


# podemos omitir registros con 3 o menos palabra
# aportan poca informacion al modelo

print("Número de registros omitidos",
        merged_data.loc[merged_data["clean_words_number"] < 4].shape[0])

merged_data = merged_data.loc[merged_data["clean_words_number"] >= 4]

# cantidad de palabras por ticket
lenght_hist = px.histogram(merged_data, x="raw_words_number",
                           title="Palabras en mensaje sin limpiar")
lenght_hist.show()


lenght_hist = px.histogram(merged_data, x="clean_words_number",
                           title="Palabras en mensaje limpios")
lenght_hist.show()


# longitud promedio
merged_data["clean_words_number"].mean()
# longitud minima
merged_data["clean_words_number"].min()
# longitud maxima
merged_data["clean_words_number"].max()
# mediana
merged_data["clean_words_number"].median()
# percentil 90
merged_data["clean_words_number"].quantile(q=0.9)

# tamaño vacabulario sin limpiar
def word_number(column: pd.Series) -> Counter:
    word_count = column.values
    word_count = ' '.join(word_count)
    word_count = word_count.split()
    return Counter(word_count)        


raw_word_count = word_number(merged_data["consumer-message"])
print(f"tamaño vacabualario sin procesar {len(raw_word_count):,}" )

clean_word_count = word_number(merged_data["clean_message"])
print(f"tamaño vacabualario con texto limpio {len(clean_word_count):,}" )

# grafico de frecuancia de palabras
clean_word_count = pd.DataFrame.from_dict(clean_word_count, orient="index")
clean_word_count = clean_word_count.sort_values(by=0, ascending=False)
clean_word_count.columns = ["clean_words_number"]
clean_word_count["word"] = clean_word_count.index

# top 200 palabras
words_fig = px.bar(clean_word_count[:200],
                x='word', y='clean_words_number', color='clean_words_number',
                title="Frecuencia de palabras más comunes")
words_fig.show()

# palabras poco comunes
words_fig = px.bar(clean_word_count.tail(200),x='word',
                   y='clean_words_number', color='clean_words_number',
                   title="Frecuencia de palabras poco comunes")
words_fig.show()

# top 100 palabras
word_cloud = WordCloud(background_color="white",
                        max_words=100).generate(' '.join(merged_data["clean_message"].values))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# palabras más frecuentes por producto       


# datos por categoria en train
tickets_by_category = merged_data[merged_data["type"] == "train"]\
        .groupby(["product"], as_index=False, dropna=False)\
        .agg(tickets_number=("complaint-id",  "count"))

category_fig = px.bar(tickets_by_category,
                x='product', y='tickets_number', color='tickets_number',
                title="Ticktets por categoria")

category_fig.update_layout(xaxis={'categoryorder':'total descending'}) 
category_fig.show()

tickets_by_category["tickets_number%"] = tickets_by_category["tickets_number"] / tickets_by_category["tickets_number"].sum()
# tenemos muchas clases en product, además de un datos no balanceados
# hay clases con muy pocos registros
# se pueden aplicar tecnicas de sub o sobre muestreo
# para facilitar el trabajo, podemos combinar varias clases en 1 sola
merged_data["new_product"] = merged_data["product"].map({
        'Debt collection': 'Debt collection', 
        'Mortgage':'Mortgage', 
        'Credit reporting': 'Credit reporting',
        'Credit card': 'Credit card',
        'Bank account or service': 'Bank account or service',
        'Consumer Loan': 'Consumer Loan', 
        'Student loan':'Student loan',
        'Payday loan': "Others",
        'Money transfers': "Others", 
        'Prepaid card': "Others",
        'Other financial service': "Others",
        'Virtual currency': "Others"})

tickets_by_category = merged_data[merged_data["type"] == "train"]\
        .groupby(["new_product"], as_index=False, dropna=False)\
        .agg(tickets_number=("complaint-id",  "count"))

category_fig = px.bar(tickets_by_category, 
                x='new_product', y='tickets_number', color='tickets_number',
                title="Ticktets por xategorías, despues de agrupar categorías")

category_fig.update_layout(xaxis={'categoryorder':'total descending'}) 
category_fig.show()


for _ in merged_data[merged_data["type"]=="train"]["new_product"].unique():
    products = merged_data.loc[merged_data["new_product"] == _]["clean_message"].values

    word_cloud = WordCloud(background_color="white",
                          width=800,height=600,
                           collocations=False,
                           colormap="plasma",
                           max_words=100).generate(' '.join(products))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(_.upper())
    plt.show()
    del products, word_cloud


#guardar dataset limpios
for _ in ["train", "test"]:
    merged_data.loc[merged_data["type"] == _,
           ["clean_message", "product", "complaint-id", "new_product", "type"]
           ].to_csv(f"data/clean_text/{_}.csv", index=False)


merged_data = pd.read_csv("data/clean_text/train.csv")

# crear embedings
# usar tf-idf
# usando unigramas
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), 
                                   max_df=0.95)
# crear matrix tfidf
tfidf = tfidf_vectorizer.fit_transform(merged_data[merged_data["type"]=="train"]["clean_message"])
merged_data[merged_data["type"]=="train"].shape
tfidf.shape

X_train, X_test, y_train, y_test = train_test_split(tfidf, 
                                        merged_data[merged_data["type"]=="train"]["new_product"], 
                                        test_size=0.2, random_state=10)



# compara varios modelos para ver el es mejor estimador para nuestros
# se comparan:
# * regresión logistca multinomial
# * Naive Bayes
# * Radon Forest
# * KNN
# * Extra trees

models = SelectModel(X_train, y_train)
models.evaluate_models()
print(models)

# como se observa el mejor estimador es la regresion logistca

# vamos entrenar nuevamente el modelo sin usar CV
lr_model = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs', max_iter=500)\
                              .fit(X_train, y_train)

lr_model.get_params()
lr_model.predict_proba
# calculo f score en test_data
y_pred = lr_model.predict(X_test)
base_line_score = f1_score(y_test,y_pred, average="macro")
print(base_line_score)

# matriz de confusion
mlg_cm = confusion_matrix(
    estimator=lr_model,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    percent=True,
    is_fitted=True)

mlg_cm = confusion_matrix(
    estimator=lr_model,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    is_fitted=True)


# Evaluames de manera indivual cada categoria
print(classification_report(y_pred, y_test))

# como se aprecia el modelo funciona bastente bien para las clases
# Bank account or service
# Credit card
# credit reporting
# debt collection
# student loan
# especialmente para mortage
# pero como era de esperarse no es muy bueno para clasificar constumer Loan y others
# devido a nuestro problema de desbalanceo

param_grid = {
     'penalty' : ['l2'], 
     'C' : [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]}

logreg_cv = GridSearchCV(LogisticRegression(multi_class='multinomial', solver="lbfgs", max_iter=500),
                       param_grid=param_grid, cv=3,
                        scoring='f1_macro')

logreg_cv.fit(X_train, y_train)

logreg_cv.best_params_
logreg_cv.best_score_

logreg_cv.cv_results_
logreg_cv.best_params_
logreg_cv.best_estimator_

tuned_score = f1_score(y_test, logreg_cv.best_estimator_.predict(X_test), average="macro")
tuned_score > base_line_score

# no hay una mejora probando valores distintos en la regaularizacion
# podemos dejar nuentros primer modelo como nuestro mejor estimador



classes = list(merged_data[merged_data["type"]=="train"]["new_product"].unique())



# creamos pipelin para prediccion
pre_process = FunctionTransformer(TicketMessage.parse_text)
pipe_model = make_pipeline(tfidfs_vectorizer, lr_model)



# 
explainer = LimeTextExplainer(class_names=list(pipe_model.classes_))
exp = explainer.explain_instance(merged_data["clean_message"][0],
                                pipe_model.predict_proba, 
                                num_features=5,
                                top_labels=1)

exp.show_in_notebook()