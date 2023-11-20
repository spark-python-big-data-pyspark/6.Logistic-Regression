# 6.Logistic-Regression
1. [Logistic Regression Theory and Reading](#schema1)
2. [Logistic Regression Code Along](#schema2)

<hr>

<a name="schema1"></a>

## 1. Logistic Regression Theory and Reading
![logistic](./img/lor1.png)

![logistic](./img/lor2.png)
![logistic](./img/lor3.png)
![logistic](./img/lor4.png)
![logistic](./img/lor5.png)
![logistic](./img/lor6.png)

- Confusion Matrix
![logistic](./img/lor7.png)
![logistic](./img/lor8.png)
![logistic](./img/lor9.png)
![logistic](./img/lor10.png)
![logistic](./img/lor11.png)
- Model Evaluation
![logistic](./img/lor12.png)
![logistic](./img/lor13.png)


<hr>

<a name="schema2"></a>

## 2. Logistic Regression Code Along

### Categorical Columns 
- Seleccionamos columnas que vamos a usar 
```
 my_cols = df.select(['Survived','Pclass','Sex','Age','SibSp','Parch', 'Fare','Embarked'])
```
- Missing data    

```
my_final_data = my_cols.na.drop()
```
- Crear categorías númericas para categorías con string

```
from pyspark.ml.feature import (VectorAssembler,VectorIndexer,OneHotEncoder,StringIndexer)

gender_indexer = StringIndexer(inputCol='Sex',outputCol='SexIndex')
# A B C
# 0 1 2
# One hot encode
# example A
# [1, 0, 0]
gender_encoder = OneHotEncoder(inputCol='SexIndex', outputCol= 'SexVec')

embark_indexer = StringIndexer(inputCol='Embarked',outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex',outputCol='EmbarkVec')
```






