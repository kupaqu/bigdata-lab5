from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler

def read_csv(path: str, spark: SparkSession) -> DataFrame:
    df = spark.read.csv(path, header=True, inferSchema=True)
    vecAssembler = VectorAssembler(inputCols=df.columns, outputCol='features')
    data = vecAssembler.transform(df)

    return data

def scale(df: DataFrame) -> DataFrame:
    standardScaler = StandardScaler(inputCol='features', outputCol='scaled')
    model = standardScaler.fit(df)
    data = model.transform(df)

    return data

def read_and_scale(path: str, spark: SparkSession) -> DataFrame:
    df = read_csv(path, spark)
    df = scale(df)

    return df