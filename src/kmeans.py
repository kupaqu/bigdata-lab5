import configparser

from typing import Dict

from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from logger import Logger
from preprocess import read_and_scale

SHOW_LOG = True

class KmeansEvaluator:
    def __init__(self):
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self._evaluator = ClusteringEvaluator(featuresCol='scaled')
        
        self.log.info('KmeansEvaluator initialized.')

    def fit_predict(self, df: DataFrame) -> Dict:
        scores = {}
        for k in range(2, 8):
            kmeans = KMeans(featuresCol='scaled', k=k)
            model = kmeans.fit(df)
            preds = model.transform(df)

            scores[k] = self._evaluator.evaluate(preds)

        self.log.info('Clustering evaluated.')

        return scores

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    spark = SparkSession.builder \
        .appName(config['spark']['app_name']) \
        .master(config['spark']['deploy_mode']) \
        .config('spark.driver.cores', config['spark']['driver_cores']) \
        .config('spark.executor.cores', config['spark']['executor_cores']) \
        .config('spark.driver.memory', config['spark']['driver_memory']) \
        .config('spark.executor.memory', config['spark']['executor_memory']) \
        .getOrCreate()

    data_path = config['data']['openfoodfacts']
    df = read_and_scale(data_path, spark)

    kmeans = KmeansEvaluator()
    scores = kmeans.fit_predict(df)

    for k, v in scores.items():
        print(f'{k}: {v}')

    spark.stop()