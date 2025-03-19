from pyspark.sql.types import DecimalType, DoubleType, LongType, StringType, NumericType
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window
import time
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score



KEY_TYPE = "integer"
TYPE_MAPPING = {
    "long": "int64",
    "integer": "int32",
    "short": "int16",
}

def get_idx_list(spark):
    
    num_shuffle_parts = int(spark.conf.get("spark.sql.shuffle.partitions"))
    return (spark
        .range(1_000_000, numPartitions=20)
        .select(
            F.col("id").cast(KEY_TYPE)
        )
        .select(
            F.col("id"),
            F.when(
                F.hash("id") % num_shuffle_parts >= 0,                  
                F.hash("id") % num_shuffle_parts
            ).otherwise(
                F.hash("id") % num_shuffle_parts + num_shuffle_parts
            ).alias("mod")
        )
        .select(
            F.col("id"),
            F.row_number().over(Window.partitionBy("mod").orderBy("id")).alias("rn")
        )
        .where(
            F.col("rn") == 1
        )
        .rdd.map(
            lambda row: row["id"]
        )
        .collect()
    )

def get_grouping_key(idx_list):
    @F.pandas_udf(KEY_TYPE)
    def get_grouping_key_udf(any_field):
        seed = int(time.time() * 10**6 % 1993)
        random.seed(seed)
        batch_size = any_field.shape[0]
        idx = random.choices(idx_list, k=batch_size)
        return pd.Series(idx, dtype=TYPE_MAPPING[KEY_TYPE])
    return get_grouping_key_udf

def predict(model, features_list, features_list_float , features_list_string, num_classes, date_col, subs_id_col, target_col, columns_to_drop, schema):

    @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
    def predict_udf(feats: pd.DataFrame) -> pd.DataFrame:
        X_float = pd.DataFrame(
            feats["features_float"].tolist(), 
            columns=features_list_float,
        ) #.fillna(0)

        X_string = pd.DataFrame(
            feats["features_string"].tolist(), 
            columns=features_list_string,
        ) #.fillna(0)

        X = pd.concat([X_float, X_string], axis = 1)

        X = X[features_list]

        X_test = X.drop(columns = columns_to_drop)



        pred = pd.DataFrame(
            model.predict_proba(X_test), 
            columns=[f"class_{i + 1 - 1}" for i in range(num_classes)],
        ).astype("float32")

        predicted = pd.DataFrame(
            model.predict(X_test), 
            columns=["predicted"],
        ).astype("int")

        result_df = pd.concat(
            objs=[
                feats.loc[:, [date_col, subs_id_col, 'true_class']], 
                pred, 
                predicted
            ], 
            axis=1,
        )

        return result_df
    return predict_udf



def scoring_func(df,  model,  num_classes, spark, idx_list_broadcasted,  date_col, subs_id_col, target_col, columns_to_drop, schema):
    
    num_shuffle_parts = int(spark.conf.get("spark.sql.shuffle.partitions"))
    
    features_list = df.columns
    idx_list = idx_list_broadcasted.value

    df0 = df.select('*')
    df0 = df0.withColumn(date_col, F.col(date_col).cast('string'))
    

 
    schema_df0 = df0.schema


    features_list_string = [field.name for field in schema_df0.fields if isinstance(field.dataType, StringType)]

    features_list_float = [field.name for field in schema_df0.fields if isinstance(field.dataType, NumericType)]

    get_grouping_key_udf = get_grouping_key(idx_list)
    
    result_df = (df0
                 
                 
        .select(
            F.col(date_col), 
            F.col(subs_id_col), 
            F.col(target_col).alias("true_class"),
            F.array(features_list_float).alias("features_float"),
            F.array(features_list_string).alias("features_string"),
            get_grouping_key_udf(subs_id_col).alias("key")            
        )
       .repartition(num_shuffle_parts, "key")
       .groupBy("key")
        .apply(
            predict(
                model = model,
                features_list = features_list,
                features_list_float = features_list_float,
                features_list_string = features_list_string,
                num_classes = num_classes,
                date_col = date_col,
                subs_id_col = subs_id_col,
                target_col = target_col,
                columns_to_drop = columns_to_drop, 
                schema = schema
            )
        )            
    )
    return result_df


def reports(target: dict, models_d:dict, perc:list, pred_proba:pd.core.frame.DataFrame, y_test: pd.core.frame.DataFrame  ) -> pd.core.frame.DataFrame:
    
    pred_proba = pred_proba.values
    #код для расчет таблицы с метриками
    final_full = pd.DataFrame()
    for w_key, w_value in target.items():
        week_row = pd.DataFrame()
        for m_key, m_value in models_d.items():
            perc_row = pd.DataFrame()
            in_keys = [f'{i}_in' for i in target] + [f'{i}_in_perc' for i in target]
            keys = ['precision', 'recall', 'f1', 'roc-auc', 'pr-auc', 'count', 'proba'] + in_keys
            metrics = dict.fromkeys(keys)
            index = pd.MultiIndex.from_product([[m_key], metrics.keys()], names=['model', 'metrics'])


            #здесь y_val это series с таргетом
            test = np.where(y_test.values == w_key, 1, 0)
            #pred_proba - двумерный массив с probability для каждого класса. Обычно такой выдается после model.predict_proba
            proba = pred_proba[:, w_value]


            for p in perc:
                th = np.percentile(proba, 100 - p)
                y_th_pred = np.where(np.array(proba) >= th, 1, 0)

                metrics['precision'] = precision_score(test, y_th_pred)
                metrics['recall'] = recall_score(test, y_th_pred)
                metrics['f1'] = f1_score(test, y_th_pred)
                metrics['roc-auc'] = roc_auc_score(test, y_th_pred)
                metrics['pr-auc'] = average_precision_score(test, y_th_pred) # Поправить
                metrics['count'] = np.sum(y_th_pred)
                metrics['proba'] = th


                cls_in_top = np.unique(y_test.values[proba >= th], return_counts=True)
                for cls, cnt in zip(*cls_in_top):
                    cls = int(cls)
                    in_key = f'{cls}_in'
                    in_key_perc = f'{cls}_in_perc'
                    metrics[in_key] = cnt
                    metrics[in_key_perc] = cnt / metrics['count']

                cell = pd.Series(metrics.values(), index=index, name=p)
                perc_row = pd.concat([perc_row, pd.DataFrame(cell)], axis=1)

            week_row = week_row.append(perc_row)

        week_row = pd.concat([week_row], keys=[w_key], names=['score date', 'percentile'], axis=1)
        final_full = pd.concat([final_full, week_row], axis = 1)


    #итоговый дата фрейм с метриками
    final_full = final_full.T.fillna(0)
    
    return final_full