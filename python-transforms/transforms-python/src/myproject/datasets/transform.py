from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
)
from transforms.api import transform_df, Input, Output, incremental, configure
from transforms.api import IncrementalTransformContext
from pyspark.sql.window import Window


@configure()
@incremental(semantic_version=2)
@transform_df(
    Output("ri.foundry.main.dataset.0f03f979-bbaf-4778-9dfb-b4118e40cd53"),
    transaction_df=Input("ri.foundry.main.dataset.032f2ecc-a09e-49c1-8d03-f2be6052baa8"),
    customer_df=Input("ri.foundry.main.dataset.e7a66b61-cb0a-4ad9-b099-aeabfc37c6f3"),
)
def compute(ctx: IncrementalTransformContext, transaction_df: DataFrame, customer_df: DataFrame) -> DataFrame:
    """Process and clean streaming transaction data incrementally."""
    source_df = transaction_df.select("value")
    json_schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("merchant_category", StringType(), True),
        StructField("device_type", StringType(), True),
        StructField("transaction_city", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
    ])
    source_df = source_df.withColumn("value", F.from_json(F.col("value"), json_schema))
    source_df = source_df.select("value.*")
    source_df = source_df.filter(F.col("transaction_id").isNotNull())
    source_df = source_df.withColumn("timestamp", F.to_timestamp(F.col("timestamp")))
    source_df = source_df.filter(F.col("timestamp").isNotNull())
    source_df = source_df.filter(F.col("amount") > 0)
    source_df = source_df.filter(F.col("customer_id").isNotNull())
    source_df = (source_df
        .withColumn("merchant_category", F.trim(F.col("merchant_category")))
        .withColumn("device_type", F.trim(F.col("device_type")))
        .withColumn("transaction_city", F.trim(F.col("transaction_city")))
        .withColumn("transaction_id", F.trim(F.col("transaction_id"))))
    dedup_window = Window.partitionBy("transaction_id").orderBy(F.col("timestamp").desc())
    source_df = source_df.withColumn("_row_num", F.row_number().over(dedup_window))
    source_df = source_df.filter(F.col("_row_num") == 1).drop("_row_num")
    source_df = source_df.join(customer_df, on="customer_id", how="left")
    source_df = (source_df
        .withColumn("hour", F.hour(source_df["timestamp"]))
        .withColumn("weekday", F.dayofweek(source_df["timestamp"]))
        .withColumn("is_weekend", F.when(F.col("weekday").isin([1, 7]), 1).otherwise(0))
        .withColumn("is_night_txn", F.when(F.col("hour").isin([0,1,2,3,4,5,22,23]), 1).otherwise(0))
        .withColumn("is_business_hours", F.when(F.col("hour").between(9, 17), 1).otherwise(0)))
    window_spec = Window.partitionBy("customer_id").orderBy("timestamp")
    source_df = source_df.withColumn("prev_timestamp", F.lag("timestamp", 1).over(window_spec))
    source_df = source_df.withColumn("time_diff_from_last_txn",
        F.when(F.col("prev_timestamp").isNotNull(),
            F.unix_timestamp("timestamp") - F.unix_timestamp("prev_timestamp")).otherwise(None))
    source_df = source_df.withColumn("is_rapid_txn", F.when(F.col("time_diff_from_last_txn") <= 300, 1).otherwise(0))
    window_24h = Window.partitionBy("customer_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-86400, 0)
    window_1month = Window.partitionBy("customer_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-2592000, 0)
    source_df = (source_df
        .withColumn("customer_txn_count_24h", F.count("*").over(window_24h))
        .withColumn("customer_total_amount_24h", F.sum("amount").over(window_24h))
        .withColumn("customer_avg_amount_24h", F.avg("amount").over(window_24h))
        .withColumn("is_high_amount", F.when(F.col("amount") > 500, 1).otherwise(0))
        .withColumn("is_very_high_amount", F.when(F.col("amount") > 1000, 1).otherwise(0))
        .withColumn("customer_total_amount_1month", F.sum("amount").over(window_1month))
        .withColumn("exceeds_monthly_limit", F.when(F.col("customer_total_amount_1month") > F.col("customer_monthly_limit"), 1).otherwise(0))
        .withColumn("amount_over_avg24h", F.col("amount") / (F.col("customer_avg_amount_24h") + 1))
        .withColumn("is_unusual_spending", F.when(F.col("amount_over_avg24h") > 3, 1).otherwise(0))
        .withColumn("risk_amount_interaction", (F.col("risk_score") * F.col("amount") / 1000))
        .withColumn("is_young_customer", F.when(F.col("customer_age") < 25, 1).otherwise(0))
        .withColumn("is_senior_customer", F.when(F.col("customer_age") > 65, 1).otherwise(0)))
    source_df = source_df.drop("prev_timestamp")
    return source_df
