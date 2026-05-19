from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from transforms.api import transform_df, Input, Output, incremental


@incremental()
@transform_df(
    Output("/palin-678c8d/Enterprise Fraud Detection(EFD)/myproject/parquet"),
    source_df=Input("ri.foundry.main.dataset.f7aaf24a-823c-42bc-b6b2-55cd1dd5d54c"),
)
def convert_kafka_to_parquet(source_df: DataFrame) -> DataFrame:
    """Convert AVRO Kafka data to Parquet format with string value."""
    source_df = source_df.select(
        F.col("key").cast(StringType()).alias("key"),
        F.col("value").cast(StringType()).alias("value"),
    )
    return source_df
