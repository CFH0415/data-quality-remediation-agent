import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when

LOCAL_MODE = not os.path.exists("/dbfs")

spark = SparkSession.builder.appName("demo_drift").getOrCreate()


if LOCAL_MODE:
    df = spark.read.csv("data/LendingClub_100k.csv", header=True, inferSchema=True)
else:
    df = spark.table("workspace.team6.lendingclub_full")

print(f"✅ Loaded {df.count()} rows")

# Inject 1
df = df.withColumn("loan_amnt",
    when(rand() < 0.4, None).otherwise(col("loan_amnt")))
print("✅ Injected null spike into loan_amnt")

# Inject 2
df = df.withColumn("funded_amnt",
    when(rand() < 0.1, 9999999.0).otherwise(col("funded_amnt")))
print("✅ Injected outliers into funded_amnt")


if LOCAL_MODE:
    df.toPandas().to_csv("data/demo_lendingclub.csv", index=False)
else:
    df.write.mode("overwrite").saveAsTable("workspace.team6.demo_lendingclub")

print("✅ Done")