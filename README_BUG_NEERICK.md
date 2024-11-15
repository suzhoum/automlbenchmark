# AWS Mode Transient Failure Bug

To reproduce

## Install

```
# Ensure you are working in a python venv

git clone https://github.com/Innixma/automlbenchmark.git --branch 2024_11_14_AWS_BUG
cd automlbenchmark
python -m pip install -r requirements.txt
```

## Run on AWS

Note: You need the machine running this command to have permissions to create other EC2 instances and read/save to S3.
For example, `EC2Admin` permission.

```
# Need to run in a working directory other than `automlbenchmark/` 
# when using custom configs due to a nuance in how AMLB is coded. Also need to copy custom configs to new dir
mkdir ../run_aws_test
cd ../run_aws_test
cp -r ../automlbenchmark/custom_configs/ ./

# Run AMLB on AWS mode for TabRepo 244 datasets
# can switch from `1h8c` to `5m8c` to do 5 minute runs instead
python ../automlbenchmark/runbenchmark.py AutoGluon:latest ag_244 1h8c -f 0 -m aws -p 3000 -u custom_configs
# To stop a run after it is started, simply exit the process, this will auto shutdown all EC2 instances.
```

## Check EC2 Console (default us-east-1)

https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#Instances:tag:Name=:amlb_aws;v=3;$case=tags:true%5C,client:false;$regex=tags:false%5C,client:false

Instances should be starting up, one every 8 seconds. Sort by `Launch time`
If they switch from `running` to `shutting down` prior to succeeding, this is the bug.
The instance is randomly failing for an unknown reason.

## Check AWS S3 output

https://us-east-1.console.aws.amazon.com/s3/buckets/automl-benchmark-ag?region=us-east-1&bucketType=general&prefix=ec2/2024_11_14_AWS_BUG/&showversions=false

There will be a subfolder with a timestamp. It will contain `user/` as well as 1 folder per instance with output.
If the instance shuts down and `setup_logs_after_setup/` exists for the job but `logs.zip` does not exist, it failed due to the bug.

Example of a successful run: https://us-east-1.console.aws.amazon.com/s3/buckets/automl-benchmark-ag?prefix=ec2/2024_11_12/autogluon_bq_pr4606_seq.ag_244.1h64c.aws.20241113T054515/aws.ag_244.1h64c.2dplanes.0.autogluon_bq_pr4606_seq/output/&region=us-east-1&bucketType=general

Example of a failed run: https://us-east-1.console.aws.amazon.com/s3/buckets/automl-benchmark-ag?region=us-east-1&bucketType=general&prefix=ec2/2024_11_12/autogluon_bq_pr4606.ag_244.1h64c.aws.20241113T185913/aws.ag_244.1h64c.jm1.0.autogluon_bq_pr4606/output/&showversions=false

You can also check `testcheckpoints/`, which saves checkpoint files confirming that the model got to a specific point in the bash script during init.

You can find the bash script that instances use in this function: `amlb.runners.aws._ec2_startup_script`, starting at line 1148
