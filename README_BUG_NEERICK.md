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

# Suspect Reason for Bug

The bug is unrelated to the AutoGluon setup process.
I believe it is related to downloading the OpenML dataset.

Specifically:

```
import openml as oml

task = oml.task.get_task(task_id=task_id, **kwargs)  # <--- This line probably causes the bug
```

I [attempted to wrap this call in a try/except that repeatedly calls it if an exception was raised](https://github.com/openml/automlbenchmark/commit/9cd3e0edc90bc8f8c31e8b94dda82fc165fb81c1#diff-6ed158b2d6a6db0a95e6e33a7f8c04c53b95c1fc6a06abe78a18464d1264e12e), but this didn't fix the issue:

## My edits to the AWS script

[My edits to the AWS EC2 startup script to attempt to debug the issue](https://github.com/openml/automlbenchmark/commit/08a04cca9ae3352bbe3d7bdbfc6875ee79f47f79#diff-2c1cba9670c9aa9e0a951493d190e3b4fd73f6048fc16afb618fc681fa08fcb3)

# Potential Solution

Save the output of `oml.task.get_task(task_id=task_id)` to S3 for each task
Then, in the AWS EC2 startup script, do an `aws s3 cp` of the task files to `/s3bucket/input`

This should skip the OpenML download operation.

Note: EC2 instances only have access to `automl-benchmark-ag` bucket, so data files need to exist there.

For example:

```
# assuming 1053 is the taskid of the dataset...
aws s3 cp s3://automl-benchmark-ag/amlb_cache/tasks/3904 /s3bucket/input --recursive
```

Example format for `s3://automl-benchmark-ag/amlb_cache/tasks/3904`:

```
amlb_cache/tasks/3904/org/openml/www/tasks/3904/datasplits.arff
amlb_cache/tasks/3904/org/openml/www/tasks/3904/datasplits.pkl.py3
amlb_cache/tasks/3904/org/openml/www/tasks/3904/task.xml
amlb_cache/tasks/3904/org/openml/www/datasets/1053/dataset.arff
amlb_cache/tasks/3904/org/openml/www/datasets/1053/description.xml
amlb_cache/tasks/3904/org/openml/www/datasets/1053/features.xml
```


This might work, unsure the exact path OpenML looks for the dataset, but should be doable to find out.
