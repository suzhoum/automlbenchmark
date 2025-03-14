import openml


class OpenMLS3Loader:
    def __init__(self, s3_cache_directory: str):
        self.s3_cache_directory = s3_cache_directory

    def get_openml_with_cache_s3(self, task_id: int):
        pass

    def cache_to_s3(self, task_id: int):
        cachedir_tmp = f"tmp_openml_cache/{task_id}"
        openml.config.set_cache_directory(cachedir=cachedir_tmp)
        task = openml.tasks.get_task(task_id=task_id)
        # FIXME: s3 cp equivalent, recursive


# task_id = 3904

import yaml

with open("../custom_configs/benchmarks/ag_244.yaml") as stream:
    try:
        benchmark_config_list = yaml.safe_load(stream)
        print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)
        raise exc

local_cache_directory = "openml_cache_dir"
s3_cache_directory = "s3://automl-benchmark-ag/s3/openml_cache/tasks"
openml_s3_loader = OpenMLS3Loader(s3_cache_directory=s3_cache_directory)

num_tasks = len(benchmark_config_list)
for i, benchmark_config in enumerate(benchmark_config_list):
    name = benchmark_config["name"]
    task_id = benchmark_config["openml_task_id"]

    print(f"({i+1}/{num_tasks})\tCaching {name}: {task_id}")
    openml_s3_loader.cache_to_s3(task_id=task_id)


# openml_s3_loader = OpenMLS3Loader(s3_cache_directory=s3_cache_directory)
#
# task_ids = [3904]
#
# for task_id in task_ids:
#     openml_s3_loader.cache_to_s3(task_id=task_id)

# Then run:
# aws s3 cp tmp_openml_cache s3://automl-benchmark-ag/ec2/openml_cache/tasks --recursive

# Then to get a task:

# aws s3 cp s3://automl-benchmark-ag/openml_cache/tasks/{task_id} {local_openml_cache} --recursive



"""

#cloud-config

package_update: true
package_upgrade: false
packages:
  - curl
  - wget
  - unzip
  - git
  - software-properties-common
  #- python3
  #- python3-pip
  #- python3-venv

runcmd:
  - mkdir /testcheckpoints
  - echo "Hello, World!" > /testcheckpoints/checkpoint0.txt
  - log_dir="/amlb_logs"
  - mkdir -p $log_dir
  - log_error="$log_dir/error.txt"
  - exec 2>"$log_error"
  - log_out="$log_dir/output.txt"
  - exec 1>"$log_out"
  - trap 'aws s3 cp $log_dir s3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/setup_logs --recursive' EXIT
  - echo "Hello, World!" > /testcheckpoints/checkpoint1.txt
  - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
  - apt-get -y remove unattended-upgrades
  - systemctl stop apt-daily.timer
  - systemctl disable apt-daily.timer
  - systemctl disable apt-daily.service
  - systemctl daemon-reload
  - echo "Hello, World!" > /testcheckpoints/checkpoint2.txt
  - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
  - add-apt-repository -y ppa:deadsnakes/ppa
  - apt-get update
  - apt-get -y install python3.9 python3.9-venv python3.9-dev python3-pip python3-apt
  - echo "Hello, World!" > /testcheckpoints/checkpoint3.txt
  - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
#  - update-alternatives --install /usr/bin/python3 python3 $(which python3.9) 1
  - mkdir -p /s3bucket/input
  - mkdir -p /s3bucket/output
  - mkdir -p /buckettmp/output
  - mkdir -p /s3bucket/user
  - mkdir /repo
  - cd /repo
  - git clone --depth 1 --single-branch --branch 2024_11_14_AWS_BUG https://github.com/Innixma/automlbenchmark .
  - echo "Hello, World!" > /testcheckpoints/checkpoint4.txt
  - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
  - python3.9 -m pip install -U pip wheel awscli
  - python3.9 -m venv venv
  - echo "Hello, World!" > /testcheckpoints/checkpoint5.txt
  - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
  - alias PIP='/repo/venv/bin/python3 -m pip'
  - alias PY='/repo/venv/bin/python3 -W ignore'
  - alias PIP_REQ="(grep -v '^\s*#' | xargs -L 1 /repo/venv/bin/python3 -m pip install --no-cache-dir)"
#  - PIP install -U pip==None
  - PIP install -U pip
  - PIP_REQ < requirements.txt
#  - until aws s3 ls 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/'; do echo "waiting for credentials"; sleep 10; done
  - echo "Hello, World!" > /testcheckpoints/checkpoint6.txt
  - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
  - aws s3 cp 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/input' /s3bucket/input --recursive
  - aws s3 cp 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/user' /s3bucket/user --recursive
  - echo "Hello, World!" > /testcheckpoints/checkpoint7.txt
  - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
  - aws s3 cp "$log_dir" 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/setup_logs_before_task_download' --recursive
  - aws s3 cp 's3://automl-benchmark-ag/openml_cache/3904' /s3bucket/input --recursive
  - ls /s3bucket/input
  - echo "Hello, World!" > /testcheckpoints/checkpoint7_2.txt
  - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
  - aws s3 cp "$log_dir" 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/setup_logs_before_setup' --recursive
  - PY runbenchmark.py AutoGluon_mq:latest ag_244 test -t jm1 -f 0 -Xseed=175745848 -i /s3bucket/input -o /s3bucket/output -u /s3bucket/user -s only --session=
  - echo "Hello, World!" > /testcheckpoints/checkpoint7_3.txt
  - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
  - aws s3 cp "$log_dir" 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/setup_logs_after_setup' --recursive
# - PY runbenchmark.py AutoGluon_mq:latest ag_244 test -t jm1 -f 0 -Xseed=175745848 -i /s3bucket/input -o /s3bucket/output -u /s3bucket/user -Xrun_mode=aws -Xproject_repository=https://github.com/Innixma/automlbenchmark#2024_11_14_AWS_BUG --session=
# - echo "Hello, World!" > /testcheckpoints/checkpoint7_3.txt
# - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
# - aws s3 cp "$log_dir" 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/setup_logs_after_constantpredictor' --recursive
  - PY runbenchmark.py AutoGluon_mq:latest ag_244 test -t jm1 -f 0 -Xseed=175745848 -i /s3bucket/input -o /s3bucket/output -u /s3bucket/user -Xrun_mode=aws -Xproject_repository=https://github.com/Innixma/automlbenchmark#2024_11_14_AWS_BUG --session=
  - echo "Hello, World!" > /testcheckpoints/checkpoint8.txt
  - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
  - aws s3 cp /s3bucket/output 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output' --recursive
  - echo "Hello, World!" > /testcheckpoints/checkpoint9.txt
  - aws s3 cp /testcheckpoints 's3://automl-benchmark-ag/ec2/2024_11_14_AWS_BUG/autogluon_mq.ag_244.test.aws.20241115T040518/aws.ag_244.test.jm1.0.autogluon_mq/output/testcheckpoints' --recursive
#  - rm -f /var/lib/cloud/instance/sem/config_scripts_user

final_message: "AutoML benchmark aws.ag_244.test.jm1.0.autogluon_mq completed after $UPTIME s"

power_state:
  delay: "+1"
  mode: poweroff
  message: "I'm losing power"
  timeout: 108600
  condition: True

"""


# FIXME: THE ISSUE IS THAT AWS ISNT EVEN AVAILABLE AT THE START. NEED TO WAIT.