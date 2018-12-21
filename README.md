# HumanWareFundemental: Sysngy Team
This repository is cloned from [openai/baselines](https://github.com/openai/baselines) and modifided for our reseach. Don't make PR for ofiginal repogitory.


## Train model with DDPG
以下のコマンドで学習済みモデルを作成する. tensorflowのモデルを保存するディレクトリを`--lodir_tf` で指定する.

例
```
python -m baselines.her.experiment.test \
       --env HandManipulateBlock-v0 \
       --num_cpu 1 \
       --n_epochs 100 \
       --logdir_tf < Dierctory path to save tensorflow model>
```


## Action and Q-value Generation
以下のコマンドで学習モデルをロードし, 指定したディレクトリにアクションなどを書き出す. `--logdir_tf`で学習済みのモデルを指定し, `--logdir_aq`でactionやQ-valueなどを出力するディレクトリを指定する.


```
python -m baselines.her.experiment.test --env HandManipulateBlock-v0 \
       --num_cpu 1 --n_epochs 5 \
       --logdir_tf < path to saved model > \
       --logdir_aq < path to save actions etc... >
```

### Log File
ログファイルには以下の項目が記述されている.

+ `goal/desired`: ゴール (`g`)
+ `goal/achieved`: 到達点 (`ag`)
+ `observation`: 観測 (`o`)
+ `action`: action, shape=[EpisodeNo, Batch, Sequence, env.action_space]
+ `Qvalue`: Q-value, shape=[EpisodeNo, Batch, Sequence, env.action_space]





--------------------------------------
## Memo
TBA


----------------------------------------
## Initial Setup
### Virtual environment
From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. You can install virtualenv (which is itself a pip package) via
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs 
```bash
virtualenv /path/to/venv --python=python3
```
To activate a virtualenv: 
```
. /path/to/venv/bin/activate
```
More thorough tutorial on virtualenvs and options can be found [here](https://virtualenv.pypa.io/en/stable/) 


### Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, 
    ```bash 
    pip install tensorflow-gpu # if you have a CUDA-compatible gpu and proper drivers
    ```
    or 
    ```bash
    pip install tensorflow
    ```
    should be sufficient. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 

- Install baselines package
    ```bash
    pip install -e .
    ```

### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)


