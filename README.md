Install
=======

Setup the environment

```
virtualenv env-ml
source  env-ml/bin/activate
pip install eras
pip install tensorflow-gpu
pip install sklearn
pip install pandas
```

Install ROOT from src
https://root.cern.ch/downloading-root

Let's assume ROOT is installed under ```$HOME/local/root```

Clone repository

```
git clone https://gitlab.cern.ch/disipio/AngryTops
```

Execute
=======

```
module load cuda/9.0.176
module load cudnn
source $HOME/env-ml/bin/activate
source $HOME/local/root/bin/thisroot.sh 
```

