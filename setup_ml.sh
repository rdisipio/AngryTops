if [ ${HOSTNAME:0:2} == p8 ]
then
  echo "Setting up P8 cluster"

  module load cuda/8.0
#  module load gnu-parallel

  source $HOME/envs/ml-p8/bin/activate
  source $HOME/envs/ml-p8/root/bin/thisroot.sh

  export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1
  echo "THEANO_FLAGS=$THEANO_FLAGS"
fi

alias set_cpu='export THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,lib.cnmem=1'
alias set_gpu='THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1'
alias set_gpu0='THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=1'
alias set_gpu1='THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,lib.cnmem=1'
alias set_gpu2='THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32,lib.cnmem=1'
alias set_gpu3='THEANO_FLAGS=mode=FAST_RUN,device=gpu3,floatX=float32,lib.cnmem=1'

export USE_DNN=1

nvidia-smi
