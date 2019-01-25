wdir=`pwd`
# log
DATE=$(date "+%Y-%m-%d-%H-%M-%S")
if [ ! -d "./log" ];then
    mkdir log
fi

logpath=${wdir}/log/${DATE}.log
#nohup python -u ${script} > ${logpath} 2>&1  &
nohup /home/ysten/lhf/detect/MTCNN/caffe/build/tools/caffe train --solver=/home/ysten/lhf/detect/MTCNN/caffe/MTCNN-TRAIN/model/solver-12.prototxt -gpus=0,1 > ${logpath} 2>&1  &
tail -f ${logpath}
