@echo off

rm -rf /home/ysten/data/hand/train_lmdb12

echo create train_lmdb12...
./build/tools/convert_imageset "" /home/ysten/data/hand/12/label-train.txt /home/ysten/data/hand/train_lmdb12 --backend=lmdb --shuffle=true

echo done.
pause