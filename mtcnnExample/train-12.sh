@echo off
./build/tools/caffe train --solver=solver-12.prototxt --weights=det1.caffemodel
pause