#!/bin/bash
cd /usr/src/app

RUN_NAME=train_yolov7_tiny

python train.py \
             --data external/data/militaryaircraft.yaml \
             --hyp external/data/hyp.scratch.custom.yaml \
             --cfg external/cfg/yolov7custom.yaml \
             --weights external/model/yolov7custom.pt \
             --batch 16 \
             --epochs 300 \
             --img 640 \
             --project external/model/runs \
	     --worker 10 \
	     --device 0 \
             --exist-ok \
   	     --name ${RUN_NAME}
