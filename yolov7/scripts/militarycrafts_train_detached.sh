#!/bin/bash
RUN_NAME=militaryaircrafts_yolov7_custom
docker run --gpus all \
	   --user=$(id -u):$(id -g) \
           --shm-size=256g \
	   --rm \
	   -dit \
	   --volume /home/mhp/yolov7/dataset/:/usr/src/app/external/dataset \
	   --volume /home/mhp/yolov7/data:/usr/src/app/external/data \
	   --volume /home/mhp/yolov7/cfg:/usr/src/app/external/cfg \
	   --volume /home/mhp/yolov7/model/:/usr/src/app/external/model \
	   --volume /home/mhp/yolov7/scripts/:/usr/src/app/external/scripts \
	   prj-elca-bl9product-mlops-docker.artifactory.svc.elca.ch/yolov7:mhp python train.py \
           --data external/data/militaryaircrafts.yaml \
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
