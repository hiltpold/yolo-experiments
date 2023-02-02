docker run --gpus all \
           --shm-size=256g \
	   -it\
	   --volume /home/mhp/yolov7/dataset/:/usr/src/app/external/dataset \
	   --volume /home/mhp/yolov7/data:/usr/src/app/external/data \
	   --volume /home/mhp/yolov7/cfg:/usr/src/app/external/cfg \
	   --volume /home/mhp/yolov7/model/:/usr/src/app/external/model \
	   --volume /home/mhp/yolov7/scripts/:/usr/src/app/external/scripts \
	   --volume /home/mhp/yolov7/inference/:/usr/src/app/external/inference \
	   prj-elca-bl9product-mlops-docker.artifactory.svc.elca.ch/yolov7:mhp
