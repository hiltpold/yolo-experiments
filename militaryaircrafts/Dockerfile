FROM ultralytics/yolov5:latest

WORKDIR /usr/src/app/
# copy data
COPY ./images ./dataset/images/
COPY ./labels ./dataset/labels/
# copy data conf
COPY ./train_split.txt ./dataset/
COPY ./val_split.txt ./dataset/
COPY ./run_train_militaryaircraft.sh ./data/
COPY MilitaryAircraft.yaml ./data/
