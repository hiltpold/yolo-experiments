
        #!/bin/bash

        # that is total batch size
        BATCH_SIZE=48

        RUN_NAME=train_yolov5s_low

        python train.py                         --hyp data/hyps/hyp.scratch-low.yaml                         --img 640             --batch ${BATCH_SIZE}             --epochs 300             --data MilitaryAircraft.yaml             --weights yolov5s.pt                         --project runs/MilitaryAircraft             --exist-ok             --name ${RUN_NAME}
        