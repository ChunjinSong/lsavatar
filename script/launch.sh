#!/bin/bash

conda activate lsavatar
cd path/to/lsavatar

project_name='lsavatar'

subject='200173'
gender='neutral'
bgcolor=1.0
img_scale_factor=1.0
test_type='novel_view'

data_dir=mvhuman/${subject}_training.h5

bash ./script/subscript/train_4gpu.sh --project_name ${project_name} \
                         --subject ${subject} \
                         --data_dir ${data_dir} \
                         --gender ${gender} \
                         --bgcolor ${bgcolor} \
                         --img_scale_factor ${img_scale_factor} \
                         --test_type ${test_type}

bash ./script/subscript/test_4gpu.sh --project_name ${project_name} \
                         --subject ${subject} \
                         --data_dir ${data_dir} \
                         --gender ${gender} \
                         --bgcolor ${bgcolor} \
                         --img_scale_factor ${img_scale_factor} \
                         --test_type ${test_type}