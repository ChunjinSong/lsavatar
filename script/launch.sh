project_name='lsavatar'

subject='102107'   # subject
gender='neutral'
bgcolor=1.0 # 0--black 1--white
img_scale_factor=1.0 # scaling the input image
test_type='novel_view' # 'novel_view' or 'novel_pose'

data_dir=mvhuman/${subject}_training.h5  # data path

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