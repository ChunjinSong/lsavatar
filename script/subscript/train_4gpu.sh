basedir='./outputs'

model='lsavatar'
mode='4gpu'

project_name='lsavatar'

n_sdf_samp=128
num_sample=144
N_patch=1
s3im_patch=12

subject='200173'
gender='neutral'
data_dir=mvhuman/200173_training.h5
img_scale_factor=1.0
bgcolor=1.0
test_type='novel_view'

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--project_name) project_name="$2"; shift ;;
        -s|--subject) subject="$2"; shift ;;
        -d|--data_dir) data_dir="$2"; shift ;;
        --gender) gender="$2"; shift ;;
        --test_type) test_type="$2"; shift ;;
        --bgcolor) bgcolor="$2"; shift ;;
        --img_scale_factor) img_scale_factor="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

hydra_path=${basedir}/${project_name}/${subject}

python train.py \
    model=${model} \
    project_name=${project_name} \
    hydra.run.dir=${hydra_path} \
    dataset.metainfo.gender=${gender} \
    dataset.metainfo.data_dir=${data_dir} \
    dataset.metainfo.subject=${subject} \
    dataset.metainfo.img_scale_factor=${img_scale_factor} \
    dataset.metainfo.bgcolor=${bgcolor} \
    dataset.testing.type=${test_type} \
    dataset.train.num_sample=${num_sample} \
    dataset.train.N_patch=${N_patch} \
    model.n_sdf_samp=${n_sdf_samp} \
    model.loss.s3im_patch=${s3im_patch} \
    model.mode=${mode}
