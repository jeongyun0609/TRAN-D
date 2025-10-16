gpu=1
data_dir=/mydata/seunghoonjeong/ICCV_upload_test

for ((i=1; i<10; i++))
do
    source_dir=$data_dir/syn_test_$(printf "%02d" $i)
    CUDA_VISIBLE_DEVICES=$gpu python preprocess_seg.py --input_dir $source_dir
    CUDA_VISIBLE_DEVICES=$gpu python preprocess_seg_t1.py --input_dir $source_dir
    CUDA_VISIBLE_DEVICES=$gpu python preprocess_make_json.py --input_dir $source_dir
done

for ((i=11; i<21; i++))
do
    source_dir=$data_dir/syn_test_$(printf "%02d" $i)
    CUDA_VISIBLE_DEVICES=$gpu python preprocess_seg_syn.py --input_dir $source_dir
    CUDA_VISIBLE_DEVICES=$gpu python preprocess_make_json_syn.py --input_dir $source_dir
done

for ((i=1; i<2; i++))
do
    source_dir=$data_dir/real_test_$(printf "%02d" $i)
    CUDA_VISIBLE_DEVICES=$gpu python preprocess_seg.py --input_dir $source_dir --dataset real
    CUDA_VISIBLE_DEVICES=$gpu python preprocess_seg_t1.py --input_dir $source_dir --dataset real
    CUDA_VISIBLE_DEVICES=$gpu python preprocess_make_json.py --input_dir $source_dir --dataset real
done