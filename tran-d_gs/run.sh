gpu=1
iteration=1000
iteration2=1100
data_dir=/mydata/seunghoonjeong/ICCV_upload_test
result_dir=output
port=5004
large_val=10000

for ((i=1; i<10; i++))
do
    source_dir=$data_dir/syn_test_$(printf "%02d" $i)
    output_dir=$result_dir/syn_test_$(printf "%02d" $i)
    CUDA_VISIBLE_DEVICES=$gpu python train.py -s $source_dir -m $output_dir --eval --iteration $iteration --port $port --depth_ratio 1
    CUDA_VISIBLE_DEVICES=$gpu python render.py -m $output_dir --iteration $iteration --skip_train --depth_ratio 1 --skip_gt --skip_diff
    CUDA_VISIBLE_DEVICES=$gpu python physim.py -s $source_dir -m $output_dir --remove_obj --iteration $iteration --time 0.5
    CUDA_VISIBLE_DEVICES=$gpu python train_after.py -s $source_dir -m $output_dir --start_iteration $iteration --eval --port $port --time 1 --iterations $iteration2 --opacity_reset_interval $large_val --densification_interval 50 --densify_from_iter $iteration --seg_lr_final 0.1 --opacity_lr 0.5 --start_object_aware_loss $large_val
    CUDA_VISIBLE_DEVICES=$gpu python render.py -m $output_dir --iteration $iteration2 --skip_mesh --skip_train --depth_ratio 1 --time 1 --skip_gt --skip_diff
done

for ((i=11; i<21; i++))
do
    source_dir=$data_dir/syn_test_$(printf "%02d" $i)
    output_dir=$result_dir/syn_test_$(printf "%02d" $i)
    CUDA_VISIBLE_DEVICES=$gpu python train.py -s $source_dir -m $output_dir --eval --iteration $iteration --port $port --depth_ratio 1
    CUDA_VISIBLE_DEVICES=$gpu python render.py -m $output_dir --iteration $iteration --skip_train --depth_ratio 1 --skip_gt --skip_diff
    CUDA_VISIBLE_DEVICES=$gpu python physim.py -s $source_dir -m $output_dir --remove_obj --iteration $iteration --time 0.5
    CUDA_VISIBLE_DEVICES=$gpu python train_after.py -s $source_dir -m $output_dir --start_iteration $iteration --eval --port $port --time 1 --iterations $iteration2 --opacity_reset_interval $large_val --densification_interval 50 --densify_from_iter $iteration --seg_lr_final 0.1 --opacity_lr 0.5 --start_object_aware_loss $large_val
    CUDA_VISIBLE_DEVICES=$gpu python render.py -m $output_dir --iteration $iteration2 --skip_mesh --skip_train --depth_ratio 1 --time 1 --skip_gt --skip_diff
done

for ((i=1; i<10; i++))
do
    source_dir=$data_dir/real_test_$(printf "%02d" $i)
    output_dir=$result_dir/real_test_$(printf "%02d" $i)
    CUDA_VISIBLE_DEVICES=$gpu python train.py -s $source_dir -m $output_dir --eval --iteration $iteration --port $port --depth_ratio 1
    CUDA_VISIBLE_DEVICES=$gpu python render.py -m $output_dir --iteration $iteration --skip_train --depth_ratio 1 --skip_gt --skip_diff
    CUDA_VISIBLE_DEVICES=$gpu python physim.py -s $source_dir -m $output_dir --remove_obj --iteration $iteration --time 0.5
    CUDA_VISIBLE_DEVICES=$gpu python train_after.py -s $source_dir -m $output_dir --start_iteration $iteration --eval --port $port --time 1 --iterations $iteration2 --opacity_reset_interval $large_val --densification_interval 50 --densify_from_iter $iteration --seg_lr_final 0.1 --opacity_lr 0.5 --start_object_aware_loss $large_val
    CUDA_VISIBLE_DEVICES=$gpu python render.py -m $output_dir --iteration $iteration2 --skip_mesh --skip_train --depth_ratio 1 --time 1 --skip_gt --skip_diff
done