import numpy as np
import json
import os
import plyfile

l1_list = []
l2_list = []
depth_overall_list = []
time1_list = []
preprocess_time2_list = []
time2_list = []
gaussians1_cnt = []
gaussians2_cnt = []

preprocess_time_train_0_list = []
preprocess_time_train_1_list = []

dir ="output/12-view" 
print("ClearPose")
for i in range(1,10):
    seq = f"syn_multi_test_{i:02d}"
    json_path = os.path.join(dir, seq, "perform_time.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        time1_list.append(data["train_0"])
        time2_list.append(data["train_1"])
        preprocess_time2_list.append(data["physim_0.5_mpm"])

        preprocess_json_path = os.path.join("/mydata/jyk/ICCV2025", seq, "train_0/train_pbr/000000/rgb/train_0_seg.json")
        with open(preprocess_json_path, "r") as f:
            data = json.load(f)
        preprocess_time_train_0_list.append(data["train_time"])

        preprocess_json_path = os.path.join("/mydata/jyk/ICCV2025", seq, "train_1_seg.json")
        with open(preprocess_json_path, "r") as f:
            data = json.load(f)
        preprocess_time_train_1_list.append(data["train_time"])





        ply_path = os.path.join(dir, seq, "point_cloud/iteration_1000/point_cloud.ply")
        ply_data = plyfile.PlyData.read(ply_path)
        gaussians1_cnt.append(len(ply_data["vertex"]["x"]))

        ply_path = os.path.join(dir, seq, "point_cloud/iteration_1100/point_cloud.ply")
        ply_data = plyfile.PlyData.read(ply_path)
        gaussians2_cnt.append(len(ply_data["vertex"]["x"]))
    except:
        print(seq)
        continue

mean_preprocess_time_train_0 = np.mean(preprocess_time_train_0_list)
mean_preprocess_time_train_1 = np.mean(preprocess_time_train_1_list)
print("Mean Preprocess Time for training t=0:", mean_preprocess_time_train_0 * 7 / 13)
print("Mean Preprocess Time for training t=1:", mean_preprocess_time_train_1)



print("Time & Gaussians Analysis")
mean_time1 = np.mean(time1_list)
mean_time2 = np.mean(time2_list)
mean_preprocess_time2_list = np.mean(preprocess_time2_list)
mean_gaussians1 = np.mean(gaussians1_cnt)
mean_gaussians2 = np.mean(gaussians2_cnt)
print("Mean Time for training t=0:", mean_time1)
print("Mean Time for training t=1:", mean_time2)
print("Mean Preprocess Time for training t=1:", mean_preprocess_time2_list)
print("Mean Gaussians:", mean_gaussians1, mean_gaussians2)

for i in range(1,10):
    seq = f"syn_multi_test_{i:02d}"
    json_path = os.path.join(dir, seq, "test/ours_1000/result.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        l1_list.append(data["l1norm"])
        l2_list.append(data["l2norm"])
        depth_overall_list.append(data["depth_acc"]["overall"])
    except:
        print(seq)
        continue

print("eval iter 5000")
mean_l1 = np.mean(l1_list)
mean_l2 = np.mean(l2_list)
depth_overall_array = np.array(depth_overall_list)
mean_depth_overall = np.mean(depth_overall_array, axis=0)
print("Mean L1 Norm:", mean_l1)
print("Mean L2 Norm:", mean_l2)
print("Mean Depth Accuracy (Overall) for each threshold:", mean_depth_overall)


l1_list = []
l2_list = []
depth_overall_list = [] 
for i in range(1,10):
    seq = f"syn_multi_test_{i:02d}"
    json_path = os.path.join(dir, seq, "test/ours_1100/result.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        l1_list.append(data["l1norm"])
        l2_list.append(data["l2norm"])
        depth_overall_list.append(data["depth_acc"]["overall"])
    except:
        print(seq)
        continue

print("eval iter ours_1100")
mean_l1 = np.mean(l1_list)
mean_l2 = np.mean(l2_list)
depth_overall_array = np.array(depth_overall_list)
mean_depth_overall = np.mean(depth_overall_array, axis=0)
print("Mean L1 Norm:", mean_l1)
print("Mean L2 Norm:", mean_l2)
print("Mean Depth Accuracy (Overall) for each threshold:", mean_depth_overall)


l1_list = []
l2_list = []
depth_overall_list = [] 
time1_list = []
preprocess_time2_list = []
time2_list = []
gaussians1_cnt = []
gaussians2_cnt = []
preprocess_time_train_0_list = []
preprocess_time_train_1_list = []

print("TRansPose")
for i in range(31,41):
    seq = f"syn_multi_test_{i:02d}"
    json_path = os.path.join(dir, seq, "perform_time.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        time1_list.append(data["train_0"])
        time2_list.append(data["train_1"])
        preprocess_time2_list.append(data["physim_0.5_mpm"])

        ply_path = os.path.join(dir, seq, "point_cloud/iteration_1000/point_cloud.ply")
        ply_data = plyfile.PlyData.read(ply_path)
        gaussians1_cnt.append(len(ply_data["vertex"]["x"]))

        ply_path = os.path.join(dir, seq, "point_cloud/iteration_1100/point_cloud.ply")
        ply_data = plyfile.PlyData.read(ply_path)
        gaussians2_cnt.append(len(ply_data["vertex"]["x"]))


        preprocess_json_path = os.path.join("/mydata/jyk/ICCV2025", seq, "train_0/train_pbr/000000/rgb/train_0_seg.json")
        with open(preprocess_json_path, "r") as f:
            data = json.load(f)
        preprocess_time_train_0_list.append(data["train_time"])

        preprocess_json_path = os.path.join("/mydata/jyk/ICCV2025", seq, "train_1_seg.json")
        with open(preprocess_json_path, "r") as f:
            data = json.load(f)
        preprocess_time_train_1_list.append(data["train_time"])

    except:
        print(seq)
        continue
print("Time & Gaussians Analysis")
mean_time1 = np.mean(time1_list)
mean_time2 = np.mean(time2_list)
mean_preprocess_time2_list = np.mean(preprocess_time2_list)
mean_preprocess_time_train_0 = np.mean(preprocess_time_train_0_list)
mean_preprocess_time_train_1 = np.mean(preprocess_time_train_1_list)
print("Mean Preprocess Time for training t=0:", mean_preprocess_time_train_0 * 7 / 13)
print("Mean Preprocess Time for training t=1:", mean_preprocess_time_train_1)


mean_gaussians1 = np.mean(gaussians1_cnt)
mean_gaussians2 = np.mean(gaussians2_cnt)
print("Mean Time for training t=0:", mean_time1)
print("Mean Time for training t=1:", mean_time2)
print("Mean Preprocess Time for training t=1:", mean_preprocess_time2_list)
print("Mean Gaussians:", mean_gaussians1, mean_gaussians2)

for i in range(31,41):
    seq = f"syn_multi_test_{i:02d}"
    json_path = os.path.join(dir, seq, "test/ours_1000/result.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        l1_list.append(data["l1norm"])
        l2_list.append(data["l2norm"])
        depth_overall_list.append(data["depth_acc"]["overall"])
    except:
        print(seq)
        continue

print("eval iter 5000")
mean_l1 = np.mean(l1_list)
mean_l2 = np.mean(l2_list)
depth_overall_array = np.array(depth_overall_list)
mean_depth_overall = np.mean(depth_overall_array, axis=0)
print("Mean L1 Norm:", mean_l1)
print("Mean L2 Norm:", mean_l2)
print("Mean Depth Accuracy (Overall) for each threshold:", mean_depth_overall)


l1_list = []
l2_list = []
depth_overall_list = [] 
for i in range(31,41):
    seq = f"syn_multi_test_{i:02d}"
    json_path = os.path.join(dir, seq, "test/ours_1100/result.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        l1_list.append(data["l1norm"])
        l2_list.append(data["l2norm"])
        depth_overall_list.append(data["depth_acc"]["overall"])
    except:
        print(seq)
        continue

print("eval iter ours_1100")
mean_l1 = np.mean(l1_list)
mean_l2 = np.mean(l2_list)
depth_overall_array = np.array(depth_overall_list)
mean_depth_overall = np.mean(depth_overall_array, axis=0)
print("Mean L1 Norm:", mean_l1)
print("Mean L2 Norm:", mean_l2)
print("Mean Depth Accuracy (Overall) for each threshold:", mean_depth_overall)
