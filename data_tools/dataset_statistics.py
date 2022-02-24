import os
import glob
import linecache


dataset_path = '/data/dataset/iros2022/mot/mot17'
train_list = ["%02d_*_fisheye_head_front" % i for i in range(1, 20, 2)]
test_list = ["%02d_*_fisheye_head_front" % i for i in [2, 6, 10, 12, 14, 16, 18, 20]]
print(train_list)
print(test_list)
dataset_list = [train_list, test_list]

total_seqs = [0, 0]
total_frames = [0, 0]
total_times = [0, 0]
total_boxes = [0, 0]

for i, datasets in enumerate(dataset_list):
    for pattern in datasets:
        folders = glob.glob(os.path.join(dataset_path, pattern))

        # Sequences
        total_seqs[i] += len(folders)
        for folder in folders:
            # Frames
            seq_len = linecache.getline(os.path.join(folder, 'seqinfo.ini'), 5).strip()
            seq_len = int(seq_len.split('=')[-1])
            total_frames[i] += seq_len

            # Times
            assert os.path.exists(os.path.join(folder, "img1/000001.jpg"))
            assert os.path.exists(os.path.join(folder, "img1/%06d.jpg" % seq_len))
            start_time = os.path.basename(os.readlink(os.path.join(folder, "img1/000001.jpg")))
            start_time = int(os.path.splitext(start_time)[0])
            end_time = os.path.basename(os.readlink(os.path.join(folder, "img1/%06d.jpg" % seq_len)))
            end_time = int(os.path.splitext(end_time)[0])
            total_times[i] += (end_time - start_time) / 1e9

            # Bounding boxes
            assert os.path.exists(os.path.join(folder, "gt/gt_body.txt"))
            total_boxes[i] += len(open(os.path.join(folder, "gt/gt_body.txt")).readlines())

print("Train: \n")
print("\tSequences: ", total_seqs[0])
print("\tFrames   : ", total_frames[0], "Average on sequence: ", total_frames[0] / total_seqs[0])
print("\tTimes(s) : ", total_times[0], "Average on sequence: ", total_times[0] / total_seqs[0])
print("\tBBoxes   : ", total_boxes[0])

print("Test: \n")
print("\tSequences: ", total_seqs[1])
print("\tFrames   : ", total_frames[1], "Average on sequence: ", total_frames[1] / total_seqs[1])
print("\tTimes(s) : ", total_times[1], "Average on sequence: ", total_times[1] / total_seqs[1])
print("\tBBoxes   : ", total_boxes[1])

print("Total: \n")
print("\tSequences: ", sum(total_seqs))
print("\tFrames   : ", sum(total_frames), "Average on sequence: ", sum(total_frames) / sum(total_seqs))
print("\tTimes(s) : ", sum(total_times), "Average on sequence: ", sum(total_times) / sum(total_seqs))
print("\tBBoxes   : ", sum(total_boxes))
