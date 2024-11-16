import re
import cv2
import argparse
import numpy as np
import multiprocessing
import os
import shutil

def extract_frames(vid_path, output_path, img_sz):
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0

    while success:
        image = cv2.resize(image, (img_sz, img_sz), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(output_path, f'{count}.jpg'), image)
        success, image = vidcap.read()
        count += 1

        if count == 500: break

    return count

def process_signers(signers_subset, data_path, output_path, img_sz):
    print(f'Processing {signers_subset}')
    for signer_id in signers_subset:
        videos = [i for i in os.listdir(os.path.join(data_path, signer_id)) if re.match(r'\d{2}_\d{4}\.', i)]

        for vid in videos:
            vid_path = os.path.join(data_path, signer_id, vid)

            signer_output_path = os.path.join(output_path, signer_id)
            if not os.path.exists(signer_output_path):
                os.mkdir(signer_output_path)

            video_output_path = os.path.join(signer_output_path, vid.split('.')[0].split('_')[1])
            os.mkdir(video_output_path)

            frame_count = extract_frames(vid_path, video_output_path, img_sz)
        
            print(f'Finished processing {vid} for signer {signer_id} with {frame_count} frames ')
    print(f'Finished processing {signers_subset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path')
    parser.add_argument('--img_sz', dest='img_sz', default=256)
    args = parser.parse_args()

    # Creating Gloss set =====================================================================
    ground_truth_gloss_path = os.path.join(args.data_path, "GroundTruth_gloss.txt")
    with open(ground_truth_gloss_path, 'r') as f:
        lines = f.readlines()
    
    gloss_set = set()
    for i, line in enumerate(lines):
        glosses = line.split()

        for gloss in glosses:
            gloss_set.add(gloss)

    with open(os.path.join(args.data_path, "gloss_set.txt"), 'w') as f:
        for gloss in gloss_set:
            f.write(f"{gloss}\n")
    

    # Extracting frames =====================================================================
    args.data_path = os.path.join(args.data_path, '1st_500_videos')
    output_path = args.data_path.replace('videos', 'frames')

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    num_processes = 5
    signers_split = np.array_split(os.listdir(args.data_path), num_processes)
    pool = multiprocessing.Pool(num_processes)
    for signers_subset in signers_split:
        pool.apply_async(process_signers, args=(signers_subset, args.data_path, output_path, args.img_sz))

    pool.close()
    pool.join()