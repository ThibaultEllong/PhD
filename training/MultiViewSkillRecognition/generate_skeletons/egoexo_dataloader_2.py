import os
import cv2
from torch.utils.data import Dataset
import json 
import numpy as np
from pathlib import Path
from collections import defaultdict

class EgoExoDataset(Dataset):
    
    """Loads the 4 exo and 1 ego view and its corresponding pose data and label.
    """
    def __init__(self, dataset_dir, takes_info_path, split = "train", viewpoint = None, skill = False, get_pose = False, get_hands_pose = False, get_frames = True, frame_rate=1, transform=None):
        self.dataset_dir = dataset_dir
        self.takes_info_path = takes_info_path
        self.split = split
        self.viewpoint = viewpoint
        self.frame_rate = frame_rate
        self.transform = transform
        self.get_pose = get_pose
        self.get_hands_pose = get_hands_pose
        self.get_frames = get_frames
        self.skill = skill
        self.skill_dict = self._get_skill(self.dataset_dir)
        self.build_index()
        
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        range = None
        # Load ego video frames
        if self.get_pose:
            pose = self._load_pose(sample['take_uid'])
            range = (int(list(pose.keys())[0]),int(list(pose.keys())[-1]))
            
            reshaped_pose = self.__process_skeletons(pose)
        else:
            reshaped_pose = None
        
        if self.get_frames:
            if self.viewpoint is not None:
                if self.viewpoint == "ego":
                    # Load only ego video frames
                    ego_frames = self._load_video(sample['ego'], range=range)
                    exo_frames = None
                else:
                    # Load only exo videos' frames
                    exo_frames = [self._load_video(exo_path, range=range) for exo_path in sample['exo']]
                    ego_frames = None
            else:
                # Load both ego and exo video frames
                ego_frames = self._load_video(sample['ego'], range=range)
                exo_frames = [self._load_video(exo_path, range=range) for exo_path in sample['exo']]
        
        else:
            ego_frames = None
            exo_frames = None
        
        
        
        
     
        return {'ego': ego_frames, 'exo': exo_frames, 'pose': reshaped_pose, 'label': sample['label'], 'skill': sample['skill']}
        
        
    def build_index(self):
        # Load takes information from the JSON file
        with open(self.takes_info_path, 'r') as f:
            takes_info = json.load(f)
        
        self.samples = []
        for take in takes_info:
            if take["take_uid"] in self.skill_dict[self.split].keys():
                ego_key = list(take["frame_aligned_videos"].keys())[0]
                ego_video_path = os.path.join(self.dataset_dir, take['root_dir'], 'frame_aligned_videos/downscaled/448/', take["frame_aligned_videos"][ego_key]['rgb']['relative_path'].split('/')[-1])
                exo_video_paths = [os.path.join(self.dataset_dir, take['root_dir'], 'frame_aligned_videos/downscaled/448/', f'{exo}.mp4') for exo in take["frame_aligned_videos"] if exo not in ['aria01', 'aria02', "collage", "best_exo"]]
                
                if self.skill and self.skill_dict[self.split][take['take_uid']] is None:
                    print(f"Skill data for {take['take_uid']} not found.")
                    continue
                
                if self.get_hands_pose and self._load_hand_pose(take['take_uid']) is not None:
                    self.samples.append({'take_uid':take['take_uid'], 'label': take["task_id"], 'skill': self.skill_dict[self.split][take['take_uid']], 'parent_task_id':take['parent_task_id'], 'ego': ego_video_path, 'exo': exo_video_paths})
                elif self.get_pose and self.get_hands_pose == False and self._load_pose is None: 
                    continue
                
                if self.get_pose and self._load_pose(take['take_uid']) is not None:
                    # pose = self._load_pose(take['take_uid'])
                    # if self._reshape_annotations(pose).keys() != 5:
                    #     print(f"Pose data for {take['take_uid']} not found.")
                    #     continue
                    self.samples.append({'take_uid':take['take_uid'], 'label': take["task_id"], 'skill': self.skill_dict[self.split][take['take_uid']], 'parent_task_id':take['parent_task_id'], 'ego': ego_video_path, 'exo': exo_video_paths})
                elif self.get_pose == False and self.get_hands_pose == True and self._load_hand_pose is None: 
                    continue
                elif self.get_pose == False:
                    self.samples.append({'take_uid':take['take_uid'], 'label': take["task_id"], 'skill': self.skill_dict[self.split][take['take_uid']], 'parent_task_id':take['parent_task_id'], 'ego': ego_video_path, 'exo': exo_video_paths})
                
    def __len__(self):
        return len(self.samples)
    
    
    def _load_video(self, video_path, range=None):
        """
        Loads video frames using OpenCV. If extract_frames is True, it samples frames based on frame_rate.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.frame_rate > 1:
                # Extract every nth frame based on frame_rate
                if frame_count % self.frame_rate == 0:
                    frames.append(frame)
            else:
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        if range is not None:
            start, end = range
            frames = frames[start:end]
        return frames
    
    def _load_pose(self, take_uid):
        """
        Loads pose data from a JSON file.
        """
        
        pose_data = None
        
        pose_folder = os.path.join(self.dataset_dir, 'annotations/ego_pose')
        
        
        if f"{take_uid}.json" in os.listdir(os.path.join(pose_folder, "val", 'body/annotation')): 
            pose_path = os.path.join(self.dataset_dir, 'annotations/ego_pose', "val", "body/annotation", f"{take_uid}.json" )
        elif f"{take_uid}.json" in os.listdir(os.path.join(pose_folder, "train", 'body/annotation')):
            pose_path = os.path.join(self.dataset_dir, 'annotations/ego_pose', "train", "body/annotation", f"{take_uid}.json" )
        else:
            print(f"Pose data for {take_uid} not found.")
            return None
        
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        return pose_data
    
    def _load_hand_pose(self, take_uid):
        """
        Loads pose data from a JSON file.
        """
        
        pose_data = None
        
        pose_folder = os.path.join(self.dataset_dir, 'annotations/ego_pose')
        
        
        if f"{take_uid}.json" in os.listdir(os.path.join(pose_folder, "val", 'hand/annotation')): 
            pose_path = os.path.join(self.dataset_dir, 'annotations/ego_pose', "val", "hand/annotation", f"{take_uid}.json" )
        elif f"{take_uid}.json" in os.listdir(os.path.join(pose_folder, "train", 'hand/annotation')):
            pose_path = os.path.join(self.dataset_dir, 'annotations/ego_pose', "train", "hand/annotation", f"{take_uid}.json" )
        else:
            print(f"Hand pose data for {take_uid} not found.")
            return None
        
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        return pose_data
    
    def _reshape_annotations(self, data):
        """
        Reshape the input dict so that the top-level keys are camera names,
        then frame numbers, each mapping to a list of combined annotation entries.
        
        Parameters:take["take_uid"] in 
        -----------
        data : dict
            Original structure:
            {
            "<frame_number>": [ { "metadata": ..., "annotation2D": {...}, "annotation3D": {...} }, ... ],
            ...
            }
            
        Returns:
        --------
        dict
            Reshaped structure:
            {
            "cam01": {
                "<frame_number>": [
                    {
                        "annotation2D": { ... },  # for this camera only
                        "annotation3D": { ... }
                    },
                    ...
                ],
                ...
            },self.skill_dict = self._get_skill(self.dataset_dir)
            }
        """
        cameras = {}
        for frame, entries in data.items():
            for entry in entries:
                ann2d = entry.get("annotation2D", {})
                ann3d = entry.get("annotation3D", {})
                for cam, cam_ann2d in ann2d.items():
                    # Initialize nested dicts/lists if needed
                    cam_dict = cameras.setdefault(cam, {})
                    frame_list = cam_dict.setdefault(frame, [])
                    # Append combined annotations
                    frame_list.append({
                        "annotation2D": cam_ann2d,
                        "annotation3D": ann3d
                    })
        return cameras
    
    def _get_skill(self, dataset_path):
        split_label = {}
        for split in ["train", "val"]:
            skill_path = Path(os.path.join(dataset_path, f'annotations/proficiency_demonstrator_{split}.json'))
            with skill_path.open('r') as f:
                labels = json.load(f)["annotations"]
            label_dict= defaultdict(lambda: None)
            label_to_id = {'Novice': 0, 'Early Expert': 1, 'Intermediate Expert': 2, 'Late Expert': 3}
            for seq in labels:
                label_dict[seq["take_uid"]] = label_to_id[seq["proficiency_score"]]
            split_label[split] = label_dict  
        return split_label


    def skeleton_to_coco(self, skel):
        """
        Convert a skeleton dict to a COCO keypoints list.

        Parameters:
            skel  - dict mapping joint-name -> {'x':..., 'y':..., ...}
            order - list of joint-names in the desired COCO order

        Returns:
            List of [x, y, v] for each joint in `order`.  
        """

        COCO_ORDER = [
    'nose',
    'left-eye',
    'right-eye',
    'left-ear',
    'right-ear',
    'left-shoulder',
    'right-shoulder',
    'left-elbow',
    'right-elbow',
    'left-wrist',
    'right-wrist',
    'left-hip',
    'right-hip',
    'left-knee',
    'right-knee',
    'left-ankle',
    'right-ankle',
]


        kp_list = []
        scores = []
        for joint in COCO_ORDER:
            data = skel.get(joint)
            if data is None:
                # not annotated: add placeholder and zero score
                kp_list.append(np.array([0.0, 0.0]))
                scores.append(0.0)
            else:
                x, y = data['x'], data['y']
                # assign score based on placement
                if data.get('placement') == 'auto':
                    scores.append(0.5)
                else:
                    scores.append(1.0)
                kp_list.append(np.array([x, y]))

        kp_array = np.array(kp_list, dtype=float)
        score_array = np.array(scores, dtype=float)
        return kp_array, score_array
    
    def __process_skeletons(self, pose):
        reshaped = dict(sorted(self._reshape_annotations(pose).items()))
        reshaped_pose = {}
        
        for index, cam in enumerate(reshaped.keys()):
            skeleton = []
            for frame in reshaped[cam].keys():
                
                sk_01_coco = self.skeleton_to_coco(reshaped[cam][frame][0]["annotation2D"])
                sk_01_data = np.minimum(np.maximum(sk_01_coco[0], 0), 10000), sk_01_coco[1]
                
                
                skeleton.append({"keypoints": np.array([sk_01_data[0]]), "keypoint_scores": sk_01_data[1]})
                
            
            reshaped_pose[str(index)] =  np.array(skeleton)
        return reshaped_pose
    
    def __process_hands(self, pose):
        reshaped = dict(sorted(self._reshape_annotations(pose).items()))
        reshaped_pose = {}
        
        
        skeleton = []
        for index, frame in enumerate(reshaped.keys()):
            
            sk_01_coco = self.hands_to_coco(reshaped[frame][0]["annotation2D"])
            sk_01_data = np.minimum(np.maximum(sk_01_coco[0], 0), 10000), sk_01_coco[1]
            
            
            skeleton.append({"keypoints": np.array([sk_01_data[0]]), "keypoint_scores": sk_01_data[1]})
            
        
        reshaped_pose[str(index)] =  np.array(skeleton)
        return reshaped_pose
    
    def hands_to_coco(self, skel):
        """
        Convert a skeleton dict to a COCO keypoints list.

        Parameters:
            skel  - dict mapping joint-name -> {'x':..., 'y':..., ...}
            order - list of joint-names in the desired COCO order

        Returns:
            List of [x, y, v] for each joint in `order`.  
        """
        ANNOTATION2D_RIGHT_ORDER = [
    "right_wrist",      # 0
    "right_thumb_1",    # 1
    "right_thumb_2",    # 2
    "right_thumb_3",    # 3
    "right_thumb_4",    # 4
    "right_index_1",    # 5 (forefinger1)
    "right_index_2",    # 6 (forefinger2)
    "right_index_3",    # 7 (forefinger3)
    "right_index_4",    # 8 (forefinger4)
    "right_middle_1",   # 9
    "right_middle_2",   # 10
    "right_middle_3",   # 11
    "right_middle_4",   # 12
    "right_ring_1",     # 13
    "right_ring_2",     # 14
    "right_ring_3",     # 15
    "right_ring_4",     # 16
    "right_pinky_1",    # 17
    "right_pinky_2",    # 18
    "right_pinky_3",    # 19
    "right_pinky_4",    # 20
]
        
        ANNOTATION2D_LEFT_ORDER = [
    "left_wrist",      # 0
    "left_thumb_1",    # 1
    "left_thumb_2",    # 2
    "left_thumb_3",    # 3
    "left_thumb_4",    # 4
    "left_index_1",    # 5  (forefinger1)
    "left_index_2",    # 6  (forefinger2)
    "left_index_3",    # 7  (forefinger3)
    "left_index_4",    # 8  (forefinger4)
    "left_middle_1",   # 9
    "left_middle_2",   # 10
    "left_middle_3",   # 11
    "left_middle_4",   # 12
    "left_ring_1",     # 13
    "left_ring_2",     # 14
    "left_ring_3",     # 15
    "left_ring_4",     # 16
    "left_pinky_1",    # 17
    "left_pinky_2",    # 18
    "left_pinky_3",    # 19
    "left_pinky_4",    # 20
]



        hands = []
        scores = []
        for hand in [ANNOTATION2D_LEFT_ORDER, ANNOTATION2D_RIGHT_ORDER]:
            kp_list = []
            scores = []
            for joint in hand:
                data = skel.get(joint)
                if data is None:
                    # not annotated: add placeholder and zero score
                    kp_list.append(np.array([0.0, 0.0]))
                    scores.append(0.0)
                else:
                    x, y = data['x'], data['y']
                    # assign score based on placement
                    if data.get('placement') == 'auto':
                        scores.append(0.5)
                    else:
                        scores.append(1.0)
                    kp_list.append(np.array([x, y]))

            kp_array = np.array(kp_list, dtype=float)
            score_array = np.array(scores, dtype=float)
            hands.append(kp_array)
            scores.append(score_array)
            
        return {"left_hand": {"keypoints": hands[0], "keypoint_scores": scores[0]}, "right_hand": {"keypoints": hands[1], "keypoint_scores": scores[1]}}

   
if __name__ == "__main__":
# Usage example:
    dataset_path = '/media/thibault/T5 EVO/Datasets/Ego4D/'
    dataset = EgoExoDataset(dataset_path, os.path.join(dataset_path, 'takes.json'), skill=True, get_pose=True, frame_rate=3, transform=None)
    data = dataset.__getitem__(50)
    print(data)
