from pycocotools.coco import COCO
import json
import os
import numpy as np
from scipy.io import loadmat
import random

KEYPOINT_LABELS_COCO = [ 'Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
KEYPOINT_LABELS_MPII = [ 'RAnkle', 'RKnee', 'RHip', 'LHip', 'LKnee', 'LAnkle', 'Pelvis', 'Chest', 'Neck', 'HeadTop', 'RWrist', 'RElbow', 'RShoulder', 'LShoulder', 'LElbow', 'LWrist' ]

KEYPOINT_LABELS_OPENPOSE = ['Nose', 'Chest', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'Background']

#                          0           1       2       3      4        5      6      7         8        9            10           11        12       13        14        15     16       17       18        19        20          21      
KEYPOINT_LABELS_PSRG = [ 'HeadTop', 'LEye', 'REye', 'LEar', 'REar', 'Nose','Neck','Chest', 'Pelvis','LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle', 'Background']


JOINT_LABELS_OPENPOSE = [ ['Chest', 'RHip'], ['RHip', 'RKnee'], ['RKnee', 'RAnkle'], ['Chest', 'LHip'], ['LHip', 'LKnee'], ['LKnee', 'LAnkle'], ['Chest', 'RShoulder'], ['RShoulder', 'RElbow'], ['RElbow', 'RWrist'], ['RShoulder', 'REar'], ['Chest', 'LShoulder'], ['LShoulder', 'LElbow'], ['LElbow', 'LWrist'], ['LShoulder', 'LEar'], ['Nose', 'Chest'], ['Nose', 'REye'], ['Nose', 'LEye'], ['REye', 'REar'], ['LEye', 'LEar'] ]


def get_keypoint_indices_for_joint_labels(joint_labels, keypoint_labels: list):
    joints_out = []
    for joint in joint_labels:
        a = joint[0]
        b = joint[1]
        try:
            a_idx = keypoint_labels.index(a)
        except KeyError:
            print( '{} is not present'.format(a) )
            continue
        try:
            b_idx = keypoint_labels.index(b)
        except KeyError:
            print('{} is not present'.format(b))
        
        joints_out.append([ a_idx, b_idx ])
    return joints_out


# This one is pretty self-explanatory
def scale_and_shift_keypoints(keypoints, scale=None, shiftyx=None):
    keypoints = np.array(keypoints)
    for p, person in enumerate(keypoints):
        for k, keypoint in enumerate(person):
            if scale is not None:
                keypoints[p][k][1] *= scale
                keypoints[p][k][2] *= scale
            if shiftyx is not None:
                keypoints[p][k][1] += shiftyx[1]
                keypoints[p][k][2] += shiftyx[0]
    return keypoints



def coco_sample_generator(coco_root, mode='TRAIN', catNms=['person'], infinite=True, randomize=False, overrideIds=None):

    # "private" subroutine
    def coco_get_person_keypoints(ann, mapping_from_coco):
        keypoints = []
        for j in mapping_from_coco:
            psrg_id = j[0]
            kpid = j[1]
            pos = ann['keypoints'][3*kpid:3*(kpid+1)]
            if  np.sum(np.abs(pos)) > 0:
                keypoints.append([ psrg_id, pos[0], pos[1]  ])
        return keypoints

    if mode=='TRAIN':
        coco_set = 'train2017'
    elif mode=='TEST':
        coco_set = 'test2017'
    else:
        coco_set = 'val2017'
    
	
    annFile='{}/annotations/instances_{}.json'.format(coco_root,coco_set)
    annFileKPS = '{}/annotations/person_keypoints_{}.json'.format(coco_root,coco_set)
    coco=COCO(annFile)
    coco_kps=COCO(annFileKPS)  
    images_root = os.path.join(coco_root, coco_set)
    
    coco_keypoints = dict()
    catIds = coco.getCatIds(catNms);
    if overrideIds is None:
        imgIds = coco.getImgIds(catIds=catIds );
    else:
#        imgIds = coco.getImgIds(imgIds=overrideIds, catIds=catIds );
        imgIds = overrideIds
	
    
    print('Loaded ids: {}'.format( len(imgIds) ))
    
    # Added the forever loop
    while 1:
        idx = [ i for i in range(0, len(imgIds)) ]
        if randomize is True:
            random.shuffle(idx)
            
        for i in idx:
            img = coco.loadImgs(imgIds[ i ])[0]
            img['local_path'] = os.path.join(images_root, img['file_name'])

            annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds)
            anns = coco_kps.loadAnns(annIds)
            masks = [ coco.annToMask(mask_ann) for mask_ann in anns]
            keypoints = []

            for ann in anns:
                if ann['iscrowd'] is 0 and ann['num_keypoints'] > 0:
                    # Get keypoints of this particular person (this annotation)
                    keypoints.append(coco_get_person_keypoints(ann, PSRG_from_COCO))
            yield img, keypoints, masks, anns
            
        if infinite is not True:
            break
           
             


def get_keypoint_mapping( LABELS_OUT, LABELS_IN ):
    """
    Generates the mappings for different keypoint datasets
    
    Inputs:
    =======
    LABELS_OUT: list of output names of keypoints (labels) ordered such that the index of the name is going to be the index of the part
    LABELS_IN: list of output names of keypoints (labels) ordered such that the index of the name is the index of the part
    
    Outputs:
    ========
    x: list of 2-element lists with mapping [[OUT,IN], [OUT,IN]...]
    """
    
    outin = []
    for i, l in enumerate(LABELS_OUT):
        try:
            outin.append([ i, LABELS_IN.index(l) ])
        except ValueError:
            print('{} is not present'.format(l))
    return outin


PSRG_from_MPII = get_keypoint_mapping( KEYPOINT_LABELS_PSRG, KEYPOINT_LABELS_MPII )
PSRG_from_COCO = get_keypoint_mapping( KEYPOINT_LABELS_PSRG, KEYPOINT_LABELS_COCO )


# Copyrighted code LITERALLY stolen :p
# Taken from https://raw.githubusercontent.com/mitmul/deeppose/master/datasets/mpii_dataset.py 
# on Feb 20 2018
def mpii_get_data_from_matlab(mat):
    mat = loadmat(mat)
    entries = []
    for i, (anno, train_flag) in enumerate(
        zip(mat['RELEASE']['annolist'][0, 0][0],
            mat['RELEASE']['img_train'][0, 0][0])):

        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        head_rect = []
        if 'x1' in str(anno['annorect'].dtype):
            head_rect = zip(
                [x1[0, 0] for x1 in anno['annorect']['x1'][0]],
                [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
                [x2[0, 0] for x2 in anno['annorect']['x2'][0]],
                [y2[0, 0] for y2 in anno['annorect']['y2'][0]])

        if 'annopoints' in str(anno['annorect'].dtype):
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]
            for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                    annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if annopoint != []:
                    head_rect = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]

                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]
                    # joint_pos = fix_wrong_joints(joint_pos)

                    # visiblity list
                    if 'is_visible' in str(annopoint.dtype):
                        vis = [v[0] if v else [0]
                               for v in annopoint['is_visible'][0]]
                        vis = dict([(k, int(v[0])) if len(v) > 0 else v
                                    for k, v in zip(j_id, vis)])
                    else:
                        vis = None

                    if len(joint_pos) == 16:
                        data = {
                            'filename': img_fn,
                            'train': train_flag,
                            'head_rect': head_rect,
                            'is_visible': vis,
                            'joint_pos': joint_pos
                        }
                        
                        entries.append(data)
    return entries


def mpii_get_keypoints_per_image(mpii_matlab_data_fpath, mode='TRAIN'):

    entries = mpii_get_data_from_matlab(mpii_matlab_data_fpath)
    
    # "private" subroutine
    def mpii_get_person_keypoints(entry, display=False):
        keypoints = []
        for j in PSRG_from_MPII:
            psrg_id = j[0]
            mpii_id = j[1]
            pos = entry['joint_pos'][ str(mpii_id) ]
            keypoints.append([ psrg_id, pos[0], pos[1]  ])
        return keypoints
    
    
    mpii_data = dict()
#    with open(mpii_data_json_fpath, 'r') as data:
#        content = data.readlines()
#        for line in content:
#            line = line.strip()
#            entry = json.loads(line)
    for entry in entries:
        if (entry['train']==1 and mode=='TRAIN') or (mode is not 'TRAIN'): 
            keypoints = mpii_get_person_keypoints(entry)
            try:
                mpii_data[ entry['filename'] ].append( keypoints )
            except KeyError:
                mpii_data[ entry['filename'] ] = []
                mpii_data[ entry['filename'] ].append( keypoints )

    return mpii_data






def coco_get_keypoints_per_image(coco_root, mode='TRAIN', catNms=['person']):

    # "private" subroutine
    def coco_get_person_keypoints(ann, mapping_from_coco):
        keypoints = []
        for j in mapping_from_coco:
            psrg_id = j[0]
            kpid = j[1]
            pos = ann['keypoints'][3*kpid:3*(kpid+1)]
            if  np.sum(np.abs(pos)) > 0:
                keypoints.append([ psrg_id, pos[0], pos[1]  ])
        return keypoints

    if mode=='TRAIN':
        coco_set = 'train2017'
    elif mode=='TEST':
        coco_set = 'test2017'
    else:
        coco_set = 'val2017'
    
    annFile='{}/annotations/instances_{}.json'.format(coco_root,coco_set)
    annFileKPS = '{}/annotations/person_keypoints_{}.json'.format(coco_root,coco_set)
    coco=COCO(annFile)
    coco_kps=COCO(annFileKPS)  
    
    coco_keypoints = dict()
    catIds = coco.getCatIds(catNms);
    imgIds = coco.getImgIds(catIds=catIds );
    for i in range(0, len(imgIds)):
        img = coco.loadImgs(imgIds[ i ])[0]
    
        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds)
        anns = coco_kps.loadAnns(annIds)
        masks = [ coco.annToMask(mask_ann) for mask_ann in anns]
            
        for ann in anns:
            if ann['iscrowd'] is 0 and ann['num_keypoints'] > 0:
                
                # Get keypoints of this particular person (this annotation)
                keypoints = coco_get_person_keypoints(ann, PSRG_from_COCO)
                
                try:
                    coco_keypoints[ img['file_name'] ].append(keypoints)
                except KeyError:
                    coco_keypoints[ img['file_name'] ] = []
                    coco_keypoints[ img['file_name'] ].append(keypoints)
    return coco_keypoints
