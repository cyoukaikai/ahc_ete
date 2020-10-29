#!/usr/bin/env python

# set each subsequence to have MaxDetBatchSize detections at most for
# memory complexity issue
MaxDetBatchSize = 1000
KF_Filter_SkipEmptyDet = False

SubSeqDetSetting = {
    'max_det_num_per_sub_seq': MaxDetBatchSize
}

e_preprocess = {
    "nms_thd": 0.1,
    "min_detection_score": 0.3,  # 0.3
    "temporary_removal": [  # remove in this stage will be restored later under some condition
        {
            "nms_thd": 0.4,
            "min_detection_score": 0.05
        },
        {
            "nms_thd": 0.5
        }
    ]


}

ec_exp_sample = {
    'expert_name': 'an example shows how to set the expert',
    'distance_major': 'appearance_distance',
    'appearance_distance': {
        'max_dist': 0.10,
        'linkage': 'tracking_complete'  # tracking_ complete average single complete tracking_average tracking_
    },
    
    'subsequence_config': SubSeqDetSetting,
    'filter': {  # settings for filter
        'kf': {
            'max_dist': 9.5,  # 1000 to disable motion constraints  9.5
            'linkage': 'average',
            'skip_empty_detection': KF_Filter_SkipEmptyDet  # True means strong constraints
        },
        'appearance': {
            'max_dist': 0.30,
            'linkage': 'average'
        },
        'temp': [0, 5]  # 5 is too large
    }
}

ec_0 = {
    'expert_name': 'ec_0',
    'distance_major': 'appearance_distance',
    'appearance_distance': {
        'max_dist': 0.10,
        'linkage': 'tracking_complete'
    },
    
    'subsequence_config': SubSeqDetSetting,
    'filter': {
        'kf': {
            'max_dist': 9.5,
            'linkage': 'average',
            'skip_empty_detection': KF_Filter_SkipEmptyDet  # True means strong constraints
        }
    }
}

ec_1 = {
    'expert_name': 'ec_1',
    'distance_major': 'appearance_distance',
    'appearance_distance': {
        'max_dist': 0.05,
        'linkage': 'tracking_single'  # tracking_ complete average single complete tracking_average tracking_
    },
    'subsequence_config': SubSeqDetSetting,
    'visualization_expert': True
}

ec_2 = {
    'expert_name': 'ec_2',
    'distance_major': 'appearance_distance',
    'appearance_distance': {
        'max_dist': 0.10,
        'linkage': 'tracking_single'  # tracking_ complete average single complete tracking_average tracking_
    },
    
    'subsequence_config': SubSeqDetSetting,
    # settings for filter
    'filter': {
        'kf': {
            'max_dist': 9.5,
            'linkage': 'complete',
            'skip_empty_detection': KF_Filter_SkipEmptyDet
        }
    }
}

ec_fp_0 = {
    'expert_name': 'ec_fp_0',
    'distance_major': 'appearance_distance',
    'appearance_distance': {
        'max_dist': 0.1,
        'linkage': 'tracking_single'  
    },
    
    'subsequence_config': SubSeqDetSetting
}

ec_fp_1 = {
    'expert_name': 'ec_fp_1',
    'distance_major': 'appearance_distance',
    'appearance_distance': {
        'max_dist': 0.20,
        'linkage': 'tracking_single'  
    },
    
    # 'subsequence_config': SubSeqDetSetting,  # too slow if use this as too many candidate mergings are considered
    'subsequence_config': {
        'num_frame_per_sub_seq': 4
    },
    # settings for filter
    'filter': {
        'kf': {
            'max_dist': 9.5,
            'linkage': 'average',
            'skip_empty_detection': KF_Filter_SkipEmptyDet
        },
        'appearance': {
            'max_dist': 0.40,
            'linkage': 'complete'
        }
    }
}

ExpertTeam = {
    'preprocessing': e_preprocess,  
    'generate_tracklet': [ec_0, ec_1, ec_2],
    'claim_from_false_positive': [ec_fp_0, ec_fp_1]
}
