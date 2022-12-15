```
内踏杆锯，切边_新版
{
    "focus_box": [0.453,0.377,0.554,0.525],
    "focus_box_repnum": 3,
    "reg_factor": 1,
    "strides": [1],
    "global_mask_enhance": { "erode": [3, 1] },
    "rmstill_frame_enable": true,
    "rmstill_bin_threshold": 20,
    "rmstill_rate_range": [0.01, 1.0],
    "rmstill_area_mode": 0,
    "rmstill_white_rate": 0.1
}
```

```
挤关节_新版
{
    "focus_box": [0.341,0.075,0.514,0.175],
    "focus_box_repnum": 3,
    "color_tracker_enable": true,
    "color_select": -1,
    "color_select_range": [
         {"v": [0, 70]}
    ],
    "color_enhance_blur": 3,
    "color_rate_range":[0.02, 1.0]
    "color_rate_threshold": 0.02,
    "color_buffer_size": 18,
    "color_lower_rate": 0.2,
    "color_upper_rate": 0.8,
    "color_track_direction": 0,
    "reg_factor":1,   "strides": [1],
    "within_period_threshold": 0.5
}
```

```
焊接001
{
    "focus_box": [0.556,0.594,0.609,0.681],
    "reg_factor": 1,
    "global_grap_speed": 16,
    "global_blur_type": "gaussian",
    "global_filter_kernel": 3,
    "stdwave_tracker_enable": true,
    "stdwave_sub_average": false,
    "stdwave_sigma_count": 3.2,
    "stdwave_window_secs": 20,
    "stdwave_distance_secs": 30,
    "stdwave_minstd_thresh": 1.1
}
```

```
翻边_001
{
    "focus_box": [0.386,0.447,0.551,0.734],
    "focus_box_repnum": 3,
    "reg_factor": 1,
    "strides": [1],
    "global_mask_enhance": { "erode": [5, 2], "dilate": [5, 2] },
    "global_bg_finding":false,
    "rmstill_frame_enable": true,
    "rmstill_bin_threshold": 10,
    "rmstill_rate_range": [0.1, 1.0],
    "rmstill_area_mode": 0,
    "within_period_threshold": 0.5,
    "tsm_last_enable": true,
    "tsm_last_threshold": 0.4,
    "tsm_last_smooth": true
}
```

```
敲框
{
    "focus_box": [0.328,0.266,0.634,0.746],
    "reg_factor": 1,
    "global_hdiff_rate": 0.3,
    "stdwave_tracker_enable": true,
    "stdwave_color_select":[
           {"h":[1, 20], "s": [20, 50], "v": [80, 255]}
    ],
    "stdwave_hsv_rate": 0.28,
    "stdwave_sub_average": true,
    "stdwave_sigma_count":5,
    "stdwave_window_secs": 4,
    "stdwave_distance_secs": 12,
    "stdwave_minstd_thresh": 0.8
}
```

```
锯切2梯柱_新版
{
    "focus_box": [0.477,0.14,0.595,0.453],
    "focus_box_repnum": 3,
    "reg_factor": 1,
    "strides": [1],
    "global_mask_enhance": { "erode": [3, 1], "dilate": [5, 2] },
    "rmstill_frame_enable": true,
    "rmstill_bin_threshold": 10,
    "rmstill_rate_range": [0.1, 1.0],
    "rmstill_area_mode": 1,
    "rmstill_white_rate": 0.1
}
```

```
3级，4级1次外踏杆冲孔_新版
{
    "focus_box": [0.479,0.658,0.638,0.860],
    "reg_factor": 1,
    "global_blur_type": "gaussian",
    "global_filter_kernel": 3,
    "global_hdiff_rate": 0.3,
    "stdwave_tracker_enable": true,
    "stdwave_sub_average": false,
    "stdwave_sigma_count": 2.8,
    "stdwave_window_secs": 0.4,
    "stdwave_distance_secs": 0.4,
    "stdwave_minstd_thresh":0.8
}

{
    "focus_box": [0.466,0.576,0.640,0.895],
    "focus_box_repnum": 3,
    "reg_factor": 1,
    "strides": [1],
    "global_mask_enhance": { "erode": [3, 1], "dilate": [3, 1] },
    "rmstill_frame_enable": true,
    "rmstill_bin_threshold": 10,
    "rmstill_rate_range": [0.02, 1.0],
    "rmstill_area_mode": 0,
    "rmstill_white_rate": 0.1
}
```
