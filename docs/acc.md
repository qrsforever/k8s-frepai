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
