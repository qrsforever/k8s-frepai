## proxy node

1. kubectl taint nodes proxy001  private/proxy:NoSchedule-
2. kubectl label --overwrite nodes proxy001 private/proxy="true"


## extra args

```json
{
    "focus_box": [0, 0, 1, 1],
    "focus_pts": [[0, 0], [1, 1], [2, 2]],
    "black_box": [1, 1, 1, 1],
    "check_box": [],
    "focus_box_repnum": 0,
    "angle": 0,
    "reg_factor": 1,
    "global_bg_finding": false,
    "global_grap_interval": -1,
    "global_grap_speed": -1,
    "global_blur_type": "none",
    "global_filter_kernel": 3,
    "global_remove_shadow": [3, 5, true],
    "global_feature_select": "mean",
    "global_hdiff_rate": 0.3,
    "global_lowest_bright": 50,
    "global_mask_enhance": {
        "erode": [3, 1],
        "dilate": [3, 1]
    },
}
```

### rmstill & color

```json
{
    "sort_brightness": false,
    "ef_is_send": false,
    "ef_url": "",
    "ef_alpha": 0.01,
    "ef_beta": 0.7,
    "ef_gamma": 0.8,
    "strides": [1, 2],
    "avg_pred_score": 0.2,
    "input_tile_shuffle": false,
    "within_period_threshold": 0.5,
    "tsm_last_threshold": 0.5,
    "tsm_last_smooth": false,
    "smooth_interpolate": false,
}
```

### rmstill

```json
{
    "rmstill_frame_enable": true,
    "rmstill_bin_threshold": 20,
    "rmstill_rate_range": [0.01, 0.8],
    "rmstill_brightness_norm": false,
    "rmstill_area_mode": 0,
    "rmstill_white_rate": 0.1,
    "rmstill_white_window": 10,
}
```

### color tracker

`color_select`: 0 - 9 (红,橙,.... 黑,白,灰）
`color_track_direction`: (0, 1, 2)

```json
{
    "color_tracker_enable": true,
    "color_select": 8,
    "color_rate_range": [0.2, 0.9],
    "color_enhance_blur": 25,
    "color_buffer_size": 12,
    "color_lower_rate": 0.2,
    "color_upper_rate": 0.8,
    "color_track_direction": 0,
    "color_select_range": [
        {"h":[1, 20], "s": [20, 50], "v": [80, 255]},
        {"v":[0, 70]}
    ]
}
```


### stdwave

```json
{
    "stdwave_tracker_enable": true,
    "stdwave_sub_average": true,
    "stdwave_sigma_count": 3.5,
    "stdwave_window_secs": 4,
    "stdwave_distance_secs": 12,
    "stdwave_minstd_thresh": 0.5,
    "stdwave_hsv_rate": 0.3,
    "stdwave_color_select": [
        {"h":[1, 20], "s": [20, 50], "v": [80, 255]},
        {"v":[0, 70]}
    ]
}
```


### featpeak

```json
{
    "featpeak_tracker_enbale": true,
    "featpeak_detect_trough": false,
    "featpeak_window_size": 15,
    "featpeak_data_normal": true,
    "featpeak_distance_size": 10,
    "featpeak_min_threshold": -1,
    "featpeak_relative_height": 0.9,
    "featpeak_height_minmax": [-1, -1],
    "featpeak_width_minmax": [-1, -1],
    "featpeak_prominence_minmax": [10, -1],
}
```

### diffimpulse

```json
{
    "rmstill_frame_enable": false,
    "color_tracker_enable": false,
    "featpeak_tracker_enbale": false,
    "stdwave_tracker_enable": false,
    "diffimpulse_tracker_enable": true,

    "diffimpulse_rate_threshold": 0.006,
    "diffimpulse_bin_threshold": 20,
    "diffimpulse_window_size": [10,5],
    "diffimpulse_blur_type": "gaussian",
    "debug_pre_write_video": false
}
```


```
1/2级外踏冲孔_新版
{
    "focus_box": [0.45,0.32,0.55,0.5],
    "focus_box_repnum": 3,
    "reg_factor": 1,
    "strides": [1],
    "rmstill_frame_enable": true,
    "rmstill_bin_threshold": 10,
    "rmstill_rate_threshold": 0.01,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0.15,
    "rmstill_white_window": 10,
    "tsm_last_enable": true,
    "tsm_last_threshold": 0.2,
    "tsm_last_smooth": true
}


3级外踏杆冲孔_新版
{
    "stdwave_tracker_enable": true,
    "focus_box": [0.459,0.597,0.566,0.789],
    "global_blur_type": "gaussian",
    "global_filter_kernel": 3,
    "reg_factor": 0.5,
    "stdwave_sub_average": true,
    "stdwave_sigma_count": 3,
    "stdwave_window_size": 3,
    "stdwave_distance_size": 10,
    "stdwave_minstd_thresh": 0.02
}


内踏杆锯，切边_新版
{
    "focus_box": [0.45,0.354,0.553,0.538],
    "focus_box_repnum": 3,
    "reg_factor": 1,
    "strides": [1],
    "rmstill_frame_enable": true,
    "rmstill_bin_threshold": 20,
    "rmstill_rate_threshold": 0.002,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0.1
}


折弯_新版
{
    "focus_box": [0.4,0,0.548,0.141],
    "focus_box_repnum": 5,
    "reg_factor": 1,
    "strides": [1],
    "rmstill_frame_enable": true,
    "rmstill_bin_threshold": 20,
    "rmstill_rate_threshold": 0.01,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 0,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0.15
}


敲框_wifi
{
    "focus_box": [0.25,0.1,0.721,0.921],
    "black_box": [0.575,0.757,0.729,1],
    "focus_box_repnum": 3,
    "reg_factor": 1,
    "strides": [1],
    "rmstill_frame_enable": true,
    "rmstill_bin_threshold": 40,
    "rmstill_rate_threshold": 0.007,
    "rmstill_area_mode": 1,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": true,
    "tsm_last_threshold": 0.3,
    "tsm_last_smooth": true
}


敲框2
{
    "focus_box": [0.3,0.05,0.717,0.91],
    "focus_box_repnum": 5,
    "reg_factor": 1,
    "strides": [1],
    "rmstill_frame_enable": true,
    "rmstill_bin_threshold": 28,
    "rmstill_rate_threshold": 0.02,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": true,
    "tsm_last_threshold": 0.3,
    "tsm_last_smooth": true
}


机器焊接1_老版wifi
{
    "stdwave_tracker_enable": true,
    "focus_box": [0.716,0.789,0.779,0.893],
    "global_blur_type": "gaussian",
    "global_filter_kernel": 3,
    "reg_factor": 1,
    "stdwave_sub_average": true,
    "stdwave_sigma_count": 3,
    "stdwave_window_size": 120,
    "stdwave_distance_size": 900,
    "stdwave_minstd_thresh": 1.1
}


机器焊接2_新版
{
    "stdwave_tracker_enable": true,
    "focus_box": [0.287,0.367,0.341,0.445],
    "global_blur_type": "gaussian",
    "global_filter_kernel": 3,
    "reg_factor": 1,
    "stdwave_sub_average": true,
    "stdwave_sigma_count": 3,
    "stdwave_window_size": 120,
    "stdwave_distance_size": 900,
    "stdwave_minstd_thresh": 1.1
}


人工焊接
{
    "focus_box": [0.5,0.2,0.75,0.9],
    "focus_box_repnum": 5,
    "reg_factor": 1,
    "strides": [1],
    "rmstill_frame_enable": true,
    "rmstill_brightness_norm": true,
    "rmstill_bin_threshold": 20,
    "rmstill_rate_threshold": 0.02,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": false,
    "tsm_last_threshold": 0.3,
    "tsm_last_smooth": true
}

包装_新版
{
    "focus_box": [0.234,0,0.979,0.987],
    "focus_box_repnum": 2,
    "reg_factor": 1,
    "strides": [1,2,4],
    "rmstill_frame_enable": true,
    "rmstill_brightness_norm": true,
    "rmstill_bin_threshold": 20,
    "rmstill_rate_threshold": 0.002,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 0,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": false,
    "tsm_last_threshold": 0.3,
    "tsm_last_smooth": true
}


外框清洁_新版
{
    "focus_box": [0.2,0.333,0.861,0.998],
    "black_box": [0.4,0.773,0.75,0.998],
    "focus_box_repnum": 2,
    "reg_factor": 1,
    "strides": [3],
    "rmstill_frame_enable": true,
    "rmstill_brightness_norm": true,
    "rmstill_bin_threshold": 20,
    "rmstill_rate_threshold": 0.002,
    "rmstill_area_mode": 1,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": true,
    "tsm_last_threshold": 0.2,
    "tsm_last_smooth": true
}

打铆钉_新版
{
    "focus_box": [0.115,0.654,0.448,1],
    "focus_box_repnum": 3,
    "reg_factor": 1,
    "strides": [3],
    "rmstill_frame_enable": true,
    "rmstill_brightness_norm": true,
    "rmstill_bin_threshold": 40,
    "rmstill_rate_threshold": 0.01,
    "rmstill_area_mode": 1,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": true,
    "tsm_last_threshold": 0.5,
    "tsm_last_smooth": true
}

锯切踏杆_新版
{
    "focus_box": [0.281,0.292,0.452,0.609],
    "focus_box_repnum": 2,
    "reg_factor": 1,
    "strides": [1],
    "rmstill_frame_enable": true,
    "rmstill_brightness_norm": true,
    "rmstill_bin_threshold": 20,
    "rmstill_rate_threshold": 0.02,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": true,
    "tsm_last_threshold": 0.5,
    "tsm_last_smooth": true
}


锯切2梯柱_新版
{
    "focus_box": [0.46,0.625,0.584,0.846],
    "black_box": [0.432,0.675,0.474,0.9],
    "focus_box_repnum": 1,
    "reg_factor": 1,
    "strides": [1],
    "rmstill_frame_enable": true,
    "rmstill_brightness_norm": true,
    "rmstill_bin_threshold": 23,
    "rmstill_rate_threshold": 0.003,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": true,
    "tsm_last_threshold": 0.3,
    "tsm_last_smooth": true
}


冲斜铁孔_新版
{
    "focus_box": [0.405,0.825,0.679,0.997],
    "focus_box_repnum":3,
    "reg_factor": 1,
    "strides": [2],
    "rmstill_frame_enable": true,
    "rmstill_brightness_norm": false,
    "rmstill_bin_threshold": 40,
    "rmstill_rate_threshold": 0.01,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": false,
    "tsm_last_threshold": 0.3,
    "tsm_last_smooth": true
}

冲锁具孔
{
    "focus_box": [0.4,0.685,0.596,0.997],
    "focus_box_repnum":3,
    "reg_factor": 1,
    "strides": [1],
    "rmstill_frame_enable": true,
    "rmstill_brightness_norm": false,
    "rmstill_bin_threshold":20,
    "rmstill_rate_threshold": 0.01,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": false,
    "tsm_last_threshold": 0.3,
    "tsm_last_smooth": true
}

内外梯柱打磨_新版
{
    "focus_box": [0.35,0.2,0.7,0.8],
    "focus_box_repnum":3,
    "reg_factor": 1,
    "strides": [2],
    "rmstill_frame_enable": true,
    "rmstill_brightness_norm": false,
    "rmstill_bin_threshold":30,
    "rmstill_rate_threshold": 0.004,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": true,
    "tsm_last_threshold": 0.5,
    "tsm_last_smooth": true
}

翻边2_新版非CPI
{
    "focus_box": [0.272,0.433,0.7,1],
    "focus_box_repnum":2,
    "reg_factor": 1,
    "strides": [2],
    "rmstill_frame_enable": true,
    "rmstill_brightness_norm": false,
    "rmstill_bin_threshold":10,
    "rmstill_rate_threshold": 0.1,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0,
    "tsm_last_enable": false,
    "tsm_last_threshold": 0.5,
    "tsm_last_smooth": true
}


坐姿切割_wifi更换视角
{
    "focus_box": [0.856,0.325,0.892,0.38],
    "focus_box_repnum":1,
    "reg_factor": 1,
    "strides": [2],
    "rmstill_frame_enable": true,
    "rmstill_brightness_norm": true,
    "rmstill_bin_threshold":5,
    "rmstill_rate_threshold": 0.001,
    "rmstill_area_mode": 0,
    "rmstill_noise_level": 1,
    "rmstill_filter_kernel": 3,
    "rmstill_white_rate": 0.2,
    "tsm_last_enable": true,
    "tsm_last_threshold": 0.3,
    "tsm_last_smooth": true
}
```


```
{
    "focus_box": [0.291,0.066,0.327,0.223],
    "global_hdiff_rate": 0.3,
    "stdwave_tracker_enable": true,
    "stdwave_sub_average": true,
    "stdwave_sigma_count": -2,
    "stdwave_window_secs": 4,
    "stdwave_distance_secs": 12,
    "stdwave_minstd_thresh": 3.5
}

```
