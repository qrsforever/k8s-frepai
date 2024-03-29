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
    "global_thresh_binary": -1,
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
