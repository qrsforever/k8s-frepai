## proxy node

1. kubectl taint nodes proxy001  private/proxy:NoSchedule-
2. kubectl label --overwrite nodes proxy001 private/proxy="true"


## extra args

```json
{
    "focus_box": [0, 0, 1, 1],
    "black_box": [1, 1, 1, 1],
    "global_grap_interval": -1,
    "global_blur_type": "none",
    "global_filter_kernel": 3,
    "global_feature_select": "mean",
    "global_hdiff_rate": 0.3,
    "global_bg_window": 150,
    "global_bg_atonce": true,
}
```

### featpeak

```json
{
    "rmstill_frame_enable": false,
    "color_tracker_enable": false,
    "diffimpulse_tracker_enable": false,
    "featpeak_tracker_enbale": true,
    "stdwave_tracker_enable": false,

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

### stdwave

```json
{
    "rmstill_frame_enable": false,
    "color_tracker_enable": false,
    "diffimpulse_tracker_enable": false,
    "featpeak_tracker_enbale": false,
    "stdwave_tracker_enable": true,

    "stdwave_sub_average": true,
    "stdwave_sigma_count": 3.5,

    "stdwave_window_size": 50,
    "stdwave_distance_size": 100,
    "stdwave_minstd_thresh": 0.5,
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
