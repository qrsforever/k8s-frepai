## proxy node

1. kubectl taint nodes proxy001  private/proxy:NoSchedule-
2. kubectl label --overwrite nodes proxy001 private/proxy="true"


## extra args

blur: averaging, median, gaussian

```json
{
    "focus_box": [0, 0, 1, 1],
    "black_box": [0, 0, 0, 0],
    "global_remove_shadow":[3, 7, false],
    "global_grap_interval": -1,
    "global_blur_type": "none",
    "global_filter_kernel": 3,
    "global_feature_select": "mean",
    "global_hdiff_rate": [0.1 0.3],
    "global_bg_window": 150,
    "global_bg_atonce": true,
    "global_bg_finding": false,
    "debug_write_video": false
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

### stdwave

```json
{
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
    "diffimpulse_tracker_enable": true,
    "diffimpulse_rate_threshold": 0.006,
    "diffimpulse_bin_threshold": 20,
    "diffimpulse_window_size": [10,5],
    "diffimpulse_blur_type": "gaussian",
    "debug_pre_write_video": false
}
```

### direction

```json
{
    "focus_box": [0.229,0.474,0.443,0.581],
    "global_feature_select": "mean",
    "global_hdiff_rate": 0.3,
    "direction_tracker_enable": true,
    "direction_arrow": "lr"
}

{
    "focus_box": [0.271,0.506,0.470,0.604],
    "global_blur_type": "median",
    "global_filter_kernel": 5,
    "global_feature_select": "mean",
    "global_hdiff_rate": 0.3,
    "debug_pre_write_video": true,
    "direction_tracker_enable": true,
    "direction_arrow": "lr"
}

{
    "focus_box": [0.277,0.539,0.455,0.597],
    "global_blur_type": "median",
    "global_filter_kernel": 5,
    "global_feature_select": "mean",
    "global_hdiff_rate": 0.3,
    "debug_pre_write_video": true,
    "direction_tracker_enable": true,
    "direction_arrow": "lr"
}
```
