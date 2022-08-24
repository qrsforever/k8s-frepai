## proxy node

1. kubectl taint nodes proxy001  private.proxy:NoSchedule-
2. kubectl label --overwrite nodes proxy001 private/proxy="true"


## extra args

```json
{
    "rmstill_frame_enable": false,
    "color_tracker_enable": false,
    "stdwave_tracker_enable": true,
    "diffimpulse_tracker_enable": false,
    "stdwave_feature_select": "mean",
    "stdwave_hdiff_rate": 0.15,
    "stdwave_sub_average": true,
    "stdwave_sigma_count": 3.5,
    "stdwave_bg_window": 250,
    "stdwave_minstd_thresh": 0.5,
    "diffimpulse_rate_threshold": 0.006,
    "diffimpulse_bin_threshold": 20,
    "diffimpulse_window_size": [10,5],
    "diffimpulse_blur_type": "gaussian",
    "debug_pre_write_video": false
}
```
