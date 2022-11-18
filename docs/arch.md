```txt
                                                                           
╭─────────────╮                                                                         ╭──────────────────╮
│             │                                                                         │      COS         │
│  摄像头     │                                                                         │  对象存储服务器  │
│             │                                                                  ╭─────>│                  │
╰──────┬──────╯                                                                  │      │                  │
       │                  srs-rtc                  srs-app                       │      ╰──────────────────╯
       │                 ╭───────────────╮         ╭───────────────╮    put      │             |
       ╰───────────────> │  SRS/RTMP     │         │   SRS/Hooks   │─────────────╯             |
                         │  流媒体服务   │ ───────>│               │{"video": xxx.mp4"}        |
 推流时应该已确定视      ╰───────────────╯         ╰───────────────╯                           |
 内容（工序，工单等）         │             on_publish│    │   │                               |
 它们共同决定参数ID           │                       │    │   │on_dvr                         |
                              │ redis                 │    │   │                               |
                              │                       ╰────┤通 │                               |
             {"video": url"}  │               on_unpublish │   │                               |
                              │                            │知 │                               |
                              v                            │   │              +----------------+
                      ╭────────────────╮                   │服 │              |
                      │      SQL       │<──────╮           │   │              |
            ╭────────>│  业务服务器    │       │ 更新      │务 │              |
            │         │                ├───╮   │ 数据      │   │              |
            │         ╰───────┬────────╯   │   │           │器 │              |
            │            |    │ 定时任务   │   │           │   │              |
            │            |    ╰────────────╯   │    结果   │   │              |
            │            |          线 │ 配    │   ╭───────┼───┼──────────╮   |
            │        获取|          上 │ 置    │   │       │   │          │   |
            │        历史|             │       │   v       │   │          │   |
            │        数据|      参数ID │   ╭───┴───────╮   │   │          │   |
            │            |             ╰──>│   Kafka   │───┼───┼──────╮   │   |
            │            |       ╭────────>│ 消息队列  │   │   │      │   │   |
            │            |       │         ╰───────────╯   │   │      │   │   |
            │            v       │ 调试参数      ^  ^      │   │      v   │   v get
            │       ╭────────────────╮           │  │      │   │╭──────────────────╮
            │       │     Admin      │           │  │      │   ││    算法服务器    │
            │       │  开发调试平台  │           │  ╰──────╯   ││     frepai       │
            │       │                │           ╰─────────────╯│                  │
            │       ╰──────┬──────┬──╯                          ╰──────────────────╯
            │              │ 部署 │
            │              ╰──────╯
            │ 更新线上配置     │
            ╰──────────────────╯

```