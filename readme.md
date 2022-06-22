## proxy node

1. kubectl taint nodes proxy001  private.proxy:NoSchedule-
2. kubectl label --overwrite nodes proxy001 private/proxy="true"
