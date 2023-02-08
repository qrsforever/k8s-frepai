git clone --depth 1 https://github.91chi.fun/https://github.com/ossrs/srs.git -b v5.0.25 srs5.0
git clone --depth 1 https://github.com/ossrs/srs.git -b v5.0-a2 srs5.0

kubectl delete secret frepai-srs-secret -n frepai
kubectl create secret tls frepai-srs-secret -n frepai --key srs-server.key  --cert srs-server.crt
kubectl get secret  frepai-srs-secret  -n frepai
kubectl describe secret frepai-srs-secret -n frepai

```
http://101.42.139.3:30808/players/rtc_player.html?vhost=seg.30s&app=live&stream=00335ebc0407&server=101.42.139.3:31985&port=30808&autostart=true&schema=http
```

https://stackoverflow.com/questions/7580508/getting-chrome-to-accept-self-signed-localhost-certificate
