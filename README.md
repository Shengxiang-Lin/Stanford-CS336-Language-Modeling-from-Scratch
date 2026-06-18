# Stanford-CS336-Language-Modeling-from-Scratch
## [Course Video](https://www.bilibili.com/video/BV14X41zZEPh/)
## [Course Website](https://cs336.stanford.edu/)

## Set up a proxy server to access external resources (when local Clash and server are not on the same LAN)    
Establish SSH remote forwarding on the local terminal     
```
ssh -o ServerAliveInterval=60 -R 8888:localhost:7897 lsx@10.130.138.35
```    
Reset proxy environment variables to avoid conflicts with old configurations     
```    
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   
```     
Configure proxy to point to the forwarded port 8888     
```   
export http_proxy=http://127.0.0.1:8888   
export https_proxy=http://127.0.0.1:8888  
```    
Test basic connectivity with curl  
```    
curl -v -x http://127.0.0.1:8888 https://huggingface.co   
```  
