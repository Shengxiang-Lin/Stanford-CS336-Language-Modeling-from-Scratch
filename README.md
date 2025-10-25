# Stanford-CS336-Language-Modeling-from-Scratch

## Set up a proxy server to access external resources (when local Clash and server are not on the same LAN)    
Establish SSH remote forwarding on the local terminal     
```
ssh -o ServerAliveInterval=60 -R 7891:localhost:7890 lsx@10.130.138.35
```    
Reset proxy environment variables to avoid conflicts with old configurations     
```    
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   
```     
Configure proxy to point to the forwarded port 7891     
```   
export http_proxy=http://127.0.0.1:7891   
export https_proxy=http://127.0.0.1:7891  
```    
Test basic connectivity with curl  
```    
curl -v -x http://127.0.0.1:7891 https://huggingface.co   
```  