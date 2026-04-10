当集群掉了的时候，先clash换一个端口，然后重新连接就可以使用了

### GPT账号

2698173554@qq.com

Wzg2003318666.

### 本机

ssh -N -R 17897:127.0.0.1:7897 -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3 用户名@远程服务器

> ssh -N -R 17897:127.0.0.1:7897 -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3 root@192.168.29.224

密码三个空格

### 远程

export http_proxy=http://127.0.0.1:17897
export https_proxy=http://127.0.0.1:17897
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY=http://127.0.0.1:17897



curl -v --max-time 10 -x http://127.0.0.1:17897 https://github.com

然后登录codex~

（可以试试删掉vscode的.vscode-server文件）

### 不需要确认指令

codex --ask-for-approval never --sandbox danger-full-access
