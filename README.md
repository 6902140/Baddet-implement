# `Baddet-implement`
复现神經網絡目标检测后门攻击算法 BadDet


## 首先我們先學習後門投毒的開山之作`BadNets`:

踩到的坑:

- (1) 鏈接`libcudnn_ops_infer.so.8`失效

```
python: symbol lookup error: /data1/xzr/src/anaconda3/envs/dev/lib/python3.10/site-packages/torch/lib/../../nvidia/cudnn/lib/libcudnn_cnn_infer.so.8: undefined symbol: _Z20traceback_iretf_implPKcRKN5cudnn16InternalStatus_tEb, version libcudnn_ops_infer.so.8
```

原因:復現中使用的`cudnn`加速庫默認連接的`libcudnn_ops_infer.so.8`不是虛擬環境的so文件


```
如何定位錯誤

ldd /data1/xzr/src/anaconda3/envs/dev/lib/python3.10/site-packages/torch/lib/../../nvidia/cudnn/lib/libcudnn_cnn_infer.so.8
        linux-vdso.so.1 (0x00007ffc695b6000)
        libcudnn_ops_infer.so.8 => /data1/xzr/src/anaconda3/envs/dev/lib/python3.10/site-packages/nvidia/cudnn/lib/ # 找到指向的文件

        ... 

        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fab3b23d000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fab3b04b000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fab87103000)

# 解決: 添加環境變量,使用虛擬環境中的libcudnn_ops_infer.so.8
export LD_LIBRARY_PATH=/data1/xzr/src/anaconda3/envs/dev/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

```