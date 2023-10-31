# 3D-Reconstruction
大工件三维重建
# 大工件定位

## end to end 流程图



```mermaid
graph LR
A[图片读取]--> B[立体校正]-->c[语义分割]
-->d[三维重建]-->e[坐标系转换]-->f[融合]
-->g[信息输出]

```

