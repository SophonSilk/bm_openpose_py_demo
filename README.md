# bm_yolo_py_demo 使用说明

使用到的API主要可以参考文档 sdk路径/documents/Sophon_Inference_zh.pdf
开发环境配置参考文档：https://bitmain-doc.gitbook.io/bmnnsdk2-bm1684/on-linux

python版本支持 3.5 - 3.8

1. 安装Sophon Inference的python包

   a) 在sdk目录find -name "*sophon*.whl"

     选择对应的whl，比如在sc5、python3.5就选择./lib/sail/python3/pcie/py35/sophon-2.2.0-py3-none-any.whl

   b）安装

     先卸载之前可能安装的包

     pip3 uninstall sophon

     安装

     pip3 install sophon ./lib/sail/python3/pcie/py35/sophon-2.2.0-py3-none-any.whl

2. 编译c++ so
     运行编译脚本，目前仅支持pcie模式
     ./build_lib.sh
3. 执行demo

    python3 bm_openpose.py --config=config.json

    执行后会有弹窗显示检测结果
