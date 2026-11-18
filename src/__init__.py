"""src 包。

**刻意保持为空**（只有这段说明），原因有三：

1. 如果这里做便捷导出（`from src.xxx import ...`），`import src` 就会连锁导入
   torch / torchvision / pandas 等重依赖；任何只想用其中一个纯函数模块的脚本
   都会被拖慢，测试收集也变慢。
2. 更麻烦的是循环导入：`train.py` 依赖 dataset/models/losses/...，
   一旦 `__init__` 又反过来导入 `train`，就会出现 partially-initialized module。
3. `__init__.py` 的实际作用只是把 `src/` 标记为**常规包**（而非命名空间包），
   让 `python -m src.train`、pytest 的模块解析、以及 `from src.labels import ...`
   这种绝对导入在 Windows / 不同 workdir 下行为一致。

所以请始终按模块导入：
    from src.labels import load_split_df
    from src.metrics import macro_auc
而不要 `from src import *`。
"""
