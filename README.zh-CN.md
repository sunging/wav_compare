# WAV Compare

[English](README.md) · [比较规则](docs/comparison.md) · [版本发布](docs/releasing.md)

跨平台 WAV 音频比较工作台与命令行工具，覆盖 **8 kHz 起的音频**、多声道和长音频目录批处理。

![使用合成音频的工作台](docs/images/workbench.png)

## 安装与启动

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后执行：

```sh
uv tool install --python 3.12 git+https://github.com/sunging/wav_compare.git
wav-compare
```

支持 Python 3.11 及以上，开发环境固定为 3.12。找不到命令时执行 `uv tool update-shell` 并重新打开终端。无需 PyPI 即可安装。

首个正式版本 `0.1.0` 通过发布 PR 准备，合并之前 main 的包版本为未发布的 `0.0.0`。正式发布后才可在 Git URL 后添加 `@v0.1.0` 安装固定版本。

## 使用

- 打开或拖入两个文件/目录，点击“开始比较”。目录按相对路径递归配对，明确显示缺失、冲突和错误。
- 默认将 **B 重采样至 A 的采样率**，再估计固定时延。A 为 8 kHz 时目标就是 8 kHz。严格模式关闭两项处理。
- 支持逐声道比较、输入 `1:1,2:2` 自定义声道配对，或显式混为单声道。
- 查看波形、差值、分段统计、频谱和声谱图；支持选区分析、联动缩放、最大差异和前后差异定位。
- 试听 A、B、差值，支持选区循环、暂停和跳转；播放音量不改变分析。原始输入试听使用原文件的秒数，处理后试听使用对齐后的时间轴。
- 单文件导出反映当前选区分析；目录导出包含整批结果。JSON/CSV 保留实际参数和异常状态。
- 中英切换、系统/浅色/深色主题、路径与布局记忆。快捷键：Ctrl+Enter 比较，Esc 取消，Ctrl+0 重置缩放，Space 暂停/继续。

```sh
wav-compare compare a.wav b.wav --json report.json --csv report.csv
wav-compare compare before/ after/ --strict --threshold 0.0001
wav-compare compare a.wav b.wav --offset 160 --region 2 8
```

正时延代表 B 比 A 晚，单位为 A 采样率下的采样点。阈值随数值模式变化：浮点模式使用归一化幅度，PCM 模式使用整数码值。退出码 0 表示满足条件且阈值内，1 表示差异或缺失，2 表示错误或无法比较，130 表示取消。

## 长音频与限制

分块读取、流式重采样、后台分析和多级波形包络避免全量音频常驻内存。视口缓存预算为 256 MiB，后台最多两个任务；切换目录结果不会重新分析整个目录。

自动对齐仅补偿固定时延，不修正变速、时钟漂移或内容增删。无法可靠估计时显示“未可靠对齐”。频谱/声谱图按选区生成，长选区声谱图是最多 512 时间列的概览，缩小选区可查看细节。

Windows 已进行真实窗口和音频设备测试；三平台 CI 覆盖自动化测试。Linux 需要桌面 Qt 运行库及音频服务；无输出设备仍可分析和导出。无音频设备的 CI 不代表试听已验证。

性能见 [报告](docs/performance.md)。重采样缓存需要额外磁盘空间，不修改原音频。正常关闭、结果淘汰及“清理缓存”会释放临时数据。

## 开发与版本

```sh
uv sync --locked
uv run wav-compare
uv run pytest -q
uv run ruff check .
uv build
```

Conventional Commits 和 release-please 自动生成版本号与 CHANGELOG。发布 PR 不自动合并；合并是创建标签和 GitHub Release 的发布操作。详见 [发布流程](docs/releasing.md)。

来源为 `sunging/py_script` 的 `wav_compare_qt.py`，提交 `1780b081541e424d73ae2b0e24edebfc7f7ab4af`。新项目经原代码所有者授权使用 MIT 许可证。
