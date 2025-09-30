# 语音转录服务

一个支持中文和西文语音转录的服务，具备说话人分离功能。

## 1. 部署

### 1.1 环境要求
- Conda
- Python 3.9
- CUDA 支持

### 1.2 安装步骤

1. **创建虚拟环境并激活**
   ```bash
   conda create --name speech_transcribe python=3.9
   conda activate speech_transcribe
   ```

2. **安装PyTorch**
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   ```
   > 注：开发环境使用torch==2.6.0+cu124，生产环境部署时可根据实际情况在PyTorch官网安装对应版本

3. **安装其他依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **下载模型**
   
   项目已包含以下预下载模型：
   - **说话人分离模型**：`models/models--pyannote--speaker-diarization-3.1`
   - **中文转录模型**：`models/SenseVoiceSmall`, `models/models--jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn`, `models/wav2vec2_fairseq_base_ls960_asr_ls960.pth`
   - **西文转录模型**：`models/models--Systran--faster-whisper-large-v3`
   
   下载剩余的对齐模型：
   ```bash
   python download_align_models.py
   ```

## 2. 配置参数

在`config.py`中的重要配置参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `USE_CHINESE_MODEL` | 是否使用中文模型 | `True` |
| `USE_FOREIGN_MODEL` | 是否使用西文模型 | `True` |
| `HOST` | 服务地址 | `"0.0.0.0"` |
| `PORT` | 服务端口 | `8000` |

## 3. 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

> 通过`CUDA_VISIBLE_DEVICES`指定GPU，目前仅支持单GPU


## 功能特性

- ✅ 说话人分离
- ✅ 中文语音转录
- ✅ 西文语音转录
- ✅ 文本对齐
- 🔄 支持GPU加速

---

如有问题，请参考相关模型文档或联系开发团队。