import os

# 服务配置
MAX_QUEUE_SIZE = 5
MAX_AUDIO_DURATION = 6000  # 最大音频时长（秒）TODO:待测
ESTIMATED_PROCESSING_RATIO = 0.06  # 处理时间估计系数

# 模型配置
MODEL_NAME = "large-v3"
BATCH_SIZE = 16
COMPUTE_TYPE = "float16"  # 或 "int8"
MODEL_CACHE_DIR = "./models"
ALIGN_MODEL_DIR = "./models/align_models"
DIARIZE_MODEL_PATH = "./models/models--pyannote--speaker-diarization-3.1/snapshots/84fd25912480287da0247647c3d2b4853cb3ee5d/config.yaml"
USE_CHINESE_MODEL = True  # 是否使用中文模型
USE_FOREIGN_MODEL = True  # 是否使用西文模型

# GPU配置
DEVICE = "cuda" # if torch.cuda.is_available() and os.getenv("USE_GPU", "true").lower() == "true" else "cpu"
GPU_MEMORY_LIMIT = 0.85  # GPU内存使用率限制

# 服务器配置
HOST = "0.0.0.0"
PORT = 8000
WORKERS = 1
LOG_LEVEL = "info"

# 音频配置
SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.mp4', '.avi', '.mov', '.amr', '.amrwb'] # 暂不支持 '.wmv', '.evs', '.pcmu', '.wma'
SAMPLE_RATE = 16000

# 线程池配置
MAX_WORKERS = 2

# 语言配置
SUPPORTED_CHINESE_LANGUAGES = {
    "zh",      # Chinese / 中文
    "yue"      # Cantonese / 粤语
}

SUPPORTED_FOREIGN_LANGUAGES = {
    "en",      # English / 英语
    "fr",      # French / 法语
    "de",      # German / 德语
    "es",      # Spanish / 西班牙语
    "it",      # Italian / 意大利语
    "ja",      # Japanese / 日语
    "nl",      # Dutch / 荷兰语
    "uk",      # Ukrainian / 乌克兰语
    "pt",      # Portuguese / 葡萄牙语
    "ar",      # Arabic / 阿拉伯语
    "cs",      # Czech / 捷克语
    "ru",      # Russian / 俄语
    "pl",      # Polish / 波兰语
    "hu",      # Hungarian / 匈牙利语
    "fi",      # Finnish / 芬兰语
    "fa",      # Persian / 波斯语
    "el",      # Greek / 希腊语
    "tr",      # Turkish / 土耳其语
    "da",      # Danish / 丹麦语
    "he",      # Hebrew / 希伯来语
    "vi",      # Vietnamese / 越南语
    "ko",      # Korean / 韩语
    "ur",      # Urdu / 乌尔都语
    "te",      # Telugu / 泰卢固语
    "hi",      # Hindi / 印地语
    "ca",      # Catalan / 加泰罗尼亚语
    "ml",      # Malayalam / 马拉雅拉姆语
    "no",      # Norwegian (Bokmål) / 挪威语（书面挪威语）
    "nn",      # Norwegian (Nynorsk) / 挪威语（新挪威语）
    "sk",      # Slovak / 斯洛伐克语
    "sl",      # Slovenian / 斯洛文尼亚语
    "hr",      # Croatian / 克罗地亚语
    "ro",      # Romanian / 罗马尼亚语
    "eu",      # Basque / 巴斯克语
    "gl",      # Galician / 加利西亚语
    "ka",      # Georgian / 格鲁吉亚语
    "lv",      # Latvian / 拉脱维亚语
    "tl",      # Tagalog / 他加禄语
}