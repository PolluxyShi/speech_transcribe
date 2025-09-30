import whisperx
import gc
import torch

DEVICE = "cuda"
ALIGN_MODEL_DIR = "./models/align_models"
SUPPORTED_LANGUAGES = {
    # "en",      # English / 英语
    # "fr",      # French / 法语
    # "de",      # German / 德语
    # "es",      # Spanish / 西班牙语
    # "it",      # Italian / 意大利语
    # "ja",      # Japanese / 日语
    # "nl",      # Dutch / 荷兰语
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

for lang in SUPPORTED_LANGUAGES:
    print(f"Language: {lang}")
    model_a, metadata = whisperx.load_align_model(
        language_code=lang, 
        device=DEVICE,
        model_dir=ALIGN_MODEL_DIR
    )

    if hasattr(model_a, 'cpu'):
        model_a.cpu()
    del model_a

    if hasattr(metadata, 'cpu'):
        metadata.cpu()
    del metadata

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()