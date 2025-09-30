import os
import io
# from pyannote.audio import Pipeline
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import tempfile
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, HTTPException, status, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import whisperx
import whisperx.diarize
import gc
import torch
import librosa
import numpy as np
from contextlib import asynccontextmanager
import aiofiles
import psutil
import GPUtil
import logging
from enum import Enum
from dataclasses import dataclass
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO


# 导入配置
from config import *

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    BUSY = "busy"

@dataclass
class ServiceState:
    status: ProcessingStatus = ProcessingStatus.IDLE
    current_audio_duration: float = 0.0
    max_queue_size: int = MAX_QUEUE_SIZE
    current_queue_size: int = 0
    gpu_utilization: float = 0.0
    last_update: float = 0.0

# class TranscriptionRequest(BaseModel):
#     min_speakers: Optional[int] = None
#     max_speakers: Optional[int] = None
#     return_char_alignments: bool = False

class TranscriptionResponse(BaseModel):
    status: str
    segments: List[Dict[str, Any]]
    processing_time: float
    audio_duration: float
    language: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# 全局变量
service_state = ServiceState()
models = {}

# 模型加载
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info(f"正在加载模型到设备: {DEVICE}")
    
    try:
        # 创建模型缓存目录
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        
        # 预加载说话人识别模型
        try:
            models['diarize_model'] = whisperx.diarize.DiarizationPipeline(
                model_name = DIARIZE_MODEL_PATH,
                device=DEVICE
            )
            logger.info("说话人识别模型预加载完成")
        except Exception as e:
            logger.warning(f"说话人识别模型加载失败: {e}")

        # 根据配置选择加载中文模型或外文模型：
        if USE_CHINESE_MODEL:
            logger.info("配置为使用中文模型")
            # 预加载中文ASR模型
            try:
                models['asr_model'] = AutoModel(
                    model="./models/SenseVoiceSmall",
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=DEVICE,
                    disable_update=True
                )
                logger.info("SenseVoice ASR模型预加载完成")
            except Exception as e:
                logger.warning(f"SenseVoice ASR模型加载失败: {e}")


        if USE_FOREIGN_MODEL:
            logger.info("配置为使用西文模型")

            # 加载主转录模型
            models['whisper_model'] = whisperx.load_model(
                MODEL_NAME,
                DEVICE,
                compute_type=COMPUTE_TYPE,
                download_root=MODEL_CACHE_DIR,
                # multilingual=True,  # 新增参数
                # max_new_tokens=128,  # 你可以根据需要调整
                # clip_timestamps=None,
                # hallucination_silence_threshold=None,
                # hotwords=None
            )
            logger.info(f"Whisper模型 {MODEL_NAME} 加载完成")
            
            # 预加载对齐模型（英语）
            try:
                models['align_model_en'], models['align_metadata_en'] = whisperx.load_align_model(
                    language_code="en", 
                    device=DEVICE,
                    model_dir=MODEL_CACHE_DIR
                )
                logger.info("英语对齐模型预加载完成")
            except Exception as e:
                logger.warning(f"英语对齐模型预加载失败: {e}")
            
        
        service_state.status = ProcessingStatus.IDLE
        logger.info("所有模型加载完成，服务准备就绪")
        
        # 启动GPU监控
        asyncio.create_task(monitor_gpu_usage())
        
        yield
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise e
    finally:
        # 清理资源
        cleanup_models()

def cleanup_models():
    """清理模型资源"""
    for name, model in models.items():
        if model is not None:
            try:
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model
            except:
                pass
    models.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

async def monitor_gpu_usage():
    """监控GPU使用率"""
    while True:
        try:
            if torch.cuda.is_available():
                # 获取CUDA_VISIBLE_DEVICES环境变量
                cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                gpus = GPUtil.getGPUs()
                target_gpu = gpus[0]
                if cuda_visible_devices:
                    target_gpu = gpus[int(cuda_visible_devices.split(',')[0])]

                service_state.gpu_utilization = target_gpu.memoryUtil * 100
                # print(f"当前GPU使用率: {service_state.gpu_utilization:.2f}%")
                # 如果GPU使用率过高，限制新请求
                if service_state.gpu_utilization > GPU_MEMORY_LIMIT * 100:
                    service_state.status = ProcessingStatus.BUSY
                elif service_state.current_queue_size == 0:
                    service_state.status = ProcessingStatus.IDLE
                else:
                    service_state.status = ProcessingStatus.PROCESSING
                        
            await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"GPU监控错误: {e}")
            await asyncio.sleep(10)

app = FastAPI(
    title="WhisperX语音转写服务",
    description="高性能语音转写服务，支持多种音频格式和说话人识别",
    version="1.0.0",
    lifespan=lifespan
)

# 线程池用于CPU密集型任务
#test diff
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def estimate_processing_time(audio_duration: float) -> float:
    """估计处理时间（基于经验值）"""
    return audio_duration * ESTIMATED_PROCESSING_RATIO

def get_audio_duration(audio_data: bytes) -> float:
    """获取音频时长"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            temp_file.flush()
            duration = librosa.get_duration(path=temp_file.name)
            os.unlink(temp_file.name)
            return duration
    except Exception as e:
        logger.error(f"获取音频时长失败: {e}")
        raise HTTPException(status_code=400, detail="无法解析音频文件")

def is_supported_format(filename: str) -> bool:
    """检查文件格式是否支持"""
    file_ext = os.path.splitext(filename.lower())[1]
    return file_ext in SUPPORTED_FORMATS

def convert_audio_format(audio_data: bytes, original_filename: str) -> bytes:
    """
    将音频bytes数据转换为WAV格式的bytes数据
    
    Args:
        audio_data: 原始音频的bytes数据
        original_filename: 原始文件名（用于确定格式）
    
    Returns:
        wav_data: WAV格式的bytes数据
    """
    if not is_supported_format(original_filename):
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件格式。支持格式: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    try:
        audio_buffer = BytesIO(audio_data)
        file_extension = original_filename.split('.')[-1].lower()
        
        audio = AudioSegment.from_file(audio_buffer, format=file_extension)
        
        wav_buffer = BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_bytes = wav_buffer.getvalue()
        
        audio_buffer.close()
        wav_buffer.close()
        
        return wav_bytes
            
    except Exception as e:
        logger.error(f"音频格式转换失败: {e}")
        raise HTTPException(status_code=400, detail="音频格式转换失败")

def process_audio(
    audio_data: bytes, 
    language_code: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_char_alignments: bool = False
) -> Dict[str, Any]:
    """处理音频的核心函数"""
    start_time = time.time()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            # 保存音频到临时文件
            temp_file.write(audio_data)
            temp_file.flush()

            if language_code in ["zh", "yue"]:
                logger.info("使用中文处理流程")

                language = language_code
                audio = None  # 保证audio变量定义
                # 1.说话人分离（pyannote.audio）
                try:
                    diarize_pipeline = models.get('diarize_model')
                    asr_model = models.get('asr_model')
                    if diarize_pipeline is None:
                        raise RuntimeError("DiarizationPipeline未加载")
                    if asr_model is None:
                        raise RuntimeError("SenseVoice ASR模型未加载")
                    diarize_result = diarize_pipeline(temp_file.name)
                    diarize_segments = []
                    for _,row in diarize_result.iterrows():
                        diarize_segments.append({
                            'start': row['start'],
                            'end': row['end'],
                            'speaker': row['speaker']
                        })
                except Exception as e:
                    logger.error(f"中文说话人分离失败: {e}")
                    raise RuntimeError(f"中文说话人分离失败: {e}")

                # 2. 分段转录
                try:
                    audio_seg = AudioSegment.from_wav(temp_file.name)
                    audio = audio_seg.get_array_of_samples()  # 用于后续audio_duration
                    segments = []
                    # 兼容 DataFrame 或 list of dict
                    if hasattr(diarize_segments, 'iterrows'):
                        diarize_iter = diarize_segments.iterrows()
                    else:
                        diarize_iter = enumerate(diarize_segments)
                    for idx, seg_info in diarize_iter:
                        try:
                            # DataFrame 行
                            if hasattr(seg_info, 'keys') and 'start' in seg_info:
                                start_s = seg_info['start']
                                end_s = seg_info['end']
                                speaker = seg_info['speaker']
                            # Series
                            elif hasattr(seg_info, 'to_dict'):
                                seg_dict = seg_info.to_dict()
                                start_s = seg_dict.get('start')
                                end_s = seg_dict.get('end')
                                speaker = seg_dict.get('speaker')
                            # list/tuple
                            elif isinstance(seg_info, (list, tuple)) and len(seg_info) > 1:
                                seg_dict = seg_info[1] if isinstance(seg_info[1], dict) else dict(seg_info[1])
                                start_s = seg_dict.get('start')
                                end_s = seg_dict.get('end')
                                speaker = seg_dict.get('speaker')
                            else:
                                logger.error(f"seg_info内容: {seg_info}, 类型: {type(seg_info)}")
                                raise RuntimeError("分段数据格式异常")
                            logger.info(f"分段: start={start_s}, end={end_s}, speaker={speaker}")
                            seg = audio_seg[int(float(start_s)*1000):int(float(end_s)*1000)]
                        except Exception as e:
                            logger.error(f"diarize_segments: {diarize_segments}")
                            logger.error(f"seg_info: {seg_info}")
                            logger.error(f"分段处理失败1: {e}")
                            raise RuntimeError(f"分段处理失败1: {e}")
                        
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                seg.export(tmp.name, format="wav")
                                seg_path = tmp.name
                            res = asr_model.generate(
                                input=seg_path,
                                language="auto",
                                use_itn=True,
                                batch_size_s=60,
                                merge_vad=True,
                                merge_length_s=15
                            )
                        except Exception as e:
                            logger.error(f"res: {res}")
                            logger.error(f"分段处理失败2: {e}")
                            raise RuntimeError(f"分段处理失败2: {e}")
                        
                        raw_text = res[0]["text"]
                        text = rich_transcription_postprocess(raw_text)
                        segments.append({
                            "start": float(f"{start_s:.2f}"),
                            "end": float(f"{end_s:.2f}"),
                            "speaker": speaker,
                            "text": text
                        })
                        os.remove(seg_path)
                except Exception as e:
                    logger.error(f"中文分段转录失败: {e}")
                    raise RuntimeError(f"中文分段转录失败: {e}")

                processing_time = time.time() - start_time
                # audio 可能为None（如果分段转录失败），此时用0
                audio_length = len(audio) if audio is not None else 0
                return {
                    "status": "success",
                    "segments": segments,
                    "processing_time": processing_time,
                    "audio_duration": audio_length / SAMPLE_RATE,
                    "language": language
                }

            else:
                logger.info("使用西文处理流程")

                # 1. 转录
                audio = whisperx.load_audio(temp_file.name)
                result = models['whisper_model'].transcribe(audio, batch_size=BATCH_SIZE, language=language_code)
                language = result["language"]

                # 2. 对齐
                if language in ['en'] and models.get('align_model_en'):
                    model_a = models['align_model_en']
                    metadata = models['align_metadata_en']                    
                else:
                    model_a, metadata = whisperx.load_align_model(
                        language_code=language, 
                        device=DEVICE,
                        model_dir=MODEL_CACHE_DIR
                    )
                
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, DEVICE, 
                    return_char_alignments=return_char_alignments
                )
                
                # 3. 说话人识别                
                if models.get('diarize_model'):
                    diarize_segments = models['diarize_model'](
                        audio, 
                        min_speakers=min_speakers, 
                        max_speakers=max_speakers
                    )
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                
                processing_time = time.time() - start_time
                
                return {
                    "status": "success",
                    "segments": result["segments"],
                    "processing_time": processing_time,
                    "audio_duration": len(audio) / SAMPLE_RATE,
                    "language": language
                }
            
    except Exception as e:
        logger.error(f"音频处理失败: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_file.name)
        except:
            pass


@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "WhisperX语音转写服务",
        "version": "1.0.0",
        "status": service_state.status.value,
        "models_loaded": len(models),
        "device": DEVICE
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:        
        # 检查服务状态，确保没有因为异常被设置为BUSY
        current_status = service_state.status
        
        return {
            "status": current_status.value,
            "gpu_utilization": service_state.gpu_utilization,
            "queue_size": service_state.current_queue_size,
            "max_queue_size": service_state.max_queue_size,
            "device": DEVICE,
            "models_loaded": list(models.keys())  # 显示已加载的模型
        }
        
    except Exception as e:
        logger.error(f"健康检查异常: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": f"健康检查异常: {str(e)}"
            }
        )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language_code: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    return_char_alignments: bool = False
):
    """语音转写主接口"""
    logger.info(f"收到转写请求: 文件名={file.filename}, language_code={language_code}, min_speakers={min_speakers}, max_speakers={max_speakers}")
    # 检查服务状态
    if service_state.status == ProcessingStatus.BUSY:
        logger.warning("服务繁忙，拒绝请求")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务繁忙，请稍后重试"
        )
    if service_state.current_queue_size >= service_state.max_queue_size:
        logger.warning("请求队列已满，拒绝请求")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="请求队列已满，请稍后重试"
        )
    # 读取音频数据
    try:
        audio_data = await file.read()
        logger.info(f"读取音频文件成功，字节数: {len(audio_data)}")
        if len(audio_data) == 0:
            logger.error("音频文件为空")
            raise HTTPException(status_code=400, detail="音频文件为空")
    except Exception as e:
        logger.error(f"读取音频文件失败: {e}")
        raise HTTPException(status_code=400, detail=f"读取音频文件失败: {str(e)}")
    # 转换音频格式
    try:
        wav_data = convert_audio_format(audio_data, file.filename)
        logger.info(f"音频格式转换成功，转换后字节数: {len(wav_data)}")
        audio_duration = get_audio_duration(wav_data)
        logger.info(f"音频时长: {audio_duration:.2f} 秒")
    except Exception as e:
        logger.error(f"音频处理失败: {e}")
        raise HTTPException(status_code=400, detail=f"音频处理失败: {str(e)}")
    # 检查音频时长
    if audio_duration > MAX_AUDIO_DURATION:
        logger.warning(f"音频时长超限: {audio_duration:.2f} > {MAX_AUDIO_DURATION}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"音频时长超过限制（{MAX_AUDIO_DURATION}秒）"
        )
    estimated_time = estimate_processing_time(audio_duration)
    if estimated_time > 6000:  # 100分钟
        logger.warning(f"音频预计处理时间过长: {estimated_time:.1f}秒")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"音频时长过长（预计处理时间{estimated_time:.1f}秒），请使用较短的音频"
        )
    # 更新服务状态
    service_state.status = ProcessingStatus.PROCESSING
    service_state.current_audio_duration = audio_duration
    service_state.current_queue_size += 1
    try:
        logger.info("开始线程池音频处理...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            thread_pool, 
            process_audio, 
            wav_data, 
            language_code,
            min_speakers, 
            max_speakers, 
            return_char_alignments
        )
        logger.info(f"音频处理完成，状态: {result.get('status')}")
        if result["status"] == "error":
            logger.error(f"音频处理错误: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result["error"])
        logger.info("转写成功，返回结果")
        return TranscriptionResponse(**result)
    except Exception as e:
        logger.exception(f"转录处理错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    finally:
        # 更新服务状态
        service_state.current_queue_size -= 1
        if service_state.current_queue_size == 0:
            service_state.status = ProcessingStatus.IDLE
        else:
            service_state.status = ProcessingStatus.PROCESSING

@app.post("/transcribe/bytes")
async def transcribe_audio_bytes(
    background_tasks: BackgroundTasks,
    audio_bytes: bytes = File(...),
    filename: str = "audio.wav",
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
):
    """字节输入接口"""
    # 创建UploadFile对象以复用现有逻辑
    upload_file = UploadFile(filename=filename, file=io.BytesIO(audio_bytes))
    return await transcribe_audio(background_tasks, upload_file, min_speakers, max_speakers)

# @app.get("/models/preload/{language_code}")
# async def preload_align_model(language_code: str):
#     """预加载对齐模型"""
#     try:
#         model_a, metadata = whisperx.load_align_model(
#             language_code=language_code, 
#             device=DEVICE,
#             model_dir=MODEL_CACHE_DIR
#         )
#         models[f'align_model_{language_code}'] = model_a
#         models[f'align_metadata_{language_code}'] = metadata
#         return {"status": "success", "message": f"对齐模型 {language_code} 预加载完成"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"预加载失败: {str(e)}")

@app.get("/config")
async def get_config():
    """获取当前配置"""
    return {
        "max_queue_size": MAX_QUEUE_SIZE,
        "max_audio_duration": MAX_AUDIO_DURATION,
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "compute_type": COMPUTE_TYPE,
        "device": DEVICE,
        "supported_formats": SUPPORTED_FORMATS,
        "sample_rate": SAMPLE_RATE
    }

if __name__ == "__main__":
    # 在启动前确保日志配置生效
    logging.getLogger().setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.INFO)

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("可用GPU数量:", torch.cuda.device_count())
    print("当前使用GPU索引:", torch.cuda.current_device())
    print("当前GPU名称:", torch.cuda.get_device_name(0))
    import uvicorn
    
    # 使用配置文件中的服务器设置
    uvicorn.run(
        # "main:app",
        app,
        host=HOST,
        port=PORT,
        workers=WORKERS,
        log_level=LOG_LEVEL
    )
