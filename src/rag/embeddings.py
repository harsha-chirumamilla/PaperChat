import torch
import threading
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from src.common.config import Config

logger = logging.getLogger(__name__)


def get_device():
    """Returns 'cuda' if available, else 'cpu'. Safe for multiprocessing."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# Global embedding function cache (lazy-loaded)
_embedding_cache = None
_embedding_lock = threading.Lock()


def get_embedding_function():
    """Get or create the embedding function (cached, thread-safe) with FP16 optimization."""
    global _embedding_cache
    
    if _embedding_cache is None:
        with _embedding_lock:
            if _embedding_cache is None:
                logger.info("Initializing embeddings: %s", Config.EMBEDDING_MODEL_ID)
                device = get_device()
                logger.info("Using device: %s", device)
                
                # Use FP16 for GPU (30-50% faster) or FP32 for CPU
                model_kwargs = {'device': device}
                if device == 'cuda':
                    import torch
                    model_kwargs['model_kwargs'] = {'torch_dtype': torch.float16}
                    logger.info("Using FP16 precision for faster inference")
                
                _embedding_cache = HuggingFaceEmbeddings(
                    model_name=Config.EMBEDDING_MODEL_ID,
                    model_kwargs=model_kwargs,
                    encode_kwargs={'normalize_embeddings': True}
                )
    
    return _embedding_cache
