"""
FastAPI server for serving KuiperAI models
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import sys
import os
import json

sys.path.append('..')
from src.models.transformer import Transformer
from src.data.dataset import TextDataset

app = FastAPI(title="KuiperAI API", version="1.0.0")


class TextRequest(BaseModel):
    """Request model for text generation"""
    prompt: str
    max_length: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = 50


class TextResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str
    tokens_generated: int
    model_version: str


class ClassificationRequest(BaseModel):
    """Request model for classification"""
    text: str


class ClassificationResponse(BaseModel):
    """Response model for classification"""
    label: str
    confidence: float
    all_scores: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str


class ModelConfig(BaseModel):
    """Model configuration"""
    checkpoint_path: str = "checkpoints/best_model.json"
    vocab_path: str = "knowledge/vocab.json"
    model_params: Dict = {
        "vocab_size": 10000,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "d_ff": 2048,
        "max_seq_len": 512
    }


# Global model instance
model = None
vocab = None
idx_to_token = None
config = ModelConfig()


def load_model_from_checkpoint(checkpoint_path: str, vocab_path: str, model_params: Dict):
    """Load model from checkpoint"""
    global model, vocab, idx_to_token
    
    print(f"Loading vocabulary from {vocab_path}")
    if os.path.exists(vocab_path):
        vocab = TextDataset.load_vocab(vocab_path)
        idx_to_token = {v: k for k, v in vocab.items()}
        print(f"✓ Vocabulary loaded: {len(vocab)} tokens")
    else:
        print(f"⚠ Vocabulary not found, using default")
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        idx_to_token = {v: k for k, v in vocab.items()}
    
    # Update vocab size
    model_params['vocab_size'] = len(vocab)
    
    print(f"Creating model with params: {model_params}")
    model = Transformer(**model_params)
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        # Load model weights
        if 'model_state' in checkpoint:
            params = model.parameters()
            for i, param in enumerate(params):
                if f'param_{i}' in checkpoint['model_state']:
                    param.data = np.array(checkpoint['model_state'][f'param_{i}'], dtype=np.float32)
            print(f"✓ Model weights loaded from checkpoint")
        else:
            print("⚠ No model_state in checkpoint, using random weights")
    else:
        print(f"⚠ Checkpoint not found at {checkpoint_path}, using random weights")
    
    print("✓ Model loaded successfully!")
    return model, vocab, idx_to_token


@app.on_event("startup")
async def startup_event():
    """Load model on server startup"""
    try:
        load_model_from_checkpoint(
            config.checkpoint_path,
            config.vocab_path,
            config.model_params
        )
    except Exception as e:
        print(f"⚠ Failed to load model: {e}")
        print("Server will start but model endpoints will not work")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.post("/generate", response_model=TextResponse)
async def generate_text(request: TextRequest):
    """
    Generate text from a prompt
    
    Args:
        request: TextRequest with prompt and generation parameters
    
    Returns:
        Generated text and metadata
    """
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize prompt
        tokens = [vocab.get(word, vocab.get('<UNK>', 1)) for word in request.prompt.split()]
        
        if not tokens:
            tokens = [vocab.get('<SOS>', 2)]
        
        # Add SOS token if not present
        if tokens[0] != vocab.get('<SOS>', 2):
            tokens = [vocab.get('<SOS>', 2)] + tokens
        
        tokens = np.array([tokens], dtype=np.int32)
        
        # Generate
        generated = model.generate(
            tokens,
            max_length=min(request.max_length, model.max_seq_len),
            temperature=request.temperature
        )
        
        # Decode
        generated_tokens = []
        for token_id in generated[0]:
            if token_id == vocab.get('<EOS>', 3):
                break
            if token_id not in [vocab.get('<PAD>', 0), vocab.get('<SOS>', 2)]:
                generated_tokens.append(idx_to_token.get(int(token_id), '<UNK>'))
        
        generated_text = ' '.join(generated_tokens)
        
        return {
            "generated_text": generated_text,
            "tokens_generated": len(generated_tokens),
            "model_version": "1.0.0"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """
    Classify text into categories
    
    Args:
        request: ClassificationRequest with text to classify
    
    Returns:
        Predicted label and confidence scores
    """
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize text
        tokens = [vocab.get(word, vocab.get('<UNK>', 1)) for word in request.text.split()]
        
        if not tokens:
            raise HTTPException(status_code=400, detail="Empty text")
        
        # Pad/truncate to fixed length
        max_len = 128
        if len(tokens) < max_len:
            tokens = tokens + [vocab.get('<PAD>', 0)] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        
        tokens = np.array([tokens], dtype=np.int32)
        
        # Forward pass
        logits = model.forward(tokens)
        
        # Get logits for last non-pad token
        last_logits = logits.data[0, -1, :]
        
        # Apply softmax
        exp_logits = np.exp(last_logits - np.max(last_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Get top predictions
        top_k = 3
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        all_scores = {}
        for idx in top_indices:
            label = idx_to_token.get(int(idx), f"class_{idx}")
            all_scores[label] = float(probs[idx])
        
        # Best prediction
        best_idx = top_indices[0]
        label = idx_to_token.get(int(best_idx), f"class_{best_idx}")
        confidence = float(probs[best_idx])
        
        return {
            "label": label,
            "confidence": confidence,
            "all_scores": all_scores
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/embed")
async def embed_text(text: str):
    """
    Get embedding vector for text
    
    Args:
        text: Input text to embed
    
    Returns:
        Embedding vector
    """
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize text
        tokens = [vocab.get(word, vocab.get('<UNK>', 1)) for word in text.split()]
        
        if not tokens:
            raise HTTPException(status_code=400, detail="Empty text")
        
        # Pad/truncate
        max_len = 128
        if len(tokens) < max_len:
            tokens = tokens + [vocab.get('<PAD>', 0)] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        
        tokens = np.array([tokens], dtype=np.int32)
        
        # Get embeddings (before output projection)
        batch_size, seq_len = tokens.shape
        positions = np.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        token_emb = model.token_embedding(tokens)
        pos_emb = model.position_embedding(positions)
        
        from src.core.tensor import Tensor
        x = Tensor(token_emb.data + pos_emb.data, requires_grad=False)
        
        # Pass through transformer blocks
        for block in model.blocks:
            x = block(x, None)
        
        x = model.norm(x)
        
        # Mean pooling over sequence
        embedding = np.mean(x.data[0], axis=0)
        
        return {
            "embedding": embedding.tolist(),
            "dimension": len(embedding)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate parameter count
    total_params = sum(np.prod(p.shape) for p in model.parameters())
    
    return {
        "name": "KuiperAI",
        "version": "1.0.0",
        "architecture": "Transformer",
        "parameters": int(total_params),
        "vocab_size": len(vocab) if vocab else 0,
        "d_model": model.d_model,
        "max_seq_len": model.max_seq_len,
        "num_layers": len(model.blocks),
        "capabilities": [
            "text_generation",
            "classification",
            "embedding"
        ]
    }


@app.post("/model/reload")
async def reload_model(new_config: Optional[ModelConfig] = None):
    """Reload model with new configuration"""
    global config
    
    if new_config:
        config = new_config
    
    try:
        load_model_from_checkpoint(
            config.checkpoint_path,
            config.vocab_path,
            config.model_params
        )
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Starting KuiperAI API server...")
    print("=" * 70)
    print("API documentation available at: http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
