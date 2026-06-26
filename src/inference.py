import torch
import fire
from PIL import Image

from src.pre_processing.paligemma_preproccessing import PaligemmaPreprocessing
from src.gemma.pali_gemma_modeling import PaliGemmaModel, KVCache
from src.config.pali_gemma_config import PaliGemmaConfig

from utils import load_hf_model

def _sample_top_p(logits, top_p):
    """
        How this works:
        1. Sort the logits in descending order
        2. Calculate the cumulative probabilities
        3. Remove the tokens with cumulative probability greater than top_p
        4. Sample from the remaining tokens

        Why this works:
        - We maintain diversity by not being too greedy
        - We maintain quality by not being too random
    """
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cumulative_probs - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))

    next_token_probs = torch.multinomial(sorted_probs, num_samples=1)
    next_token = torch.gather(next_token_probs, dim=-1, index=sorted_indices)
    return next_token

def get_model_inputs(preprocessor: PaligemmaPreprocessing, device: str, prompt: str, image_path: str):
    image = Image.open(image_path)

    data = preprocessor([image], [prompt])
    input_ids = data["input_ids"].to(device)
    pixel_values = data["pixel_values"].to(device)
    attention_mask = data["attention_mask"].to(device)
    return input_ids, pixel_values, attention_mask

def test_inference(
    model: PaliGemmaModel,
    preprocessor: PaligemmaPreprocessing,
    device: str,
    prompt: str,
    image_path: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    input_ids, pixel_values, attention_mask = get_model_inputs(preprocessor, device, prompt, image_path)

    kv_cache = KVCache()

    stop_token = preprocessor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens):
        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, kv_cache=kv_cache)

        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

        if do_sample:
            next_token_logits=torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)

        if next_token.item() == stop_token:
            break

        # append next token to input_ids
        input_ids = next_token.unsqueeze(-1) # (1, 1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1)
    
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    return generated_tokens
        
def main(
    model_path: str,
    image_path: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, tokenizer = load_hf_model(model_path, device)
    model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    preprocessor = PaligemmaPreprocessing(tokenizer, num_image_tokens, image_size)

    print("Running Inference...")
    with torch.no_grad():
        generated_tokens = test_inference(
            model, 
            preprocessor, 
            device, 
            prompt,
            image_path, 
            max_tokens, 
            temperature, 
            top_p, 
            do_sample, 
        )

    print("Generated Tokens:", generated_tokens)
    print("Generated Text:", preprocessor.tokenizer.decode(generated_tokens))

if __name__ == "__main__":
    fire.Fire(main)
