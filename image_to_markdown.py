#!/usr/bin/env python3
"""
DeepSeek-OCR-2 Image to Markdown Converter
Optimized for AMD ROCm (RX 9070 XT)

Usage:
    python image_to_markdown.py <image_path> [output_dir]
    python image_to_markdown.py --benchmark <image_dir> --ground-truth <gt_file>

Example:
    python image_to_markdown.py document.png ./output
    python image_to_markdown.py --benchmark ocr_benchmark_images --ground-truth ground_truth.txt
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
import sys
import re
from difflib import SequenceMatcher

# Use ONLY dedicated GPUs (RX 9070 XT = gfx1201), exclude iGPU (gfx1036)
os.environ["HIP_VISIBLE_DEVICES"] = "0"
# ROCm memory optimization
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = 'deepseek-ai/DeepSeek-OCR-2'
DEFAULT_OUTPUT_DIR = '/tmp/ocr_output'

BENCHMARK_CONFIGS = [
    {'base_size': 512, 'image_size': 512},
    {'base_size': 640, 'image_size': 640},
    {'base_size': 768, 'image_size': 768},
]

def load_model():
    """Load model and tokenizer with ROCm optimizations"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True
    )
    
    print("Loading model (this may take a few minutes)...")
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        _attn_implementation='eager',  # ROCm compatible
        trust_remote_code=True, 
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"}  # Force use of first dedicated GPU only
    )
    
    model.eval()
    print(f"✓ Model loaded on: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    return tokenizer, model

def convert_image_to_markdown(tokenizer, model, image_path, output_dir, prompt=None, base_size=768, image_size=768):
    """Convert single image to markdown"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if prompt is None:
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    
    print(f"Processing: {image_path}")
    
    results = model.infer(
        tokenizer, 
        prompt=prompt, 
        image_file=image_path, 
        output_path=output_dir, 
        base_size=base_size, 
        image_size=image_size, 
        crop_mode=True, 
        save_results=True,
        test_compress=True
    )
    
    # Find and return the markdown file
    md_file = os.path.join(output_dir, 'result.mmd')
    if os.path.exists(md_file):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    return None

def calculate_accuracy(predicted_text, ground_truth):
    """Calculate text accuracy using sequence matching"""
    predicted_clean = re.sub(r'\s+', ' ', predicted_text.strip()).lower()
    ground_truth_clean = re.sub(r'\s+', ' ', ground_truth.strip()).lower()
    
    matcher = SequenceMatcher(None, predicted_clean, ground_truth_clean)
    return matcher.ratio() * 100

def run_benchmark(image_dir, ground_truth_path, output_base_dir):
    """Run benchmark with multiple parameter combinations"""
    import glob as glob_module
    import gc
    import subprocess
    
    images = glob_module.glob(os.path.join(image_dir, '*.png'))[:1]
    
    if not images:
        print(f"No images found in {image_dir}")
        return
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = f.read()
    
    original_img_size = os.path.getsize(images[0])
    image_name = os.path.basename(images[0])
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK: Testing {len(BENCHMARK_CONFIGS)} parameter combinations")
    print(f"Image: {image_name} ({original_img_size} bytes)")
    print(f"Ground truth: {ground_truth_path} ({len(ground_truth)} chars)")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, config in enumerate(BENCHMARK_CONFIGS):
        base_size = config['base_size']
        image_size = config['image_size']
        
        print(f"\n[{i+1}/{len(BENCHMARK_CONFIGS)}] Testing base_size={base_size}, image_size={image_size}")
        
        output_dir = os.path.join(output_base_dir, f"test_{base_size}x{image_size}")
        os.makedirs(output_dir, exist_ok=True)
        
        result_file = os.path.join(output_dir, 'result.mmd')
        
        cmd = [
            'python', 'image_to_markdown.py',
            images[0], output_dir,
            '--base-size', str(base_size),
            '--image-size', str(image_size)
        ]
        
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    markdown = f.read()
                
                md_size = len(markdown.encode('utf-8'))
                compression_ratio = (1 - md_size / original_img_size) * 100
                accuracy = calculate_accuracy(markdown, ground_truth)
                
                results.append({
                    'base_size': base_size,
                    'image_size': image_size,
                    'md_size': md_size,
                    'compression': compression_ratio,
                    'accuracy': accuracy
                })
                
                print(f"  ✓ Markdown: {md_size} bytes | Compression: {compression_ratio:.1f}% | Accuracy: {accuracy:.1f}%")
            else:
                print(f"  ✗ No output generated")
                if proc.stderr:
                    print(f"     Error: {proc.stderr[-200:]}")
                results.append({
                    'base_size': base_size,
                    'image_size': image_size,
                    'md_size': 0,
                    'compression': 0,
                    'accuracy': 0
                })
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout")
            results.append({
                'base_size': base_size,
                'image_size': image_size,
                'md_size': 0,
                'compression': 0,
                'accuracy': 0
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'base_size': base_size,
                'image_size': image_size,
                'md_size': 0,
                'compression': 0,
                'accuracy': 0
            })
    
    print(f"\n{'='*80}")
    print("RESULTS TABLE")
    print(f"{'='*80}")
    
    headers = ['Base Size', 'Image Size', 'Markdown Size', 'Compression', 'Accuracy']
    rows = []
    for r in results:
        rows.append([
            str(r['base_size']),
            str(r['image_size']),
            f"{r['md_size']} bytes",
            f"{r['compression']:.1f}%",
            f"{r['accuracy']:.1f}%"
        ])
    
    col_widths = []
    for i, h in enumerate(headers):
        max_width = len(h)
        for row in rows:
            if i < len(row):
                max_width = max(max_width, len(row[i]))
        col_widths.append(max_width)
    
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    
    print(header_line)
    print(separator)
    
    for row in rows:
        print(" | ".join(val.ljust(width) for val, width in zip(row, col_widths)))
    
    best_by_compression = max(results, key=lambda x: x['compression'])
    best_by_accuracy = max(results, key=lambda x: x['accuracy'])
    
    print(f"\nBest compression: base_size={best_by_compression['base_size']}, image_size={best_by_compression['image_size']} ({best_by_compression['compression']:.1f}%)")
    print(f"Best accuracy: base_size={best_by_accuracy['base_size']}, image_size={best_by_accuracy['image_size']} ({best_by_accuracy['accuracy']:.1f}%)")
    print(f"{'='*80}\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepSeek-OCR-2 Image to Markdown Converter')
    parser.add_argument('image_path', nargs='?', help='Path to input image')
    parser.add_argument('output_dir', nargs='?', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--benchmark', help='Run benchmark on image directory')
    parser.add_argument('--ground-truth', help='Ground truth file for accuracy measurement')
    parser.add_argument('--base-size', type=int, default=768, help='Base size for model (default: 768)')
    parser.add_argument('--image-size', type=int, default=768, help='Image size for model (default: 768)')
    
    args = parser.parse_args()
    
    if args.benchmark:
        if not args.ground_truth:
            print("Error: --ground-truth required for benchmark mode")
            sys.exit(1)
        
        run_benchmark(args.benchmark, args.ground_truth, './benchmark_output')
        return
    
    if not args.image_path:
        print(__doc__)
        print("Error: No image path provided")
        sys.exit(1)
    
    image_path = args.image_path
    output_dir = args.output_dir
    
    # Load model (cached on first run)
    tokenizer, model = load_model()
    
    # Convert
    try:
        markdown = convert_image_to_markdown(
            tokenizer, model, image_path, output_dir,
            base_size=args.base_size, image_size=args.image_size
        )
        print(f"\n✓ Conversion complete!")
        print(f"Output directory: {output_dir}")
        if markdown:
            print(f"\nExtracted text preview:")
            print("-" * 50)
            print(markdown[:500] if len(markdown) > 500 else markdown)
            print("-" * 50)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)
        
        run_benchmark(args.benchmark, args.ground_truth, './benchmark_output')
        return
    
    if not args.image_path:
        print(__doc__)
        print("Error: No image path provided")
        sys.exit(1)
    
    image_path = args.image_path
    output_dir = args.output_dir
    
    # Load model (cached on first run)
    tokenizer, model = load_model()
    
    # Convert
    try:
        markdown = convert_image_to_markdown(
            tokenizer, model, image_path, output_dir,
            base_size=args.base_size, image_size=args.image_size
        )
        print(f"\n✓ Conversion complete!")
        print(f"Output directory: {output_dir}")
        if markdown:
            print(f"\nExtracted text preview:")
            print("-" * 50)
            print(markdown[:500] if len(markdown) > 500 else markdown)
            print("-" * 50)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)
        
        tokenizer, model = load_model()
        run_benchmark(args.benchmark, args.ground_truth, './benchmark_output')
        return
    
    if not args.image_path:
        print(__doc__)
        print("Error: No image path provided")
        sys.exit(1)
    
    image_path = args.image_path
    output_dir = args.output_dir
    
    # Load model (cached on first run)
    tokenizer, model = load_model()
    
    # Convert
    try:
        markdown = convert_image_to_markdown(tokenizer, model, image_path, output_dir)
        print(f"\n✓ Conversion complete!")
        print(f"Output directory: {output_dir}")
        if markdown:
            print(f"\nExtracted text preview:")
            print("-" * 50)
            print(markdown[:500] if len(markdown) > 500 else markdown)
            print("-" * 50)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
