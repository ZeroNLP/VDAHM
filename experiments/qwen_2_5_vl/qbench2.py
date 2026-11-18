# qbench2.py of VDAHM
import torch
import argparse
from datasets import *
from generation import replace_generation
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import get_layers
import json
import os
from tqdm import tqdm
from PIL import Image

replace_generation()

def get_image_positions_by_tags(input_ids, start_token_id=151652, end_token_id=151653):
    """
    Find all ranges enclosed by start_token_id and end_token_id.
    
    Example:
        input: [..., 151652, 151655, 151655, 151653, ...]
        output: [(pos_151652, pos_151653 + 1)]  # or exclude head/tail
    
    Args:
        input_ids: torch.Tensor of shape (batch_size, seq_len)
        start_token_id: int, e.g., 151652 (<|vision_start|>)
        end_token_id: int, e.g., 151653 (<|vision_end|>)
    
    Returns:
        List of tuples [(start1, end1), (start2, end2), ...] for batch[0]
        where each tuple is a range that includes both start and end tokens.
    """
    batch_ranges = []
    
    for i in range(input_ids.shape[0]):
        ids = input_ids[i].tolist()  
        ranges = []
        stack = []  
        for idx, token_id in enumerate(ids):
            if token_id == start_token_id:
                stack.append(idx)
            elif token_id == end_token_id:
                if stack:
                    start_pos = stack.pop() 
                    ranges.append((start_pos + 1, idx))  
                else:
                    pass  
        batch_ranges.append(ranges)
    
    # Assume batch size 1; return first item
    return batch_ranges[0]


def get_img_names(img_path, prefix):
    """
    Get image paths from the original path string
    
    Args:
        img_path: original path string containing image names
        prefix: base directory for original images
    
    Returns:
        Tuple of (img1_path, img2_path)
    """
    img_paths = img_path.split('\\')[1][:-4].split("_cat_")
    img1_name = os.path.join(prefix, img_paths[0])
    img2_name = os.path.join(prefix, img_paths[1])
    
    print("Original img1_name: {}, img2_name: {}".format(img1_name, img2_name))
    
    # Check if original images exist
    if not os.path.exists(img1_name):
        raise FileNotFoundError(f"Image not found: {img1_name}")
    if not os.path.exists(img2_name):
        raise FileNotFoundError(f"Image not found: {img2_name}")
    
    return img1_name, img2_name


def format_candidates(candidates):
    """
    Format candidates into a readable string format for the model.
    
    Args:
        candidates: List of candidate answers
    
    Returns:
        Formatted string with options
    """
    formatted_options = []
    for i, candidate in enumerate(candidates):
        option_letter = chr(65 + i)  # A, B, C, D, ...
        formatted_options.append(f"{option_letter}. {candidate}")
    
    return " ".join(formatted_options)


def evaluate_model_on_json(json_file_path, model_path, image_prefix, 
                          vdahm_aug_ratio=1.3, vdahm_layers=14, vdahm_layer1=True, 
                          vdahm_filter=True, output_json_path=None):
    """
    Evaluate a vision-language model on a JSON dataset using VDAHM method.
    
    Args:
        json_file_path: Path to the evaluation dataset
        model_path: Path to the pretrained model
        image_prefix: Base directory for images
        vdahm_aug_ratio: VDAHM augmentation ratio (default: 1.3)
        vdahm_layers: Number of layers for VDAHM (default: 14)
        vdahm_layer1: Whether to include layer 1 in VDAHM (default: True)
        vdahm_filter: Whether to apply VDAHM filtering (default: True)
        output_json_path: Path to save results (optional)
    
    Returns:
        Tuple of (accuracy, results_list)
    """
    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype="auto", 
        device_map="auto",
        attn_implementation="sdpa" 
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    correct_predictions = 0
    total_predictions = 0
    results = []  # Store all results

    for item in tqdm(data, desc="Evaluating"):
        try:
            # Get original image paths (no resizing)
            img1_path, img2_path = get_img_names(item["img_path"], image_prefix)
            
            # Format options
            formatted_candidates = format_candidates(item["candidates"])
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img1_path},
                        {"type": "image", "image": img2_path},
                        {"type": "text", "text": f"{item['question']}\n\nOptions:\n{formatted_candidates}\n\nOnly answer with the letter corresponding to the best choice like: A"}
                    ],
                }
            ]

            # Prepare inputs
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Get image ranges for VDAHM
            image_ranges = get_image_positions_by_tags(inputs['input_ids'], start_token_id=151652, end_token_id=151653)

            generate_kwargs = {
                "num_beams": 1,
                "max_new_tokens": 4,
            }

            # Set VDAHM parameters
            vdahm_layers_list = list(range(len(get_layers(model))))[-vdahm_layers:] if vdahm_layers > 0 else []
            vdahm_layers_list = [1] + vdahm_layers_list if vdahm_layer1 else vdahm_layers_list

            generate_kwargs.update(
                {
                    "vdahm_aug_ratio": vdahm_aug_ratio,
                    "vdahm_layers": vdahm_layers_list,
                    "vdahm_filter": vdahm_filter,
                    "vdahm_image_ranges": image_ranges,
                }
            )

            inputs = inputs.to("cuda")

            # Inference
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs, 
                    **generate_kwargs,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            generated_sequences = generated_ids.sequences
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_sequences)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Check if output contains correct answer
            correct_answer = item["correct_ans"]
            correct_choice = item["correct_choice"]  # The correct letter choice (A, B, C, D)
            
            # Clean up the output text
            output_text_clean = output_text.strip().upper()
            
            # Try to extract the answer from the output
            extracted_answer = None
            for char in output_text_clean:
                if char.isalpha() and ord(char.upper()) >= ord('A'):
                    extracted_answer = char.upper()
                    break
            
            # If no letter found, check if any candidate appears in the output
            if extracted_answer is None:
                output_text_lower = output_text.lower()
                for i, candidate in enumerate(item["candidates"]):
                    if candidate.lower() in output_text_lower:
                        extracted_answer = chr(65 + i)  # Convert index to letter (A, B, C, D)
                        break
            
            # Check if the extracted answer matches the correct choice
            is_correct = (extracted_answer == correct_choice)
            
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            # Create result dictionary
            result_item = {
                "question": item["question"],
                "candidates": item["candidates"],
                "correct_answer": correct_answer,
                "correct_choice": correct_choice,
                "model_output": output_text,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct
            }
            
            results.append(result_item)

            print(f"Question: {item['question']}")
            print(f"Options: {formatted_candidates}")
            print(f"Expected Answer: {correct_answer} ({correct_choice})")
            print(f"Model Output: {output_text}")
            print(f"Extracted Answer: {extracted_answer}")
            print(f"Correct: {is_correct}")
            print(f"Accuracy so far: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions*100:.2f}%)")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing item: {e}")
            # Record error information even if there's an exception
            error_result = {
                "question": item.get("question", ""),
                "candidates": item.get("candidates", []),
                "correct_answer": item.get("correct_ans", ""),
                "correct_choice": item.get("correct_choice", ""),
                "model_output": "ERROR",
                "extracted_answer": None,
                "is_correct": False,
                "error": str(e)
            }
            results.append(error_result)
            continue

    # Calculate final accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nFinal Results:")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Save results to JSON file
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {output_json_path}")

    return accuracy, results


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate vision-language model on a dataset using VDAHM method")
    
    parser.add_argument(
        "--json_file_path", 
        type=str, 
        required=True,
        help="Path to the evaluation dataset JSON file containing questions and answers"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the pretrained vision-language model directory"
    )
    parser.add_argument(
        "--image_prefix", 
        type=str, 
        required=True,
        help="Base directory path where image files are stored"
    )
    parser.add_argument(
        "--output_json_path", 
        type=str, 
        required=True,
        help="Path where evaluation results will be saved in JSON format"
    )
    parser.add_argument(
        "--vdahm_aug_ratio", 
        type=float, 
        default=1.3,
        help="VDAHM augmentation ratio (default: 1.3)"
    )
    parser.add_argument(
        "--vdahm_layers", 
        type=int, 
        default=10,
        help="Number of layers to apply VDAHM (default: 10)"
    )
    parser.add_argument(
        "--vdahm_layer1", 
        type=bool, 
        default=True,
        help="Whether to include layer 1 in VDAHM (default: True)"
    )
    parser.add_argument(
        "--vdahm_filter", 
        type=bool, 
        default=True,
        help="Whether to apply VDAHM filtering (default: True)"
    )
    parser.add_argument(
        "--num_beams", 
        type=int, 
        default=1,
        help="Number of beams for beam search during generation (default: 1 for greedy decoding)"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=4,
        help="Maximum number of new tokens to generate (default: 4)"
    )
    parser.add_argument(
        "--attn_implementation", 
        type=str, 
        default="sdpa",
        help="Attention implementation to use: 'sdpa' (scaled dot product attention), 'flash_attention_2', etc. (default: sdpa)"
    )

    return parser.parse_args()


def main():
    """
    Main function to run the evaluation using VDAHM method.
    """
    args = parse_args()
    
    # Evaluate model with VDAHM
    accuracy, results = evaluate_model_on_json(
        json_file_path=args.json_file_path,
        model_path=args.model_path,
        image_prefix=args.image_prefix,
        vdahm_aug_ratio=args.vdahm_aug_ratio,
        vdahm_layers=args.vdahm_layers,
        vdahm_layer1=args.vdahm_layer1,
        vdahm_filter=args.vdahm_filter,
        output_json_path=args.output_json_path
    )

    print(f"\nModel accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()