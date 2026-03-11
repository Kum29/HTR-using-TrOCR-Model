#this returns confidence of the words in the images.

@torch.inference_mode()
def process_batch_chunked(model, processor, all_images):
    all_texts = []
    all_confs = []
    inputs = processor(images=all_images, return_tensors="pt", padding=True).to(device)
    if device == "cuda":
        inputs.pixel_values = inputs.pixel_values.half()
    outputs = model.generate(
        inputs.pixel_values,
        max_length=64,
        output_scores=True,
        return_dict_in_generate=True,
        num_beams=1
    )
    texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    logits = torch.stack(outputs.scores, dim=1) 
    probs = torch.nn.functional.softmax(logits, dim=-1)
    gen_ids = outputs.sequences[:, 1:].unsqueeze(-1)
    batch_probs = torch.gather(probs, -1, gen_ids).squeeze(-1)
    confs = batch_probs.mean(dim=1).tolist()
    all_texts.extend(texts)
    all_confs.extend(confs)
    return all_texts, all_confs
