# Here, all images are going together to model_base. If a word has prob less than 0.9, model_large will be called.

final_text = [None] * len(lines)
if lines:
    start_time = time.time()
    pred_base, confs_base = process_batch_chunked(model_base, processor_base, lines)
    to_improve =[]
    for idx, confs_base in enumerate(confs_base): #confs and pred both
        if confs_base < 0.90:
            to_improve.append(idx)
        else:
            final_text[idx]= pred_base[idx]
    if to_improve:
        images_to_improve = [lines[idx] for idx in to_improve]
        pred_large_improve, confs_large_improve = process_batch_chunked(model_large, processor_large, images_to_improve)
    
    for idx, confs_large in enumerate(confs_large_improve):
        if final_text[idx] is None:
            final_text[idx]= pred_large_improve[idx]
    total_elapsed = time.time() - start_time

print(f"\nTotal time for {len(lines)} lines: {total_elapsed:.2f} seconds")
result = "\n".join(final_text) #result is a string
print(result)
