import json
import zipfile

def fuse():
    print("Fusing Gemini and T5 predictions...")
    
    # Load Gemini (High Semantic Understanding)
    try:
        with open("predictions_symbolic.json") as f:
            gemini_preds = {item['id']: item for item in json.load(f)}
    except FileNotFoundError:
        print("Error: predictions_symbolic.json not found. Did you run Cell 1 with Gemini?")
        return

    # Load T5 (High Syntactic Precision)
    try:
        with open("predictions_t5.json") as f:
            t5_preds = {item['id']: item for item in json.load(f)}
    except FileNotFoundError:
        print("Error: predictions_t5.json not found. Please run the T5 generation command.")
        return
    
    fused_submission = []
    changes = 0
    
    for uid, g_item in gemini_preds.items():
        t_item = t5_preds.get(uid)
        
        g_valid = g_item['validity']
        t_valid = t_item['validity'] if t_item else False
        
        # STRATEGY: LOGICAL OR (Union)
        # This gave us the highest accuracy (~96.86%).
        # To fix the TCE, we will just ensure BOTH inputs are Anonymized.
        
        final_valid = g_valid or t_valid
        
        if final_valid != g_valid:
            changes += 1
            
        fused_submission.append({
            "id": uid,
            "validity": final_valid
        })
        
    print(f"Fused {len(fused_submission)} predictions.")
    print(f"Total Valid: {sum(1 for x in fused_submission if x['validity'])}")
    print(f"Recovered {changes} valid syllogisms that Gemini missed (using T5).")

    with open("predictions.json", "w") as f:
        json.dump(fused_submission, f, indent=2)

    with zipfile.ZipFile("predictions.zip", "w") as zf:
        zf.write("predictions.json")
    print("Created predictions.zip")

if __name__ == "__main__":
    fuse()
