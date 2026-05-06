import os
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY     = os.environ.get("OPENAI_API_KEY", "")
MODEL       = "gpt-4o-mini"  # or any OpenAI model with JSON support
MAX_WORKERS = 20
INPUT_CSV   = "data/val_relevant_questions.csv"
OUTPUT_CSV  = "data/val_relevant_questions_spatial.csv"

client = OpenAI(api_key=API_KEY)

# ── Prompt ─────────────────────────────────────────────────────────────────────
SPATIAL_FILTER_PROMPT = """You are building a benchmark to evaluate spatial reasoning in multimodal LLMs.

Your task is to classify whether a question genuinely requires **spatial reasoning** to be answered — meaning the model must understand the position, layout, distance, or relationship between objects in an image.

## Answer YES only if the question requires:
- Relative position between objects (left/right, above/below, in front/behind, next to, between)
- Counting objects within a spatial context (e.g., "how many X are to the left of Y")
- Distance or proximity judgment (e.g., "what is closest to X")
- Spatial containment or grouping (e.g., "what is inside/outside X")
- Layout understanding (e.g., "what is in the corner/center/edge")
- Sequential or directional arrangement

## Answer NO if the question can be answered by:
- Simply detecting or recognizing an object ("is there a dog?", "what color is the car?")
- Reading text or identifying attributes (color, size, material) without spatial context
- Generic scene understanding without positional reasoning ("what is the weather?")
- Yes/no questions about object existence

## Examples:
Q: What is to the left of the table? A: The chair → YES
Q: Is the dog sitting? A: Yes → NO
Q: What is between the lamp and the window? A: A plant → YES
Q: What color is the sky? A: Blue → NO
Q: How many people are behind the counter? A: Two → YES
Q: Is there a car in the image? A: Yes → NO

Now classify the following:
Q: {question}
A: {answer}

Reply with only YES or NO."""


# ── Classification ─────────────────────────────────────────────────────────────
def is_spatial_question(question, answer):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a precise dataset curator. Reply only with YES or NO."},
            {"role": "user",   "content": SPATIAL_FILTER_PROMPT.format(question=question, answer=answer)}
        ],
    )
    return response.choices[0].message.content.strip().upper() == "YES"


def classify_row(args):
    idx, question, answer = args
    try:
        return idx, is_spatial_question(question, answer)
    except Exception as e:
        print(f"\nError at index {idx}: {e}")
        return idx, None


def classify_parallel(df, max_workers=MAX_WORKERS):
    args    = [(idx, row["question"], row["fullAnswer"]) for idx, row in df.iterrows()]
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_row, arg): arg[0] for arg in args}
        with tqdm(total=len(futures), desc="Classifying") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                pbar.update(1)

    return pd.Series(results)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df):,} rows loaded.\n")

    # First pass
    df["spatial"] = classify_parallel(df)

    # Retry failed rows (None)
    failed = df[df["spatial"].isna()]
    if not failed.empty:
        print(f"\nRetrying {len(failed):,} failed rows...")
        df.loc[failed.index, "spatial"] = classify_parallel(failed, max_workers=10)

    # Report
    spatial_df = df[df["spatial"] == True]
    print(f"\nResults:")
    print(f"  Total rows   : {len(df):,}")
    print(f"  Spatial (YES): {len(spatial_df):,}")
    print(f"  Filtered (NO): {len(df) - len(spatial_df):,}")
    print(f"  Failed       : {df['spatial'].isna().sum():,}")

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")