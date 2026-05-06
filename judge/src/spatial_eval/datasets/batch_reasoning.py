"""
Batch Reasoning Augmentation with Claude Haiku 4.5
Processes spatial questions via Anthropic Message Batches API.

Usage:
  python batch_reasoning.py submit [--all] [--n N]   Submit batch(es) from train set. Default: 1.
                                                      --all: all remaining. --n N: exactly N batches.
  python batch_reasoning.py submit-val               Submit validation set as a single batch group.
  python batch_reasoning.py status                   Check status of all submitted batches.
  python batch_reasoning.py wait [--interval S]      Poll until all in-flight batches finish (live progress).
  python batch_reasoning.py results                  Download results from completed batches (train + val).
"""

import anthropic
import ast
import csv
import json
import os
import sys
import time
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    timeout=300.0,
)

BATCH_SIZE = 10_000
MODEL = "claude-haiku-4-5-20251001"
STATE_FILE = "batch_state.json"
RESULTS_DIR = "batch_results"

SYSTEM_PROMPT = """You are given a structured description of an image (a scene graph with objects, attributes, positions, and spatial relationships). Use this information to understand the scene, but respond AS IF you were directly looking at the image.

IMPORTANT RULES FOR YOUR RESPONSE:
- Never mention the scene graph, object IDs, coordinates, pixel positions, or any technical data.
- Never say things like "according to the graph", "object X is at position (a, b)", or reference any ID numbers.
- Write your reasoning as natural visual observation, as if you are describing what you see in a photograph. Use language like "I can see...", "Looking at the image...", "The cheese appears to be...", "On the left side of the image...".
- Describe spatial relationships naturally (e.g., "the cheese is sitting to the left of the main dish" instead of "cheese at x=1 is left of food at x=275").
- Your response will be paired with the actual image later, so it must read as authentic visual reasoning."""


def format_scene_graph(graph_row):
    objects = graph_row["objects"]
    if isinstance(objects, str):
        objects = ast.literal_eval(objects)
    width = graph_row["width"]
    height = graph_row["height"]
    location = graph_row.get("location", "") or ""

    lines = [f"Scene: {width}x{height} pixels" + (f", {location}" if location else "")]
    lines.append("Objects:")

    for obj_id, obj in objects.items():
        attrs = ", ".join(obj.get("attributes", []))
        attr_str = f" [{attrs}]" if attrs else ""
        pos = f"  at ({obj['x']}, {obj['y']}) size {obj['w']}x{obj['h']}"
        relations = obj.get("relations", [])

        rel_strs = []
        for rel in relations:
            target_id = rel["object"]
            target_name = objects.get(target_id, {}).get("name", target_id)
            rel_strs.append(f"{rel['name']} {target_name}({target_id})")

        rel_str = f"  relations: {'; '.join(rel_strs)}" if rel_strs else ""

        lines.append(f"  - {obj['name']}({obj_id}){attr_str}{pos}")
        if rel_str:
            lines.append(f"    {rel_str}")

    return "\n".join(lines)


def build_user_prompt(question, scene_graph_text):
    return f"""Scene information (for your internal use only, do NOT reference directly):
{scene_graph_text}

Question: {question}

Think step-by-step about what you observe in the image, then provide your final answer in the format:
ANSWER: <your answer>"""


def load_all_data(questions_path="data/relevant_questions_spatial.csv"):
    """Load spatial questions from a CSV and map each to its correct graph.

    Train: graphs from graphs_subset.csv, matched via object IDs in annotations.
    Val:   graphs from val_sceneGraphs.json, matched directly by imageId.
    """
    is_val = "val_" in os.path.basename(questions_path)

    if is_val:
        with open("data/val_sceneGraphs.json", "r", encoding="utf-8") as f:
            graphs = json.load(f)

        spatial_questions = []
        skipped = 0
        with open(questions_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if row["spatial"] != "True":
                    continue
                img_id = str(row["imageId"])
                if img_id in graphs:
                    spatial_questions.append((i, row, img_id))
                else:
                    skipped += 1

        print(f"  Mapped {len(spatial_questions)} val questions to graphs ({skipped} skipped)")
        return spatial_questions, graphs

    # Load all graphs indexed by row, and build object_id -> graph_row mapping
    graphs = {}
    obj_to_graph = {}
    with open("data/graphs_subset.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            graphs[i] = row
            objects = ast.literal_eval(row["objects"])
            for obj_id in objects:
                obj_to_graph[obj_id] = i

    # Load spatial questions and find their correct graph via object IDs
    spatial_questions = []
    skipped = 0
    with open(questions_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row["spatial"] != "True":
                continue
            # Find graph by matching object IDs from annotations
            ann = ast.literal_eval(row.get("annotations", "{}"))
            q_obj_ids = set()
            for k, v in ann.get("question", {}).items():
                q_obj_ids.add(v)
            for k, v in ann.get("fullAnswer", {}).items():
                q_obj_ids.add(v)

            graph_rows = set()
            for oid in q_obj_ids:
                if oid in obj_to_graph:
                    graph_rows.add(obj_to_graph[oid])

            if len(graph_rows) == 1:
                graph_idx = list(graph_rows)[0]
                spatial_questions.append((i, row, graph_idx))
            else:
                skipped += 1

    print(f"  Mapped {len(spatial_questions)} questions to graphs ({skipped} skipped)")
    return spatial_questions, graphs


def build_batch_requests(spatial_questions, graphs, start_idx, end_idx):
    """Build batch request objects for a slice of questions."""
    requests = []
    for q_idx, (row_idx, q, graph_idx) in enumerate(spatial_questions[start_idx:end_idx]):
        g = graphs.get(graph_idx)
        if g is None:
            continue

        question = q["question"]
        scene_text = format_scene_graph(g)
        user_prompt = build_user_prompt(question, scene_text)

        custom_id = f"row_{row_idx}"
        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": 4096,
                "system": SYSTEM_PROMPT,
                "thinking": {"type": "enabled", "budget_tokens": 3000},
                "messages": [{"role": "user", "content": user_prompt}],
            },
        })

    return requests


def load_state():
    default = {
        "batches": [],
        "next_start": 0,
        "total_spatial": 0,
        "val_batches": [],
        "val_next_start": 0,
        "val_total": 0,
    }
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        for k, v in default.items():
            state.setdefault(k, v)
        return state
    return default


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _submit_batches(state, state_key_batches, state_key_next, state_key_total,
                    questions_path, label, max_batches):
    """Shared core for submitting batches from train or val set."""
    print(f"Loading data from {questions_path}...")
    spatial_questions, graphs = load_all_data(questions_path)
    total = len(spatial_questions)
    state[state_key_total] = total
    print(f"Total {label} spatial questions: {total}")

    start = state[state_key_next]
    if start >= total:
        print(f"All {label} batches already submitted!")
        save_state(state)
        return 0

    iterations = 0
    while start < total:
        if max_batches is not None and iterations >= max_batches:
            break

        end = min(start + BATCH_SIZE, total)
        batch_num = len(state[state_key_batches]) + 1
        print(f"\nPreparing {label} batch {batch_num}: questions {start}-{end-1} ({end - start} items)...")

        requests = build_batch_requests(spatial_questions, graphs, start, end)
        print(f"  Built {len(requests)} requests")

        if len(requests) == 0:
            print("  No valid requests, skipping...")
            start = end
            state[state_key_next] = start
            save_state(state)
            continue

        print(f"  Submitting batch to API...")
        for attempt in range(3):
            try:
                batch = client.messages.batches.create(requests=requests)
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {str(e)[:100]}")
                if attempt < 2:
                    time.sleep(10)
                else:
                    raise

        batch_info = {
            "batch_id": batch.id,
            "batch_num": batch_num,
            "start_idx": start,
            "end_idx": end,
            "request_count": len(requests),
            "status": batch.processing_status,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "set": label,
        }
        state[state_key_batches].append(batch_info)
        state[state_key_next] = end
        save_state(state)

        print(f"  Batch submitted: {batch.id}")
        print(f"  Status: {batch.processing_status}")

        start = end
        iterations += 1
        time.sleep(2)

    remaining = max(0, total - state[state_key_next])
    print(f"\nDone. Submitted {iterations} {label} batch(es). {remaining} questions remaining.")
    return iterations


def cmd_submit(submit_all=False, n_batches=None):
    state = load_state()
    if submit_all:
        limit = None
    elif n_batches is not None:
        limit = n_batches
    else:
        limit = 1
    _submit_batches(
        state,
        state_key_batches="batches",
        state_key_next="next_start",
        state_key_total="total_spatial",
        questions_path="data/relevant_questions_spatial.csv",
        label="train",
        max_batches=limit,
    )


def cmd_submit_val():
    state = load_state()
    _submit_batches(
        state,
        state_key_batches="val_batches",
        state_key_next="val_next_start",
        state_key_total="val_total",
        questions_path="data/val_relevant_questions_spatial.csv",
        label="val",
        max_batches=None,
    )


def _all_batches(state):
    """Yield (group_label, batch_info) for every tracked batch."""
    for b in state.get("batches", []):
        yield "train", b
    for b in state.get("val_batches", []):
        yield "val", b


def cmd_status():
    state = load_state()
    total_batches = len(state["batches"]) + len(state["val_batches"])
    if total_batches == 0:
        print("No batches submitted yet.")
        return

    print(f"Train: {state['total_spatial']} questions, next_start={state['next_start']}, "
          f"batches={len(state['batches'])}")
    print(f"Val:   {state['val_total']} questions, next_start={state['val_next_start']}, "
          f"batches={len(state['val_batches'])}\n")

    for group, b in _all_batches(state):
        print(f"  [{group}] Batch {b['batch_num']} [{b['batch_id']}]")
        print(f"    Submitted: {b['submitted_at']} | Requests: {b['request_count']}")
        try:
            batch = client.messages.batches.retrieve(b["batch_id"])
        except anthropic.NotFoundError:
            b["status"] = "ended"
            print(f"    Status: ended (gone from server, local state={b.get('status', 'ended')})")
            continue
        b["status"] = batch.processing_status
        counts = batch.request_counts
        print(f"    Status: {batch.processing_status}")
        print(f"    Succeeded: {counts.succeeded} | Errored: {counts.errored} | "
              f"Expired: {counts.expired} | Canceled: {counts.canceled} | "
              f"Processing: {counts.processing}")

    save_state(state)


def cmd_wait(interval=15):
    """Poll in-flight batches (train + val) with live tqdm progress bars.
    Skips batches already marked 'ended' in local state (no reason to re-poll
    old completed batches, which the API may have since purged)."""
    state = load_state()
    tracked = [(g, b) for g, b in _all_batches(state) if b.get("status") != "ended"]
    if not tracked:
        print("No in-flight batches to wait for.")
        return

    bars = {}
    for group, b in tracked:
        key = b["batch_id"]
        desc = f"[{group}] batch {b['batch_num']}"
        bars[key] = tqdm(total=b["request_count"], desc=desc, unit="req",
                         position=len(bars), leave=True)

    try:
        while True:
            all_done = True
            for group, b in tracked:
                key = b["batch_id"]
                try:
                    batch = client.messages.batches.retrieve(key)
                except anthropic.NotFoundError:
                    b["status"] = "ended"
                    bars[key].set_postfix_str("[gone from server]")
                    bars[key].refresh()
                    continue
                b["status"] = batch.processing_status
                c = batch.request_counts
                finished = c.succeeded + c.errored + c.expired + c.canceled
                bar = bars[key]
                bar.n = finished
                bar.set_postfix_str(
                    f"ok={c.succeeded} err={c.errored} proc={c.processing} "
                    f"[{batch.processing_status}]"
                )
                bar.refresh()
                if batch.processing_status != "ended":
                    all_done = False

            if all_done:
                break
            time.sleep(interval)
    finally:
        for bar in bars.values():
            bar.close()
        save_state(state)

    print("\nAll batches ended.")


def _load_gt_map(questions_path):
    gt_map = {}
    with open(questions_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row["spatial"] == "True":
                gt_map[i] = {
                    "question": row["question"],
                    "answer": row["answer"].strip().lower(),
                    "fullAnswer": row["fullAnswer"],
                    "type": ast.literal_eval(row["types"])["detailed"],
                }
    return gt_map


def _download_group(batches, gt_map, file_prefix, label):
    """Download results for a list of batches, writing to file_prefix_N.jsonl."""
    correct = processed = errors = 0
    for b in batches:
        result_file = os.path.join(RESULTS_DIR, f"{file_prefix}_{b['batch_num']}.jsonl")

        # Already downloaded: count local stats and skip API retrieve entirely
        if os.path.exists(result_file):
            print(f"  [{label}] Batch {b['batch_num']}: already downloaded -> {result_file}")
            with open(result_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("correct"):
                        correct += 1
                    processed += 1
            continue

        try:
            batch = client.messages.batches.retrieve(b["batch_id"])
        except anthropic.NotFoundError:
            print(f"  [{label}] Batch {b['batch_num']} [{b['batch_id']}]: gone from server, cannot re-download")
            b["status"] = "ended"
            continue
        b["status"] = batch.processing_status

        if batch.processing_status != "ended":
            print(f"  [{label}] Batch {b['batch_num']} [{b['batch_id']}]: {batch.processing_status} (not ready)")
            continue

        print(f"  [{label}] Downloading batch {b['batch_num']} [{b['batch_id']}]...")
        results_iter = client.messages.batches.results(b["batch_id"])

        batch_correct = 0
        batch_total = 0
        batch_errors = 0

        pbar = tqdm(total=b["request_count"], desc=f"{label} batch {b['batch_num']}",
                    unit="req", leave=True)
        with open(result_file, "w", encoding="utf-8") as out:
            for result in results_iter:
                pbar.update(1)
                row_idx = int(result.custom_id.replace("row_", ""))
                gt = gt_map.get(row_idx, {})

                entry = {
                    "row_idx": row_idx,
                    "custom_id": result.custom_id,
                    "question": gt.get("question", ""),
                    "ground_truth": gt.get("answer", ""),
                    "type": gt.get("type", ""),
                }

                if result.result.type == "succeeded":
                    msg = result.result.message
                    thinking = ""
                    response_text = ""
                    for block in msg.content:
                        if block.type == "thinking":
                            thinking = block.thinking
                        elif block.type == "text":
                            response_text = block.text

                    # Extract answer
                    model_answer = ""
                    for line in response_text.split("\n"):
                        if line.strip().upper().startswith("ANSWER:"):
                            model_answer = line.split(":", 1)[1].strip().lower()
                            break
                    if not model_answer:
                        model_answer = response_text.strip().split("\n")[-1].strip().lower()

                    correct = model_answer == gt.get("answer", "") or gt.get("answer", "") in model_answer

                    entry.update({
                        "model_answer": model_answer,
                        "correct": correct,
                        "thinking": thinking,
                        "response": response_text,
                        "input_tokens": msg.usage.input_tokens,
                        "output_tokens": msg.usage.output_tokens,
                    })

                    if correct:
                        batch_correct += 1
                    batch_total += 1
                else:
                    entry.update({
                        "model_answer": "ERROR",
                        "correct": False,
                        "error": str(result.result.type),
                        "thinking": "",
                        "response": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                    })
                    batch_errors += 1
                    batch_total += 1

                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        pbar.close()

        correct += batch_correct
        processed += batch_total
        errors += batch_errors

        print(f"    Saved: {result_file}")
        print(f"    Accuracy: {batch_correct}/{batch_total} ({100*batch_correct/max(batch_total,1):.1f}%)")
        print(f"    Errors: {batch_errors}")

    return correct, processed, errors


def cmd_results():
    state = load_state()
    if not state["batches"] and not state["val_batches"]:
        print("No batches submitted yet.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_correct = total_processed = total_errors = 0

    if state["batches"]:
        gt_train = _load_gt_map("data/relevant_questions_spatial.csv")
        c, p, e = _download_group(state["batches"], gt_train, "batch", "train")
        total_correct += c
        total_processed += p
        total_errors += e

    if state["val_batches"]:
        gt_val = _load_gt_map("data/val_relevant_questions_spatial.csv")
        c, p, e = _download_group(state["val_batches"], gt_val, "val_batch", "val")
        print(f"\n[val] {c}/{p} correct ({100*c/max(p,1):.1f}%), errors={e}")

    save_state(state)

    if total_processed > 0:
        print(f"\n{'=' * 60}")
        print(f"TRAIN OVERALL: {total_correct}/{total_processed} ({100*total_correct/total_processed:.1f}%)")
        print(f"Errors: {total_errors}")
        print(f"{'=' * 60}")


def _parse_flag_value(argv, flag, cast=int):
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return cast(argv[idx + 1])
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    if cmd == "submit":
        submit_all = "--all" in sys.argv
        n_batches = _parse_flag_value(sys.argv, "--n", int)
        cmd_submit(submit_all=submit_all, n_batches=n_batches)
    elif cmd == "submit-val":
        cmd_submit_val()
    elif cmd == "status":
        cmd_status()
    elif cmd == "wait":
        interval = _parse_flag_value(sys.argv, "--interval", int) or 15
        cmd_wait(interval=interval)
    elif cmd == "results":
        cmd_results()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()