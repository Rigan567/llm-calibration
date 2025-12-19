import pandas as pd
from datasets import load_dataset, Dataset
import random

# ============================================================
# Helper: Safe string conversion
# ============================================================

def to_str(x):
    """Convert any object (list, dict, None) safely into a string."""
    if x is None:
        return ""
    if isinstance(x, (list, dict)):
        return str(x)
    return str(x)

# ============================================================
# 1. LOAD DATASETS
# ============================================================

# --- Astro-QA (XLSX file) ---
astro_df_judgement = pd.read_excel("astroqa/judgment_EN.xlsx")
astro_df_subjective = pd.read_excel("astroqa/subjective question_EN.xlsx")

astro_df_subjective = astro_df_subjective.astype(str)
astro_df_judgement = astro_df_judgement.astype(str)

astro_judgement = Dataset.from_pandas(astro_df_judgement)
astro_subjective = Dataset.from_pandas(astro_df_subjective)

# --- GlobalMedQA (HuggingFace) — English only ---
medqa = load_dataset("mariocedo/GlobalMedQA", "full", split="train")
medqa_en = medqa.filter(lambda x: x.get("language") == "EN")
medqa_en_single = medqa_en.filter(lambda x: x.get("multiple_answers") is False)

# --- TORQUE (GitHub local JSON files) ---
torque = load_dataset(
    "json",
    data_files="torque/*.json",
    split="train"
)

# --- HotpotQA (HuggingFace) ---
hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

# ============================================================
# 2. SAMPLE 250 ITEMS FROM EACH (pre-mapping)
# ============================================================

def sample_250(ds):
    ds = ds.shuffle(seed=42)
    return ds.select(range(min(250, len(ds))))

astro_200_s  = sample_250(astro_subjective)
astro_200_j  = sample_250(astro_judgement)
medqa_200    = sample_250(medqa_en_single)
torque_200   = sample_250(torque)
hotpot_200   = sample_250(hotpot)

# ============================================================
# 3. NORMALIZE & MAP TO UNIFIED FORMAT
# ============================================================

def map_astro_j(example):
    question = to_str(example.get("Question"))
    answer = to_str(example.get("Answer"))
    prompt = to_str(example.get("Prompt"))

    full_question = f"{question} {prompt}"

    return {
        "question": full_question,
        "answer": answer,
        "source": "Astro-QA_Judgement"
    }

def map_astro_s(example):
    question = to_str(example.get("Question"))
    answer = to_str(example.get("Answer"))
    prompt = to_str(example.get("Prompt"))

    full_question = f"{question} {prompt}"

    return {
        "question": full_question,
        "answer": answer,
        "source": "Astro-QA_Subjective"
    }

def map_medqa(example):
    options = example.get("options", {})
    valid_options = {k: v for k, v in options.items() if v is not None}
    options_text = "; ".join([f"{k}: {v}" for k, v in valid_options.items()])
    question_with_options = f"{to_str(example.get('question'))} [Options: {options_text}]"

    correct_keys = example.get("answer", [])
    answer_texts = [f"{key}: {valid_options[key]}" if key in valid_options else key for key in correct_keys]

    return {
        "question": question_with_options,
        "answer": "; ".join(answer_texts),
        "source": "GlobalMedQA_EN"
    }

def map_torque(example, max_per_context=1):
    """
    Map TORQUE examples to QA pairs, including context in the question.
    """
    mapped_qas = []

    for passage in example.get("passages", []):
        context = passage.get("passage", "").strip()
        valid_qas = []

        for qa in passage.get("question_answer_pairs", []):
            question_text = qa.get("question", "").strip()
            answer_spans = qa.get("answer", {}).get("spans", [])
            answer_text = "; ".join([span.strip() for span in answer_spans])

            if not question_text or not answer_text:
                continue

            full_question = f"{context} [Question: {question_text}]"
            valid_qas.append({
                "question": full_question,
                "answer": answer_text,
                "source": "TORQUE"
            })

        if valid_qas:
            sampled_qas = random.sample(valid_qas, min(len(valid_qas), max_per_context))
            mapped_qas.extend(sampled_qas)

    return mapped_qas

def map_hotpot(example):
    return {
        "question": to_str(example.get("question")),
        "answer": to_str(example.get("answer")),
        "source": "HotpotQA"
    }

# ============================================================
# Map datasets
# ============================================================

astro_mapped_j  = astro_200_j.map(map_astro_j)
astro_mapped_s  = astro_200_s.map(map_astro_s)
medqa_mapped    = medqa_200.map(map_medqa)

# TORQUE: map each example and flatten the list
torque_mapped_list = []
for ex in torque_200:
    torque_mapped_list.extend(map_torque(ex))

# Ensure max 250 QAs for TORQUE after mapping
torque_mapped_list = random.sample(torque_mapped_list, min(250, len(torque_mapped_list)))
torque_mapped = Dataset.from_list(torque_mapped_list)

hotpot_mapped = hotpot_200.map(map_hotpot)

# ============================================================
# 4. COMBINE ALL INTO ONE DATASET
# ============================================================

questions = (
    astro_mapped_j["question"][:] +
    astro_mapped_s["question"][:] +
    medqa_mapped["question"][:] +
    torque_mapped["question"][:] +
    hotpot_mapped["question"][:]
)

answers = (
    astro_mapped_j["answer"][:] +
    astro_mapped_s["answer"][:] +
    medqa_mapped["answer"][:] +
    torque_mapped["answer"][:] +
    hotpot_mapped["answer"][:]
)

sources = (
    astro_mapped_j["source"][:] +
    astro_mapped_s["source"][:] +
    medqa_mapped["source"][:] +
    torque_mapped["source"][:] +
    hotpot_mapped["source"][:]
)

ids = list(range(len(questions)))  # or start from 1 if you prefer

combined = Dataset.from_dict({
    "id": ids,              # <-- FIRST COLUMN
    "question": questions,
    "answer": answers,
    "source": sources,
})

print(combined)
print("Total examples:", len(combined))

print(combined)
print("Total examples:", len(combined))

# ============================================================
# 5. SAVE FINAL MERGED DATASET
# ============================================================

combined.to_json("combined_qa_dataset_800.jsonl")
