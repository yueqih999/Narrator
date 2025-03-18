import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import logging


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


MODEL_NAME = 'all-MiniLM-L6-v2' 
OUTPUT_DIR = 'model/novel-search-model'
BATCH_SIZE = 8  
NUM_EPOCHS = 3 
WARMUP_STEPS = 100
MAX_SAMPLES = 5000  
NUM_DATASET_EXAMPLES = 3000


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def split_into_sentences(text):
    sentences = []
    for raw_sentence in text.split('.'):
        sentence = raw_sentence.strip()
        if len(sentence) > 10:  
            sentences.append(sentence)
    return sentences


def create_sentence_pairs(dataset, max_samples=10000, random_seed=42):
    random.seed(random_seed)
    examples = []
    samples_count = 0
    
    logging.info("Creating sentence pairs...")

    for item in tqdm(dataset):
        if samples_count >= max_samples:
            break
            
        text = item['text']
        sentences = split_into_sentences(text)
        
        if len(sentences) < 3:
            continue
        for i in range(len(sentences) - 1):
            if samples_count >= max_samples:
                break
            examples.append(InputExample(texts=[sentences[i], sentences[i+1]], label=0.8))
            samples_count += 1

            if len(sentences) > 3 and i < len(sentences) - 2:
                j = i + 2  
                examples.append(InputExample(texts=[sentences[i], sentences[j]],label=0.4))
                samples_count += 1
            
            if random.random() < 0.7 and len(sentences) > 5:
                rand_idx = random.randint(min(i+3, len(sentences)-1), len(sentences)-1) 
                if rand_idx < len(sentences) and rand_idx > i+2:
                    examples.append(InputExample(texts=[sentences[i], sentences[rand_idx]],label=0.2))
                    samples_count += 1
            else:
                if len(dataset) > 1:
                    other_idx = random.randint(0, len(dataset)-1)
                    other_text = dataset[other_idx]['text']
                    other_sentences = split_into_sentences(other_text)
                    
                    if other_sentences:
                        other_sentence = random.choice(other_sentences)
                        examples.append(InputExample(texts=[sentences[i], other_sentence],label=0.1))
                        samples_count += 1
    
    logging.info(f"Creating {len(examples)} sentence pairs")
    return examples


logging.info(f"Loading Bookcorpus...")
dataset = load_dataset("bookcorpus", split="train")
limited_dataset = dataset.select(range(min(NUM_DATASET_EXAMPLES, len(dataset))))
logging.info(f"Using {len(limited_dataset)} sentences")


logging.info(f"Creating training dataset (mostly {MAX_SAMPLES} 个)...")
training_examples = create_sentence_pairs(dataset, max_samples=MAX_SAMPLES)

logging.info(f"Loading pre-trained model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=BATCH_SIZE)

eval_examples = training_examples[:min(500, len(training_examples)//10)]
train_examples = training_examples[min(500, len(training_examples)//10):]

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1=[ex.texts[0] for ex in eval_examples],
    sentences2=[ex.texts[1] for ex in eval_examples],
    scores=[ex.label for ex in eval_examples]
)

train_loss = losses.CosineSimilarityLoss(model)

logging.info(f"Training setting:")
logging.info(f"- model: {MODEL_NAME}")
logging.info(f"- batch size: {BATCH_SIZE}")
logging.info(f"- epochs: {NUM_EPOCHS}")
logging.info(f"- training samples: {len(train_examples)}")
logging.info(f"- eval samples: {len(eval_examples)}")

logging.info("Start training...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=NUM_EPOCHS,
    evaluation_steps=200,
    warmup_steps=WARMUP_STEPS,
    output_path=OUTPUT_DIR,
    show_progress_bar=True,
    use_amp=False
)

logging.info(f"Training completed! Model saved at {OUTPUT_DIR}")

# testing
test_sentences = [
    "The protagonist faced his greatest fear.",
    "The main character confronted what terrified him most.",
    "The weather was exceptionally pleasant that day."
]

logging.info("\nModel testing:")
logging.info("Sentence 1: " + test_sentences[0])
logging.info("Sentence 2: " + test_sentences[1])
logging.info("Sentence 3: " + test_sentences[2])

embeddings = model.encode(test_sentences)
similarity_matrix = cosine_similarity(embeddings)

logging.info("\nSimilarity matrix:")
for i in range(len(test_sentences)):
    for j in range(len(test_sentences)):
        logging.info(f"Similarity of Sentence{i+1} and Sentence{j+1}: {similarity_matrix[i][j]:.4f}")

final_model_path = os.path.join(OUTPUT_DIR, 'final')
model.save(final_model_path)
logging.info(f"Final model saved at {final_model_path}")
