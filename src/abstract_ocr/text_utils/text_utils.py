from ..imports import (
    logger,
    Counter,
    get_spacy,
    get_kw_model,
    get_summarizer,
    get_transformers,
    get_generator,
)
from ..ocr_utils import extract_text_from_image
import os
from urllib.parse import quote

# Lazy-load spaCy model
def _load_spacy_model():
    spacy = get_spacy()
    return spacy.load("en_core_web_sm")



# ─────────────────────────────────────────────────────────────
# Content length extraction
# ─────────────────────────────────────────────────────────────

def get_content_length(text):
    for each in ['into a ']:
        if each in text:
            text = text.split(each)[1]
            break
    for each in [' word']:
        if each in text:
            text = text.split(each)[0]
            break
    numbers = []
    for each in text.split('-'):
        numbers.append('')
        for char in each:
            char = str(char)
            if char in list('1234567890'):
                numbers[-1] += char
    for i, number in enumerate(numbers):
        if number:
            numbers[i] = int(number) * 10
    return numbers


# ─────────────────────────────────────────────────────────────
# LED/BigBird summarization
# ─────────────────────────────────────────────────────────────

def generate_with_bigbird(text: str, task: str = "title", model_dir: str = "allenai/led-base-16384") -> str:
    """Generate content using LED (BigBird) model."""
    try:
        LEDTokenizer = get_transformers("LEDTokenizer")
        LEDForConditionalGeneration = get_transformers("LEDForConditionalGeneration")
        
        tokenizer = LEDTokenizer.from_pretrained(model_dir)
        model = LEDForConditionalGeneration.from_pretrained(model_dir)
        
        prompt = (
            f"Generate a concise, SEO-optimized {task} for the following content: {text[:1000]}"
            if task in ["title", "caption", "description"]
            else f"Summarize the following content into a 100-150 word SEO-optimized abstract: {text[:4000]}"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=200 if task in ["title", "caption"] else 300,
            num_beams=5,
            early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error in BigBird processing: {e}")
        return ""


def refine_with_gpt(full_text: str, task=None) -> str:
    """Refine content using generator."""
    prompt = generate_with_bigbird(full_text, task=task)
    lengths = get_content_length(full_text)
    max_length = 200
    min_length = 100
    
    if lengths:
        max_l = lengths[-1]
        if max_l:
            max_length = int(max_l)
        min_l = lengths[0]
        if min_l:
            min_length = int(min_l)
    
    generator = get_generator()
    out = generator(
        prompt,
        min_length=min_length,
        max_length=max_length,
        num_return_sequences=1
    )[0]["generated_text"]
    return out.strip()


# ─────────────────────────────────────────────────────────────
# Media URL generation
# ─────────────────────────────────────────────────────────────

EXT_TO_PREFIX = {
    '.jpg': 'images',
    '.jpeg': 'images',
    '.png': 'images',
    '.gif': 'images',
    '.mp4': 'videos',
    '.pdf': 'documents',
}


def generate_media_url(fs_path: str, domain=None, repository_dir=None) -> str | None:
    """Generate accessible media URL from filesystem path."""
    fs_path = os.path.abspath(fs_path)
    if not fs_path.startswith(repository_dir):
        return None
    rel_path = fs_path[len(repository_dir):]
    rel_path = quote(rel_path.replace(os.sep, '/'))
    ext = os.path.splitext(fs_path)[1].lower()
    prefix = EXT_TO_PREFIX.get(ext, 'repository')
    return f"{domain}/{prefix}/{rel_path}"


# ─────────────────────────────────────────────────────────────
# Keyword extraction
# ─────────────────────────────────────────────────────────────

def get_keybert(full_text,
                keyphrase_ngram_range=None,
                top_n=None,
                stop_words=None,
                use_mmr=None,
                diversity=None):
    """Extract keywords using KeyBERT model."""
    keyphrase_ngram_range = keyphrase_ngram_range or (1, 3)
    top_n = top_n or 10
    stop_words = stop_words or "english"
    use_mmr = use_mmr or True
    diversity = diversity or 0.5
    
    keybert = get_kw_model().extract_keywords(
        full_text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=stop_words,
        top_n=top_n,
        use_mmr=use_mmr,
        diversity=diversity
    )
    return keybert


def extract_keywords_nlp(text, top_n=10):
    """Extract keywords using spaCy NLP."""
    if not isinstance(text, str):
        logger.info(f"this is not a string: {text}")
    nlp = _load_spacy_model()
    doc = nlp(str(text))
    word_counts = Counter(
        token.text for token in doc
        if token.pos_ in ["NOUN", "PROPN"]
        and not token.is_stop
        and len(token.text) > 2
    )
    entity_counts = Counter(
        ent.text.lower() for ent in doc.ents
        if len(ent.text.split()) > 1
    )
    top_keywords = [word for word, _ in (word_counts + entity_counts).most_common(top_n)]
    return top_keywords


def calculate_keyword_density(text, keywords):
    """Calculate keyword density in text."""
    if text:
        words = text.lower().split()
        return {
            kw: (words.count(kw.lower()) / len(words)) * 100
            for kw in keywords
            if kw and len(words) > 0
        }
    return {}


def refine_keywords(
    full_text=None,
    keywords=None,
    keyphrase_ngram_range=None,
    top_n=None,
    stop_words=None,
    use_mmr=None,
    diversity=None,
    info_data=None
):
    """Extract and refine keywords from text."""
    info_data = info_data or {}
    info_data['keywords'] = keywords or extract_keywords_nlp(full_text, top_n=top_n)
    
    keybert = get_keybert(
        full_text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        top_n=top_n,
        stop_words=stop_words,
        use_mmr=use_mmr,
        diversity=diversity
    )
    
    info_data['combined_keywords'] = list(
        {kw for kw, _ in keybert} | set(info_data['keywords'])
    )[:10]
    
    info_data['keyword_density'] = calculate_keyword_density(
        full_text,
        info_data['combined_keywords']
    )
    return info_data


# ─────────────────────────────────────────────────────────────
# Summarization
# ─────────────────────────────────────────────────────────────

def chunk_summaries(chunks,
                    max_length=None,
                    min_length=None,
                    truncation=False):
    """Summarize text chunks."""
    max_length = max_length or 160
    min_length = min_length or 40
    summaries = []
    
    summarizer = get_summarizer()
    for idx, chunk in enumerate(chunks):
        out = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            truncation=truncation
        )
        summaries.append(out[0]["summary_text"])
    
    return summaries


def split_to_chunk(full_text, max_words=None):
    """Split text into chunks by sentence."""
    max_words = max_words or 300
    sentences = full_text.split(". ")
    chunks, buf = [], ""
    
    for sent in sentences:
        if len((buf + sent).split()) <= max_words:
            buf += sent + ". "
        else:
            chunks.append(buf.strip())
            buf = sent + ". "
    
    if buf:
        chunks.append(buf.strip())
    
    return chunks


def get_summary(full_text,
                keywords=None,
                max_words=None,
                max_length=None,
                min_length=None,
                truncation=False):
    """Generate summary of full text."""
    summary = None
    
    if full_text and get_summarizer():
        chunks = split_to_chunk(full_text, max_words=max_words)
        summaries = chunk_summaries(
            chunks,
            max_length=max_length,
            min_length=min_length,
            truncation=truncation
        )
        summary = " ".join(summaries).strip()
    
    return summary
