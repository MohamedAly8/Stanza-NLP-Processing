import argparse
import stanza
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_model(lang):
    logging.info(f"Downloading model for {lang}...")
    stanza.download(lang, verbose=False)

def process_text(text, lang):
    logging.info(f"Processing text in {lang}...")
    nlp = stanza.Pipeline(lang, processors='tokenize,lemma,pos,depparse', verbose=False, use_gpu=False)
    doc = nlp(text)
    processed_text = []
    for i, sent in enumerate(doc.sentences):
        for word in sent.words:
            processed_text.append((word.text, word.lemma, word.pos, word.head, word.deprel))
    return processed_text

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description='Process text using Stanza.')
    parser.add_argument('--text', type=str, help='Text to process', required=True)
    parser.add_argument('--lang', type=str, help='Language of the text (e.g., en, zh)', required=True)
    parser.add_argument('--output', type=str, help='Output file to save processed results', default='output.txt')
    
    args = parser.parse_args()

    download_model(args.lang)
    processed_results = process_text(args.text, args.lang)

    with open(args.output, 'w') as f:
        for item in processed_results:
            f.write('\t'.join(str(i) for i in item) + '\n')

    logging.info(f"Processed results saved to {args.output}")

if __name__ == "__main__":
    main()
