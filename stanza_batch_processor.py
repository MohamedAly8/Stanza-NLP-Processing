import argparse
import stanza
import logging
import os

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

    parser = argparse.ArgumentParser(description='Batch process text files using Stanza.')
    parser.add_argument('--input_dir', type=str, help='Directory containing text files to process', required=True)
    parser.add_argument('--lang', type=str, help='Language of the texts (e.g., en, zh)', required=True)
    parser.add_argument('--output_dir', type=str, help='Directory to save processed results', default='output')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    download_model(args.lang)

    for filename in os.listdir(args.input_dir):
        filepath = os.path.join(args.input_dir, filename)
        with open(filepath, 'r') as f:
            text = f.read()
        
        processed_results = process_text(text, args.lang)

        output_filepath = os.path.join(args.output_dir, f"processed_{filename}")
        with open(output_filepath, 'w') as f:
            for item in processed_results:
                f.write('\t'.join(str(i) for i in item) + '\n')

        logging.info(f"Processed results for {filename} saved to {output_filepath}")

if __name__ == "__main__":
    main()
