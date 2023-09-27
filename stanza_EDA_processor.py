import argparse
import stanza
import logging
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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


class Watcher:

    def __init__(self, input_dir, lang, output_dir):
        self.INPUT_DIR = input_dir
        self.LANG = lang
        self.OUTPUT_DIR = output_dir

    def run(self):
        event_handler = Handler(self.INPUT_DIR, self.LANG, self.OUTPUT_DIR)
        observer = Observer()
        observer.schedule(event_handler, self.INPUT_DIR, recursive=True)
        observer.start()
        try:
            while True:
                # Keep the script running
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

class Handler(FileSystemEventHandler):

    def __init__(self, input_dir, lang, output_dir):
        self.input_dir = input_dir
        self.lang = lang
        self.output_dir = output_dir

    def on_created(self, event):
        """This method is called whenever a new file is created in the input directory"""
        if event.is_directory:
            return None  # Ignore directories
        elif event.src_path.endswith('.txt'):  # Assuming you're only interested in .txt files
            logging.info(f"Detected new file: {event.src_path}")
            with open(event.src_path, 'r') as f:
                text = f.read()
            
            processed_results = process_text(text, self.lang)
            output_filepath = os.path.join(self.output_dir, f"processed_{os.path.basename(event.src_path)}")
            
            with open(output_filepath, 'w') as f:
                for item in processed_results:
                    f.write('\t'.join(str(i) for i in item) + '\n')
            
            logging.info(f"Processed results for {event.src_path} saved to {output_filepath}")

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description='Continuously watch and process text files using Stanza.')
    parser.add_argument('--input_dir', type=str, help='Directory containing text files to process', required=True)
    parser.add_argument('--lang', type=str, help='Language of the texts (e.g., en, zh)', required=True)
    parser.add_argument('--output_dir', type=str, help='Directory to save processed results', default='output')
    args = parser.parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    download_model(args.lang)
    watcher = Watcher(args.input_dir, args.lang, args.output_dir)
    watcher.run()
