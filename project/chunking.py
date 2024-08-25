import spacy


class TranscriptChunker:
    def __init__(self, strategy="heuristic", chunk_size=512):
        self.strategy = strategy
        self.chunk_size = chunk_size
        if self.strategy == "intelligent":
            self.nlp = spacy.load("en_core_web_sm")

    def heuristic_chunk(self, transcript: str):
        # Example of a simple heuristic: split every N sentences
        chunks = transcript.split("\n\n")
        # Strip leading/trailing whitespace from each paragraph
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

    def semantic_chunking_spacy_multisentence(self, transcript_text):
        """
        Chunk the transcript_text text into smaller segments based on semantic boundaries using spaCy.

        :param transcript_text: The full transcript_text text to be chunked.
        :return: List of text chunks, each containing multiple sentences.
        """
        # Parse the text with spaCy
        doc = self.nlp(transcript_text)

        # Initialize chunk variables
        chunks = []
        chunk = ""

        for sentence in doc.sents:
            if len(chunk.split()) + len(sentence.text.split()) > self.chunk_size:
                if len(chunk) != 0:
                    chunks.append(chunk)
                chunk = sentence.text
            else:
                chunk += " " + sentence.text

        # Add the last chunk if it's not empty
        if chunk:
            chunks.append(chunk)

        return chunks

    def chunk_transcript(self, transcript: str):
        if self.strategy == "heuristic":
            return self.heuristic_chunk(transcript)
        elif self.strategy == "intelligent":
            return self.semantic_chunking_spacy_multisentence(transcript)


if __name__ == "__main__":
    chunker = TranscriptChunker(strategy="intelligent")

    # Example usage
    text = ""
    with open("transcript.txt", "r") as tf:
        text = tf.read().strip()

    # Chunk the text by paragraphs
    paragraphs = chunker.chunk_transcript(text)

    # Display the paragraphs
    for i, para in enumerate(paragraphs):
        print(f"Paragraph {i+1}:\n{para}\n")
