from transcription import YouTubeVideoTranscript
from chunking import TranscriptChunker
from indexing import ChunkIndexer
from query_handling import QueryHandler

def main(playlist_url):
    youtube_tanscript = YouTubeVideoTranscript(api_key='YOUR_YOUTUBE_API_KEY')
    transcripts = youtube_tanscript.process_playlist(playlist_url)
    
    chunker = TranscriptChunker(strategy='intelligent')
    all_chunks = []
    for video_id, transcript in transcripts.items():
        chunks = chunker.chunk_transcript(transcript)
        all_chunks.extend(chunks)

    indexer = ChunkIndexer()
    embeddings = indexer.create_embeddings(all_chunks)
    indexer.build_index(embeddings)

    handler = QueryHandler(indexer)
    response = handler.handle_query("What are the key concepts discussed?")
    print(response)

if __name__ == "__main__":
    playlist_url = 'https://www.youtube.com/playlist?list=PL3MmuxUbc_hK60wsCyvrEK2RjQsUi4Oa_'
    main(playlist_url)
