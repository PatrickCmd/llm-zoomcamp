class TranscriptChunker:
    def __init__(self, strategy='heuristic'):
        self.strategy = strategy

    def heuristic_chunk(self, transcript: str):
        # Example of a simple heuristic: split every N sentences
        chunks = transcript.split('\n\n')
        # Strip leading/trailing whitespace from each paragraph
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

    def chunk_transcript(self, transcript: str):
        if self.strategy == 'heuristic':
            return self.heuristic_chunk(transcript)


if __name__ == "__main__":
    chunker = TranscriptChunker()

    # Example usage
    text = """Okay, I think we can start. Thank you for joining. I'll start with the introduction, and maybe more people will join by the time I finish. Thank you very much for coming to this meeting. This is an experimental event, the first time I am doing something like that. Let's see how it goes, but the idea is to have something without slides, like have a conversation. The topic for today is the roles in a data science team, not necessarily in a data science team but in a data team. What kind of roles are there, what kind of responsibilities they have, what do they do?

    For those who are not part of our data talks club yet, who are not in Slack, you can join our Slack using this link: join.datatalks.com. For questions today, we will use Slido to make it more interactive. I don't know how it will work, so for the first session, we will experiment a bit and see if it makes sense to you. We'll stop sharing my screen because the idea for this session is to have something without slides. So basically just talk about something and then also release this recording as a podcast.

    The link to the slides, if you have any questions during the talk today, please feel free to use Slido to ask that. Let me start just by quickly taking a look at my notes. The topic today is, as I said, the roles in a data team. We want to understand what kind of people work in the data team, what kind of responsibilities they have, what they do, what they need to know, what they don't have to know. Maybe a few words about myself, I work as a lead data scientist, and this means that my views might be a bit biased towards the views of a data scientist. This is how a data scientist sees other people, so I might not necessarily be right in all aspects. So if you think I'm wrong somewhere, please tell me because again, maybe the views on how I see data engineers are simplified because I don't see all the complexities of the work they're doing as a data scientist.

    So in a data team, there are many different roles. The first role we have is a product manager. A product manager is responsible for the entire product, making sure that the team is building the right thing. Then we have data specialists like data analysts, data scientists, data engineers, machine learning engineers. We also have something new called envelopes engineers, and it's often difficult to understand who is who and who needs to do what. This is why we have this conversation today to answer these questions."""

    # Chunk the text by paragraphs
    paragraphs = chunker.chunk_transcript(text)

    # Display the paragraphs
    for i, para in enumerate(paragraphs):
        print(f"Paragraph {i+1}:\n{para}\n")
