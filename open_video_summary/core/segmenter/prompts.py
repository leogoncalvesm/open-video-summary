from dataclasses import dataclass, field


@dataclass
class VideoSegmenterPrompts:
    generate_subtopics: str = field(
        default="""
            @DOCUMENT
            '''{full_video_transcript}'''

            #####

            @OUTPUT

            The output should be a JSON object with the following example format
            {{
                "0": "Example topic",
                "1": "Another topic example",
                "2": ...
            }}

            #####

            Generate up to {max_subtopics} subtopics from @DOCUMENT. The response should contain only the subtopics using the @OUTPUT format specified.
        """
    )

    classify_subtopic: str = field(
        default="""
            @DOCUMENT
            '''
            {content}
            '''

            #####

            @SUBTOPICS

            {topics}

            #####

            The output should be a JSON object with the following example format:

            @OUTPUT

            The output should be the JSON object with the corresponding topic and it's text description. Example:

            {{"4": "Example subtopic that best fits the document"}}

            #####

            Select the @SUBTOPICS item that best classifies the @DOCUMENT according to its textual content. The answer should contain the subtopic using the @OUTPUT format specified, with no further description.
        """
    )

    fix_transcription: str = field(
        default="""
            @DOCUMENT
            '''
            {content}
            '''

            #####
            @EXAMPLE

            Input: "Vamos para asno tícias de hoje. Mais, antes, preciso contar para vocês do presidente que recebi. Alojado action figu des My Figurs me mandou esse bosto do soro do One Piece"
            Output: "Vamos para as notícias de hoje. Mas, antes, preciso contar para vocês do presente que recebi. A loja de action figures My Figures me mandou esse busto do Zoro do One Piece."

            #####

            The text in @DOCUMENT was automatically extracted from an audio file and might contain errors, such as typos or homophones words instead of the actual ones.
            Rewrite the @DOCUMENT string fixing those errors. The output should only contain the fixed text, as presented in the @EXAMPLE provided.
        """
    )

    def __post_init__(self) -> None:
        if (
            "{full_video_transcript}" not in self.generate_subtopics
            or "{max_subtopics}" not in self.generate_subtopics
        ):
            raise ValueError(
                "The prompt template for `generate_subtopics` must contain both a '{full_video_transcript}' and '{max_subtopics}' placeholders."
            )

        if (
            "{content}" not in self.classify_subtopic
            or "{topics}" not in self.classify_subtopic
        ):
            raise ValueError(
                "The prompt template for `classify_subtopic` must contain both a '{content}' and '{topics}' placeholders."
            )

