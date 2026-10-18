import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.ingestion import (
    _youtube_video_id,
    extract_transcript_from_youtube,
)


class YouTubeTranscriptTests(unittest.TestCase):
    def test_extracts_id_from_supported_youtube_urls(self):
        video_id = "Ty8gcCKuwNI"
        urls = (
            f"https://youtu.be/{video_id}",
            f"https://www.youtube.com/watch?v={video_id}&t=30",
            f"https://www.youtube.com/shorts/{video_id}",
            f"https://www.youtube.com/embed/{video_id}",
        )

        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(_youtube_video_id(url), video_id)

    def test_rejects_non_youtube_url(self):
        with self.assertRaisesRegex(ValueError, "valid YouTube video URL"):
            _youtube_video_id("https://example.com/watch?v=Ty8gcCKuwNI")

    def test_fetches_english_transcript_as_plain_text(self):
        transcript = SimpleNamespace(
            snippets=[
                SimpleNamespace(text="First caption."),
                SimpleNamespace(text="Second caption."),
            ]
        )
        with patch(
            "backend.ingestion.YouTubeTranscriptApi"
        ) as transcript_api:
            transcript_api.return_value.fetch.return_value = transcript

            result = extract_transcript_from_youtube(
                "https://youtu.be/Ty8gcCKuwNI"
            )

        self.assertEqual(result, "First caption.\nSecond caption.")
        transcript_api.return_value.fetch.assert_called_once_with(
            "Ty8gcCKuwNI", languages=["en", "hi"]
        )


if __name__ == "__main__":
    unittest.main()
