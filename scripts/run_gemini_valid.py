#!/usr/bin/env python
"""Run Gemini VLM annotation on the validation set."""

from __future__ import annotations

from pathlib import Path

from object_detection_training.tasks.vlm_annotation_task import VLMAnnotationTask

CLASSES = [
    "ball",
    "ball-in-basket",
    "number",
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "referee",
    "rim",
]

PROMPT = (
    "As a super recognizer, detect all of the basketball players, referees, "
    "the rim, the basketball, and the jersey numbers in this basketball game image. "
    "There are at most 3 referees and 10 players on the court, one rim and one ball. "
    "\n\n"
    "Use EXACTLY these labels for detections:\n"
    '- "ball" for the basketball when it is NOT going through the basket\n'
    '- "ball-in-basket" for the basketball when it IS going through the rim/net\n'
    '- "number" for visible jersey numbers on players\n'
    '- "player" for any standing/running/walking player NOT in a special action\n'
    '- "player-in-possession" for the player who currently has/is dribbling the ball\n'
    '- "player-jump-shot" for a player in the act of shooting a jump shot\n'
    '- "player-layup-dunk" for a player attempting a layup or dunk\n'
    '- "player-shot-block" for a player actively blocking a shot\n'
    '- "referee" for game officials\n'
    '- "rim" for the basketball hoop rim\n'
    "\n"
    "Each player should have exactly ONE label (the most specific one that applies). "
    "Return bounding boxes normalised to [0, 1] in (x, y, w, h) format where "
    "x,y is the top-left corner."
)

IMAGE_DIR = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3/valid"
)

OUTPUT_DIR = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/"
    "basketball-player-detection-3/valid_gemini_annotations"
)


def main() -> None:
    task = VLMAnnotationTask(
        name="vlm_annotation",
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        model_name="gemini-2.5-pro",
        classes=CLASSES,
        prompt_template=PROMPT,
    )
    result = task()
    print(f"Done: {result}")


if __name__ == "__main__":
    main()
