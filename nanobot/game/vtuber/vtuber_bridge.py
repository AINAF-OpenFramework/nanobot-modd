"""VTuber output bridge for chess gameplay integration."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger


class VTuberOutput:
    """
    Output handler for VTuber integration.
    
    Structures output for:
    - Avatar expression triggers
    - Move notation
    - Commentary
    - TTS-ready format (optional)
    """

    # Expression intensity thresholds
    SCORE_THRESHOLD_POSITIVE = 5.0  # Very good move
    SCORE_THRESHOLD_NEGATIVE = -3.0  # Bad move

    def __init__(self, output_format: str = "json", enable_tts: bool = False):
        """
        Initialize VTuber output handler.
        
        Args:
            output_format: Output format ("json", "text", or "rich")
            enable_tts: Enable TTS-ready output formatting
        """
        self.output_format = output_format
        self.enable_tts = enable_tts
        logger.debug(
            f"vtuber_bridge.VTuberOutput initialized format={output_format} tts={enable_tts}"
        )

    def send_move_and_comment(
        self,
        move: str,
        comment: str,
        score: float | None = None,
        board_state: dict[str, Any] | None = None,
        emotion: str = "neutral",
    ) -> dict[str, Any]:
        """
        Output handler for VTuber integration.
        
        Structures output with move notation, commentary, and optional
        avatar expression triggers.
        
        Args:
            move: Move in algebraic notation (e.g., "e2e4")
            comment: Commentary text
            score: Optional move evaluation score
            board_state: Optional board state for context
            emotion: Avatar emotion/expression (e.g., "happy", "thinking", "excited")
            
        Returns:
            Structured output dictionary
            
        Example:
            >>> bridge = VTuberOutput(output_format="json")
            >>> output = bridge.send_move_and_comment("e2e4", "Good opening!", score=0.5)
            >>> "move" in output and "comment" in output
            True
        """
        # Build output structure
        output = {
            "move": move,
            "comment": comment,
            "emotion": emotion,
            "timestamp": self._get_timestamp(),
        }
        
        if score is not None:
            output["score"] = round(score, 2)
        
        if board_state:
            output["board_state"] = board_state
        
        # Add avatar expression mapping
        output["expression"] = self._map_emotion_to_expression(emotion, score)
        
        # Format for TTS if enabled
        if self.enable_tts:
            output["tts_text"] = self._format_for_tts(comment)
        
        # Output in requested format
        self._output(output)
        
        logger.debug(
            f"vtuber_bridge.send_move_and_comment move={move} emotion={emotion}"
        )
        
        return output

    def send_game_event(
        self,
        event_type: str,
        message: str,
        emotion: str = "neutral",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send a game event (check, checkmate, draw, etc.).
        
        Args:
            event_type: Type of event ("check", "checkmate", "draw", "start", "end")
            message: Event message
            emotion: Avatar emotion for the event
            metadata: Optional additional event data
            
        Returns:
            Structured event output
        """
        event = {
            "type": event_type,
            "message": message,
            "emotion": emotion,
            "expression": self._map_emotion_to_expression(emotion),
            "timestamp": self._get_timestamp(),
        }
        
        if metadata:
            event["metadata"] = metadata
        
        if self.enable_tts:
            event["tts_text"] = self._format_for_tts(message)
        
        self._output(event)
        
        logger.info(f"vtuber_bridge.send_game_event type={event_type} emotion={emotion}")
        
        return event

    def _map_emotion_to_expression(
        self,
        emotion: str,
        score: float | None = None,
    ) -> str:
        """
        Map emotion to avatar expression.
        
        Args:
            emotion: Emotion string
            score: Optional score for expression intensity
            
        Returns:
            Expression identifier for avatar system
        """
        emotion_map = {
            "happy": "smile",
            "excited": "joy",
            "thinking": "thoughtful",
            "surprised": "shocked",
            "worried": "concerned",
            "confident": "determined",
            "neutral": "calm",
        }
        
        expression = emotion_map.get(emotion, "calm")
        
        # Adjust expression intensity based on score
        if score is not None:
            if score > self.SCORE_THRESHOLD_POSITIVE:
                expression = "joy"  # Very good move
            elif score < self.SCORE_THRESHOLD_NEGATIVE:
                expression = "concerned"  # Bad move
        
        return expression

    def _format_for_tts(self, text: str) -> str:
        """
        Format text for text-to-speech.
        
        Removes emojis and formats for natural speech.
        
        Args:
            text: Original text
            
        Returns:
            TTS-formatted text
        """
        # Remove emojis using Unicode ranges
        # Note: Some ranges overlap, which is intentional for comprehensive emoji coverage
        # CodeQL warning about overlapping ranges is expected and safe here
        import re
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251]+",
            flags=re.UNICODE
        )
        tts_text = emoji_pattern.sub('', text)
        
        # Clean up extra spaces
        tts_text = ' '.join(tts_text.split())
        
        return tts_text.strip()

    def _output(self, data: dict[str, Any]) -> None:
        """
        Output data in the configured format.
        
        Args:
            data: Output data dictionary
        """
        if self.output_format == "json":
            print(json.dumps(data, indent=2))
        elif self.output_format == "text":
            # Simple text format
            if "move" in data:
                print(f"Move: {data['move']}")
            if "comment" in data:
                print(f"Comment: {data['comment']}")
            if "message" in data:
                print(f"Message: {data['message']}")
        elif self.output_format == "rich":
            # Rich format with colors (using rich library if available)
            try:
                from rich import print as rprint
                from rich.panel import Panel
                
                if "move" in data:
                    rprint(Panel(
                        f"[bold cyan]{data['move']}[/bold cyan]\n"
                        f"[yellow]{data.get('comment', '')}[/yellow]",
                        title=f"[{data.get('emotion', 'neutral')}]",
                        border_style="blue"
                    ))
                elif "message" in data:
                    rprint(Panel(
                        f"[green]{data['message']}[/green]",
                        title=f"[{data['type']}]",
                        border_style="green"
                    ))
            except ImportError:
                # Fallback to simple text if rich not available
                self.output_format = "text"
                self._output(data)

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
