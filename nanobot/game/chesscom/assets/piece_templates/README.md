# Chess.com Piece Templates

This directory is for storing piece template images for board recognition.

## Setup Instructions

To enable accurate piece detection on Chess.com, you'll need to capture template images for each piece type in your chosen theme.

### Required Templates

For each piece theme, you need 12 template images (6 white + 6 black pieces):

- `white_pawn.png`
- `white_knight.png`
- `white_bishop.png`
- `white_rook.png`
- `white_queen.png`
- `white_king.png`
- `black_pawn.png`
- `black_knight.png`
- `black_bishop.png`
- `black_rook.png`
- `black_queen.png`
- `black_king.png`

### Capturing Templates

1. Open Chess.com with your desired piece theme
2. Start a game or use the board editor
3. Take screenshots of individual pieces
4. Crop each piece to approximately 60x60 pixels (adjust based on board size)
5. Save with transparent background if possible
6. Name files according to the convention above

### Multiple Themes

To support multiple themes, organize templates in subdirectories:

```
piece_templates/
├── default/
│   ├── white_pawn.png
│   ├── ...
├── neo/
│   ├── white_pawn.png
│   ├── ...
└── wood/
    ├── white_pawn.png
    ├── ...
```

### Template Format

- **Format**: PNG (preferred) or JPG
- **Size**: 60x60 to 80x80 pixels recommended
- **Background**: Transparent or matching the square color
- **Quality**: High resolution for accurate matching

## Alternative: CNN-Based Detection

For more robust detection, consider using a CNN model instead of template matching. The `BoardRecognizer` class can be extended to use a pre-trained model for piece detection.

## Notes

- Template matching works best with consistent lighting and board size
- Consider creating templates for both light and dark squares
- Test templates at different screen resolutions
- Update templates if Chess.com changes their piece designs
