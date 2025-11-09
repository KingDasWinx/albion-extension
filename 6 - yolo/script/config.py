# Central config to keep CLI simple
# Adjust these values once and the detector will use them when starting the minigame.

# Which minigame implementation to run: 'v2', 'v1', or 'auto'
MINIGAME_SCRIPT = 'v2'

# Region (X, Y, W, H) used by the minigame.
# Set this to your minigame UI area (progress bar region). Example from your last run:
MINIGAME_REGION = (837, 533, 243, 36)

# Detection threshold inside minigamev2
MINIGAME_THRESHOLD = 0.5

# Processing FPS for the minigame
MINIGAME_FPS = 60

# Show debug UI from the minigame
MINIGAME_DEBUG = False

# Optionally focus a window title before running minigame (or leave as None)
MINIGAME_FOCUS_WINDOW = None

# ---- Lighting Adaptation ----
# Enable normalization (CLAHE + gamma) for dark frames
ENABLE_LIGHT_NORMALIZATION = False
# Brightness threshold (0-255 mean Y) below which a frame is considered dark
LIGHT_BRIGHTNESS_THRESH = 60.0
# CLAHE parameters
LIGHT_CLAHE_CLIP = 2.0
LIGHT_CLAHE_GRID = 8
# Gamma (<1 brightens)
LIGHT_GAMMA_BRIGHTEN = 0.9
# Night bumps: increase bite threshold and consecutive requirement when dark
NIGHT_THRESHOLD_BUMP = 0.15
NIGHT_CONSEC_BUMP = 2
