#pyright: strict

from os.path import dirname

class Constants:
    MODEL_BASE_DIR = dirname(__file__) + "/../../models/"
    BPE_FLCC_CS = MODEL_BASE_DIR + "flcc-cs/flcc.bpe"
    MODEL_FLCC_CS = MODEL_BASE_DIR + "flcc-cs/flcc.model"
    MODEL_LLAMA_39 = MODEL_BASE_DIR + "llama-39m-Q8_0.gguf"


# First time:
# defaults -currentHost read NSGlobalDomain _HIHideMenuBar
# The domain/default pair of (kCFPreferencesAnyApplication, _HIHideMenuBar) does not exist

# defaults -currentHost write NSGlobalDomain _HIHideMenuBar -int 0

# defaults -currentHost write NSGlobalDomain _HIHideMenuBar -int 1

# osascript -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to false'

# defaults -currentHost delete NSGlobalDomain _HIHideMenuBar