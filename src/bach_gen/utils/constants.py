"""Constants for the Bach invention generator."""

# Voice ranges (MIDI note numbers)
SOPRANO_RANGE = (60, 84)  # C4 – C6
ALTO_RANGE = (48, 72)     # C3 – C5
TENOR_RANGE = (48, 69)    # C3 – A4
BASS_RANGE = (36, 60)     # C2 – C4

# For 2-part inventions: upper voice ~ soprano/alto, lower voice ~ alto/bass
UPPER_VOICE_RANGE = (55, 84)   # G3 – C6
LOWER_VOICE_RANGE = (36, 67)   # C2 – G4

# Per-voice ranges for each form
# Keys are (form, voice_number) where voice_number is 1-indexed (1=highest)
FORM_VOICE_RANGES: dict[tuple[str, int], tuple[int, int]] = {
    # 2-part invention: two flexible voices
    ("2-part", 1): UPPER_VOICE_RANGE,
    ("2-part", 2): LOWER_VOICE_RANGE,
    # Sinfonia: soprano / alto / bass
    ("sinfonia", 1): SOPRANO_RANGE,
    ("sinfonia", 2): ALTO_RANGE,
    ("sinfonia", 3): BASS_RANGE,
    # Chorale: SATB
    ("chorale", 1): SOPRANO_RANGE,
    ("chorale", 2): ALTO_RANGE,
    ("chorale", 3): TENOR_RANGE,
    ("chorale", 4): BASS_RANGE,
    # Fugue: uses the same ranges as chorale (voice count varies 2-4)
    ("fugue", 1): SOPRANO_RANGE,
    ("fugue", 2): ALTO_RANGE,
    ("fugue", 3): TENOR_RANGE,
    ("fugue", 4): BASS_RANGE,
    # Motet: same as chorale (capped at 4 voices)
    ("motet", 1): SOPRANO_RANGE,
    ("motet", 2): ALTO_RANGE,
    ("motet", 3): TENOR_RANGE,
    ("motet", 4): BASS_RANGE,
}

# Form defaults: (default_num_voices, default_seq_len)
FORM_DEFAULTS: dict[str, tuple[int, int]] = {
    "2-part": (2, 768),
    "invention": (2, 768),
    "sinfonia": (3, 1024),
    "trio_sonata": (3, 1024),
    "chorale": (4, 1024),
    "quartet": (4, 1024),
    "fugue": (4, 2048),
    "motet": (4, 1024),
}

VALID_FORMS = list(FORM_DEFAULTS.keys()) + ["all"]

# Pitch range for tokenization
MIN_PITCH = 36  # C2
MAX_PITCH = 84  # C6
NUM_PITCHES = MAX_PITCH - MIN_PITCH + 1  # 49

# Scale-degree tokenizer octave range (tonic-relative)
SD_MIN_OCTAVE = 2
SD_MAX_OCTAVE = 7

# Timing resolution (ticks per quarter note)
TICKS_PER_QUARTER = 480

# Duration vocabulary: common note durations in ticks
# From 32nd note to dotted whole note
DURATION_BINS = [
    60,    # 32nd note
    120,   # 16th note
    180,   # dotted 16th
    240,   # 8th note
    360,   # dotted 8th
    480,   # quarter note
    720,   # dotted quarter
    960,   # half note
    1440,  # dotted half
    1920,  # whole note
    2880,  # dotted whole
]

# Time shift vocabulary (same bins as duration)
TIME_SHIFT_BINS = DURATION_BINS

# Special token names
SPECIAL_TOKENS = [
    "PAD",
    "BOS",
    "EOS",
    "VOICE_1",
    "VOICE_2",
    "VOICE_3",
    "VOICE_4",
    "SUBJECT_START",
    "SUBJECT_END",
    "BAR",
    "BEAT_1", "BEAT_2", "BEAT_3", "BEAT_4", "BEAT_5", "BEAT_6",
    "MODE_2PART",
    "MODE_3PART",
    "MODE_4PART",
    "MODE_FUGUE",
    "STYLE_BACH",
    "STYLE_BAROQUE",
    "STYLE_RENAISSANCE",
    "STYLE_CLASSICAL",
    "FORM_CHORALE",
    "FORM_INVENTION",
    "FORM_FUGUE",
    "FORM_SINFONIA",
    "FORM_QUARTET",
    "FORM_TRIO_SONATA",
    "FORM_MOTET",
    "LENGTH_SHORT",
    "LENGTH_MEDIUM",
    "LENGTH_LONG",
    "LENGTH_EXTENDED",
    "METER_2_4",
    "METER_3_4",
    "METER_4_4",
    "METER_6_8",
    "METER_3_8",
    "METER_ALLA_BREVE",
]

# Style token names (order matches SPECIAL_TOKENS)
STYLE_NAMES = ["bach", "baroque", "renaissance", "classical"]

# Form token names
FORM_NAMES = ["chorale", "invention", "fugue", "sinfonia", "quartet", "trio_sonata", "motet"]

# Length bucket names and boundaries (in bars/measures)
LENGTH_NAMES = ["short", "medium", "long", "extended"]
LENGTH_BOUNDARIES = [16, 32, 64]  # short ≤16, medium 17-32, long 33-64, extended 65+

# Meter token names and mapping from time signature tuples
METER_NAMES = ["2_4", "3_4", "4_4", "6_8", "3_8", "alla_breve"]
METER_MAP: dict[tuple[int, int], str] = {
    (2, 4): "2_4",
    (3, 4): "3_4",
    (4, 4): "4_4",
    (6, 8): "6_8",
    (3, 8): "3_8",
    (2, 2): "alla_breve",
}

# Map directory name (from kernscores/) to style name
DIR_TO_STYLE: dict[str, str] = {
    # Bach — music21 corpus and kernscores
    "bach": "bach",
    # Baroque
    "buxtehude": "baroque",
    "pachelbel": "baroque",
    "corelli": "baroque",
    "vivaldi": "baroque",
    "frescobaldi": "baroque",
    "monteverdi": "baroque",
    # Renaissance
    "josquin": "renaissance",
    "ockeghem": "renaissance",
    "victoria": "renaissance",
    "lassus": "renaissance",
    "byrd": "renaissance",
    "dufay": "renaissance",
    "dunstable": "renaissance",
    "isaac": "renaissance",
    "banchieri": "renaissance",
    "giovannelli": "renaissance",
    "vecchi": "renaissance",
    "obrecht": "renaissance",
    "desprez": "renaissance",
    "busnois": "renaissance",
    "delarue": "renaissance",
    "martini": "renaissance",
    "agricola": "renaissance",
    "compere": "renaissance",
    "mouton": "renaissance",
    "brumel": "renaissance",
    "regis": "renaissance",
    "tinctoris": "renaissance",
    # Classical
    "haydn": "classical",
    "mozart": "classical",
    "beethoven": "classical",
    "clementi": "classical",
    "scarlatti": "classical",
    "mayer": "classical",
    "saintgeorges": "classical", 
    "hensel": "classical",
    "arriaga": "classical",
    "kalliwoda": "classical",
    "maier": "classical",
    "brahms": "classical",   # close enough for our purposes
    "schubert": "classical",
    "schumann": "classical",
    "cherubini": "classical",
    "hoffmeister": "classical",
}

# Map directory name to form name (for auto-detection)
DIR_TO_FORM: dict[str, str] = {
    # Classical quartets
    "haydn": "quartet",
    "mozart": "quartet",
    "beethoven": "quartet",
    # Renaissance motets
    "josquin": "motet",
    "ockeghem": "motet",
    "victoria": "motet",
    "lassus": "motet",
    "byrd": "motet",
    "dufay": "motet",
    "dunstable": "motet",
    "isaac": "motet",
    "banchieri": "motet",
    "giovannelli": "motet",
    "vecchi": "motet",
    "obrecht": "motet",
    "desprez": "motet",
    "busnois": "motet",
    "delarue": "motet",
    "martini": "motet",
    "agricola": "motet",
    "compere": "motet",
    "mouton": "motet",
    "brumel": "motet",
    "regis": "motet",
    "tinctoris": "motet",
    "mayer": "quartet",
    "saintgeorges": "quartet",
    "hensel": "quartet",
    "arriaga": "quartet",
    "kalliwoda": "quartet",
    "maier": "quartet",
    "brahms": "quartet",
    "schubert": "quartet",
    "schumann": "quartet",
    "cherubini": "quartet",
    "hoffmeister": "quartet",
    "boccherini": "quartet",
}


def bwv_to_form(bwv: int) -> str | None:
    """Return the form name for a Bach BWV number, or None if unknown."""
    if 772 <= bwv <= 786:
        return "invention"
    if 787 <= bwv <= 801:
        return "sinfonia"
    if bwv in set(WTC_BOOK1_FUGUES + WTC_BOOK2_FUGUES) or bwv in {565, 1080}:
        return "fugue"
    if 525 <= bwv <= 530:
        return "trio_sonata"
    if 250 <= bwv <= 438:
        return "chorale"
    return None


# Key names for tokens
KEY_NAMES = [
    "C_major", "C_minor",
    "Db_major", "Cs_minor",
    "D_major", "D_minor",
    "Eb_major", "Eb_minor",
    "E_major", "E_minor",
    "F_major", "F_minor",
    "Fs_major", "Fs_minor",
    "G_major", "G_minor",
    "Ab_major", "Gs_minor",
    "A_major", "A_minor",
    "Bb_major", "Bb_minor",
    "B_major", "B_minor",
]

# Bach invention BWV numbers
TWO_PART_INVENTIONS = list(range(772, 787))  # BWV 772–786
THREE_PART_SINFONIAS = list(range(787, 802))  # BWV 787–801

# Well-Tempered Clavier Book 1: BWV 846-869 (fugues at odd numbers)
WTC_BOOK1_FUGUES = [
    847, 849, 851, 853, 855, 857, 859, 861,
    863, 865, 867, 869,
]
# Well-Tempered Clavier Book 2: BWV 870-893 (fugues at odd numbers)
WTC_BOOK2_FUGUES = [
    871, 873, 875, 877, 879, 881, 883, 885,
    887, 889, 891, 893,
]

# Art of Fugue
ART_OF_FUGUE = list(range(1080, 1081))  # BWV 1080

# Organ trio sonatas (3-voice)
ORGAN_TRIO_SONATAS = list(range(525, 531))  # BWV 525-530

# Keyboard suites
ENGLISH_SUITES = list(range(806, 812))     # BWV 806-811
FRENCH_SUITES = list(range(812, 818))      # BWV 812-817
KEYBOARD_PARTITAS = list(range(825, 831))  # BWV 825-830

# Combined list of all targeted BWV numbers
ALL_TARGETED_BWV = sorted(set(
    TWO_PART_INVENTIONS
    + THREE_PART_SINFONIAS
    + WTC_BOOK1_FUGUES
    + WTC_BOOK2_FUGUES
    + ART_OF_FUGUE
    + ORGAN_TRIO_SONATAS
    + ENGLISH_SUITES
    + FRENCH_SUITES
    + KEYBOARD_PARTITAS
))

def length_bucket(num_bars: int) -> str:
    """Return the length bucket name for a given bar count."""
    for i, boundary in enumerate(LENGTH_BOUNDARIES):
        if num_bars <= boundary:
            return LENGTH_NAMES[i]
    return LENGTH_NAMES[-1]  # extended


def compute_measure_count(
    voices: list[list[tuple[int, int, int]]],
    time_sig: tuple[int, int] = (4, 4),
) -> int:
    """Compute the number of measures from note data.

    Uses the maximum tick span across all voices divided by ticks_per_measure.
    """
    max_tick = 0
    for voice in voices:
        for start, dur, _pitch in voice:
            end = start + dur
            if end > max_tick:
                max_tick = end
    measure_len = ticks_per_measure(time_sig)
    if measure_len <= 0:
        return 0
    # Round up to nearest measure
    return (max_tick + measure_len - 1) // measure_len


# Beat computation helpers

def beats_per_measure(time_sig: tuple[int, int]) -> int:
    """Return number of beats in a measure for the given time signature.

    Compound meters (6/8, 9/8, 12/8) are grouped into dotted-quarter beats.
    """
    numerator, denominator = time_sig
    # Compound meters: numerator divisible by 3 and > 3, with 8th-note denominator
    if numerator > 3 and numerator % 3 == 0 and denominator == 8:
        return numerator // 3
    return numerator


def ticks_per_measure(time_sig: tuple[int, int], tpq: int = TICKS_PER_QUARTER) -> int:
    """Return total ticks in one measure for the given time signature."""
    numerator, denominator = time_sig
    # Each beat = tpq * (4 / denominator) ticks; total = numerator * that
    return numerator * tpq * 4 // denominator


def beat_tick_positions(time_sig: tuple[int, int], tpq: int = TICKS_PER_QUARTER) -> list[int]:
    """Return tick offsets within a measure where each beat starts.

    E.g. 4/4 at tpq=480 → [0, 480, 960, 1440]
         6/8 at tpq=480 → [0, 720]  (compound: 2 dotted-quarter beats)
    """
    n_beats = beats_per_measure(time_sig)
    measure_len = ticks_per_measure(time_sig, tpq)
    beat_len = measure_len // n_beats
    return [i * beat_len for i in range(n_beats)]


# Model defaults
DEFAULT_SEQ_LEN = 2048
DEFAULT_EMBED_DIM = 256
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 8
DEFAULT_FFN_DIM = 1024
DEFAULT_DROPOUT = 0.1

# Training defaults
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_EPOCHS = 200
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LABEL_SMOOTHING = 0.1

# Generation defaults
DEFAULT_NUM_CANDIDATES = 100
DEFAULT_TOP_K_RESULTS = 3
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K_SAMPLING = 40
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_GEN_LENGTH = 1024
