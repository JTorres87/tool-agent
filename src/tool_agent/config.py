MODEL = "claude-haiku-5-5"
FALLBACK_MODEL = "claude-sonnet-5-5"
VERIFIER_MODEL = FALLBACK_MODEL  # checks each final answer; one call per answer
MAX_VERIFIER_REJECTIONS = 2  # after this many, return the answer flagged instead of retrying
MAX_STEPS = 10
MAX_TOKENS = 16000
