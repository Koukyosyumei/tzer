from tzer import template, fuzz
from tzer.context import Context

from tzer.relay_seeds import MODEL_SEEDS

target_seeds = MODEL_SEEDS[4::]

seed = target_seeds[0]


if __name__ == "__main__":
    ctx = fuzz.make_context(seed)
    ctx.load("/home/koukyosyumei/Dev/tzer/fuzzing-report-b18f9d3c-f98c-40da-b24c-3ceeccf4b45c/InternalError__4fccc14d-0c35-4dbf-a902-3b9007cefbeb.ctx")
    template.execute_both_mode(ctx)