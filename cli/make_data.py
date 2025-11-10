import argparse

from config import load_make_data_config
from diffusionlm.tokenizer import Tokenizer
from databuilder.dataset_builder import write_token_ids_to_memmap
from cli.utils import add_config_args, load_config_or_print
from logger import ConsoleLogger


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Make data via config file only.", allow_abbrev=False)
    add_config_args(parser, type_=str)
    return parser.parse_args()


def main():
    args_cfg = _parse_only_config()
    cfg_dc = load_config_or_print(load_make_data_config, args_cfg.config, args_cfg.print_config)
    if cfg_dc is None:
        return

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(cfg_dc.tokenizer.vocab_path),
        merges_filepath=str(cfg_dc.tokenizer.merges_path),
        special_tokens=cfg_dc.tokenizer.special_tokens,
    )

    logger = ConsoleLogger()
    write_token_ids_to_memmap(
        tokenizer,
        str(cfg_dc.input.input_filename),
        cfg_dc.input.total_tokens,
        str(cfg_dc.output.output_filename),
        logger=logger,
    )


if __name__ == "__main__":
    main()
