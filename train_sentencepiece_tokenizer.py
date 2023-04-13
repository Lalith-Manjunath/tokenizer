import sentencepiece as spm
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--vocab_size", type=int, default=128_000)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--num_threads", type=int, default=8)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model_name = args.output_name if args.output_name is not None else f"spm.{int(args.vocab_size/1000)}k"
    # The following arguments must be changed to reflect the downstream tasks.
    spm.SentencePieceTrainer.train(input=args.input,
                                   model_prefix=model_name,
                                   input_format="text",
                                   model_type="bpe",
                                   vocab_size=args.vocab_size,
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   train_extremely_large_corpus=True,
                                   num_threads=args.num_threads
                                   )
    return 0

if __name__ == "__main__":
    main()
