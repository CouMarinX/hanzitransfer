import argparse

from .generator import COMMON_RADICALS, generate_dataset, save_synth_results


def parse_range(start: str) -> int:
    """Parse integer from decimal or hex string."""
    return int(start, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Hanzi dataset")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--start", type=parse_range, default=0x4E00, help="Start Unicode code point")
    parser.add_argument("--end", type=parse_range, default=0x9E00, help="End Unicode code point (exclusive)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of characters")
    parser.add_argument(
        "--radicals",
        nargs="*",
        default=COMMON_RADICALS,
        help="Radicals to combine",
    )
    args = parser.parse_args()

    chars = [chr(i) for i in range(args.start, args.end)]
    base_arrays, target_arrays = generate_dataset(args.radicals, chars, args.limit)
    save_synth_results(base_arrays, target_arrays, args.output)
    print(f"Generated {len(target_arrays)} samples to {args.output}")


if __name__ == "__main__":
    main()
