import unittest
import argparse
from collections import Counter

from src.data.dataset import limit_samples_per_class, split_official_train


class DatasetSamplingTests(unittest.TestCase):
    samples_per_class = 2
    seed = 42
    class_count = 3
    records_per_class = 4

    def setUp(self):
        self.training_records = [
            {"image_id": f"{label}-{index}", "label": label}
            for label in range(self.class_count)
            for index in range(self.records_per_class)
        ]

    def test_returns_exact_balanced_count_per_class(self):
        sampled = limit_samples_per_class(
            self.training_records,
            samples_per_class=self.samples_per_class,
            seed=self.seed,
        )

        expected_total = self.samples_per_class * self.class_count
        self.assertEqual(len(sampled), expected_total)
        expected_counts = Counter(
            {label: self.samples_per_class for label in range(self.class_count)}
        )
        self.assertEqual(
            Counter(record["label"] for record in sampled),
            expected_counts,
        )
        print(
            f"PASS balanced subset: {len(sampled)} records, "
            f"counts={dict(expected_counts)}"
        )

    def test_same_seed_returns_same_subset(self):
        first = limit_samples_per_class(
            self.training_records,
            samples_per_class=self.samples_per_class,
            seed=self.seed,
        )
        second = limit_samples_per_class(
            self.training_records,
            samples_per_class=self.samples_per_class,
            seed=self.seed,
        )

        self.assertEqual(
            [record["image_id"] for record in first],
            [record["image_id"] for record in second],
        )
        print(f"PASS reproducibility: seed={self.seed}")

    def test_insufficient_class_raises_value_error(self):
        records = [
            {"image_id": "0-0", "label": 0},
            {"image_id": "1-0", "label": 1},
            {"image_id": "1-1", "label": 1},
        ]

        with self.assertRaisesRegex(ValueError, "class 0 has only 1"):
            limit_samples_per_class(
                records,
                samples_per_class=2,
                seed=self.seed,
            )

    def test_validation_and_test_splits_remain_fixed(self):
        records = [
            {
                "image_id": f"{label}-{index}",
                "label": label,
                "is_train": index < 4,
            }
            for label in range(3)
            for index in range(6)
        ]

        train_records, validation_records, test_records = split_official_train(
            records,
            val_split=0.25,
            seed=self.seed,
        )
        larger_train_records, larger_validation_records, larger_test_records = (
            split_official_train(
                records,
                val_split=0.25,
                seed=self.seed,
            )
        )
        small_subset = limit_samples_per_class(
            train_records,
            samples_per_class=1,
            seed=42,
        )
        larger_subset = limit_samples_per_class(
            larger_train_records,
            samples_per_class=self.samples_per_class,
            seed=self.seed,
        )

        self.assertEqual(len(small_subset), 3)
        self.assertEqual(
            len(larger_subset),
            self.samples_per_class * self.class_count,
        )
        self.assertEqual(len(validation_records), 3)
        self.assertEqual(len(test_records), 6)
        self.assertEqual(
            [record["image_id"] for record in validation_records],
            [record["image_id"] for record in larger_validation_records],
        )
        self.assertEqual(
            [record["image_id"] for record in test_records],
            [record["image_id"] for record in larger_test_records],
        )
        print(
            "PASS fixed holdouts: "
            f"validation={len(validation_records)}, test={len(test_records)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test reproducible balanced training-data sampling."
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="Number of records requested for each class (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the sampler (default: 42).",
    )
    args, unittest_args = parser.parse_known_args()

    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be greater than zero.")

    if args.samples_per_class > 4:
        parser.error(
            "--samples-per-class cannot exceed 4 for this synthetic test data."
        )

    DatasetSamplingTests.samples_per_class = args.samples_per_class
    DatasetSamplingTests.seed = args.seed

    print(
        "Sampling test configuration: "
        f"samples_per_class={args.samples_per_class}, seed={args.seed}, "
        "classes=3, records_per_class=4"
    )
    unittest.main(argv=[__file__, *unittest_args], verbosity=2)