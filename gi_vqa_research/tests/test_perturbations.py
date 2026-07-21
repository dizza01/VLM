from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from gi_vqa.perturbations import (
    apply_patch_intervention,
    build_intervention_plan,
    intervention_plan_sha256,
)


class PerturbationTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "patch_fractions": [0.25],
            "deletion_treatments": ["gray", "blur"],
            "insertion_treatments": ["blur"],
            "selection_modes": ["most_salient", "least_salient", "random"],
            "random_repeats": 2,
            "gray_value": 128,
            "blur_radius": 2.0,
        }

    def test_plan_is_deterministic_and_contains_controls(self) -> None:
        values = np.asarray([[0.1, 0.8], [0.4, 0.2]], dtype=np.float32)
        first = build_intervention_plan(
            values,
            item_id="item-1",
            method="attention",
            seed=42,
            config=self._config(),
        )
        second = build_intervention_plan(
            values,
            item_id="item-1",
            method="attention",
            seed=42,
            config=self._config(),
        )
        self.assertEqual(first, second)
        self.assertEqual(intervention_plan_sha256(first), intervention_plan_sha256(second))
        self.assertEqual(len(first), 12)
        self.assertEqual(len({item.intervention_id for item in first}), len(first))
        self.assertEqual(
            {
                (item.operation, item.treatment, item.selection)
                for item in first
            },
            {
                (operation, treatment, selection)
                for operation, treatments in (
                    ("deletion", ("gray", "blur")),
                    ("insertion", ("blur",)),
                )
                for treatment in treatments
                for selection in ("most_salient", "least_salient", "random")
            },
        )
        most_salient = next(
            item
            for item in first
            if item.operation == "deletion"
            and item.treatment == "gray"
            and item.selection == "most_salient"
        )
        self.assertEqual(most_salient.patch_indices, (1,))

    def test_deletion_and_insertion_apply_only_selected_patch(self) -> None:
        values = np.asarray([[0.1, 0.8], [0.4, 0.2]], dtype=np.float32)
        config = self._config()
        config["deletion_treatments"] = ["gray"]
        config["insertion_treatments"] = ["gray"]
        config["selection_modes"] = ["most_salient", "random"]
        config["random_repeats"] = 1
        plan = build_intervention_plan(
            values,
            item_id="item-1",
            method="attention",
            seed=42,
            config=config,
        )
        image = Image.new("RGB", (4, 4), color=(10, 20, 30))
        deletion = next(
            item
            for item in plan
            if item.operation == "deletion" and item.selection == "most_salient"
        )
        insertion = next(
            item
            for item in plan
            if item.operation == "insertion" and item.selection == "most_salient"
        )
        deleted = apply_patch_intervention(image, deletion)
        inserted = apply_patch_intervention(image, insertion)
        self.assertEqual(deleted.getpixel((3, 0)), (128, 128, 128))
        self.assertEqual(deleted.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(inserted.getpixel((3, 0)), (10, 20, 30))
        self.assertEqual(inserted.getpixel((0, 0)), (128, 128, 128))


if __name__ == "__main__":
    unittest.main()
