import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from soccer_bev_pipeline import (
    AgentType,
    CANONICAL_REQUIRED_COLUMNS,
    MetricaAdapter,
    MetricaParser,
    ParsedTrackingClip,
    SkillCornerAdapter,
    analyze_clip_events,
    normalize_coordinates_inplace,
)


class AdapterTests(unittest.TestCase):
    def test_metrica_adapter_produces_canonical_columns_and_ball(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home_path, away_path = self._write_metrica_pair(Path(tmpdir))
            adapter = MetricaAdapter(home_path, away_path)
            canonical = adapter.load_canonical()

            self.assertTrue(set(CANONICAL_REQUIRED_COLUMNS).issubset(canonical.long_df.columns))
            ball_ids = canonical.long_df.loc[
                canonical.long_df["agent_type"] == "ball", "agent_id"
            ].unique()
            self.assertEqual(["ball"], list(ball_ids))

    def test_metrica_parser_order_and_interpolation_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home_path, away_path = self._write_metrica_pair(Path(tmpdir))

            parser_a = MetricaParser(home_path, away_path)
            parser_a.load()
            clip_a = parser_a.parse_clip(start_frame=1, num_frames=3, fps=25)

            parser_b = MetricaParser(home_path, away_path)
            parser_b.load()
            clip_b = parser_b.parse_clip(start_frame=1, num_frames=3, fps=25)

            self.assertEqual(clip_a.agent_ids, clip_b.agent_ids)
            self.assertEqual((3, 3, 2), clip_a.coords_xy.shape)
            self.assertFalse(pd.isna(clip_a.coords_xy).any())

    def test_skillcorner_adapter_fails_fast_on_unsupported_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "skillcorner_bad.json"
            bad_path.write_text(json.dumps({"tracking": []}), encoding="utf-8")

            adapter = SkillCornerAdapter(bad_path)
            with self.assertRaisesRegex(ValueError, "unsupported schema|unsupported schema 'skillcorner_tracking_v2'"):
                adapter.load_canonical()

    def test_coordinate_normalization_supports_centered_ranges(self) -> None:
        df = pd.DataFrame(
            {
                "frame_id": [1, 1],
                "agent_id": ["home_1", "ball"],
                "agent_type": ["home", "ball"],
                "x": [-52.5, 0.0],
                "y": [-34.0, 0.0],
            }
        )
        normalize_coordinates_inplace(df, pitch_length_m=105.0, pitch_width_m=68.0, source_name="test")
        self.assertAlmostEqual(0.0, float(df.loc[0, "x"]), places=4)
        self.assertAlmostEqual(0.0, float(df.loc[0, "y"]), places=4)
        self.assertAlmostEqual(52.5, float(df.loc[1, "x"]), places=4)
        self.assertAlmostEqual(34.0, float(df.loc[1, "y"]), places=4)

    def test_clip_analysis_counts_stable_passes_and_possession_changes(self) -> None:
        clip, coords = self._build_analysis_clip(
            stable_teams=[
                AgentType.HOME.value,
                AgentType.HOME.value,
                AgentType.HOME.value,
                AgentType.HOME.value,
                AgentType.HOME.value,
                AgentType.HOME.value,
                AgentType.HOME.value,
                AgentType.AWAY.value,
                AgentType.AWAY.value,
                AgentType.AWAY.value,
            ],
            carrier_ids=[
                "home_1",
                "home_1",
                "home_1",
                "home_1",
                "home_2",
                "home_2",
                "home_2",
                "away_1",
                "away_1",
                "away_1",
            ],
        )

        analysis = analyze_clip_events(clip, coords)

        self.assertEqual(AgentType.HOME.value, analysis.dominant_team)
        self.assertAlmostEqual(0.7, analysis.dominant_team_poss_ratio, places=4)
        self.assertEqual(1, analysis.pass_count)
        self.assertEqual(1, analysis.possession_changes)
        self.assertEqual([0.7], analysis.possession_change_times_s)

    def test_clip_analysis_ignores_single_frame_carrier_jitter(self) -> None:
        clip, coords = self._build_analysis_clip(
            stable_teams=[AgentType.HOME.value] * 8,
            carrier_ids=[
                "home_1",
                "home_1",
                "home_1",
                "home_2",
                "home_1",
                "home_1",
                "home_1",
                "home_1",
            ],
        )

        analysis = analyze_clip_events(clip, coords)

        self.assertEqual(0, analysis.pass_count)
        self.assertEqual(0, analysis.possession_changes)
        self.assertAlmostEqual(1.0, analysis.dominant_team_poss_ratio, places=4)

    @staticmethod
    def _write_metrica_pair(tmpdir: Path) -> tuple[Path, Path]:
        home_df = pd.DataFrame(
            {
                "Frame": [1, 2, 3],
                "Period": [1, 1, 1],
                "Time [s]": [0.00, 0.04, 0.08],
                "Player1_x": [0.10, None, 0.30],
                "Player1_y": [0.50, 0.50, 0.50],
                "Ball_x": [0.11, 0.20, 0.31],
                "Ball_y": [0.50, 0.50, 0.50],
            }
        )
        away_df = pd.DataFrame(
            {
                "Frame": [1, 2, 3],
                "Period": [1, 1, 1],
                "Time [s]": [0.00, 0.04, 0.08],
                "Player7_x": [0.85, 0.82, 0.80],
                "Player7_y": [0.52, 0.52, 0.52],
            }
        )

        home_path = tmpdir / "home.csv"
        away_path = tmpdir / "away.csv"
        home_df.to_csv(home_path, index=False)
        away_df.to_csv(away_path, index=False)
        return home_path, away_path

    @staticmethod
    def _build_analysis_clip(
        stable_teams: list[str],
        carrier_ids: list[str],
    ) -> tuple[ParsedTrackingClip, np.ndarray]:
        if len(stable_teams) != len(carrier_ids):
            raise ValueError("stable_teams and carrier_ids must have equal length.")

        agent_ids = ["home_1", "home_2", "away_1", "away_2", "ball"]
        agent_types = np.array(
            [
                AgentType.HOME,
                AgentType.HOME,
                AgentType.AWAY,
                AgentType.AWAY,
                AgentType.BALL,
            ],
            dtype=object,
        )

        carrier_positions = {
            "home_1": np.array([20.0, 20.0], dtype=np.float32),
            "home_2": np.array([40.0, 20.0], dtype=np.float32),
            "away_1": np.array([60.0, 20.0], dtype=np.float32),
            "away_2": np.array([80.0, 20.0], dtype=np.float32),
        }

        num_frames = len(stable_teams)
        coords = np.full((num_frames, len(agent_ids), 2), np.nan, dtype=np.float32)
        player_order = ["home_1", "home_2", "away_1", "away_2"]

        for t in range(num_frames):
            for idx, player_id in enumerate(player_order):
                coords[t, idx, :] = carrier_positions[player_id]
            coords[t, 4, :] = carrier_positions[carrier_ids[t]]

        clip = ParsedTrackingClip(
            coords_xy=coords,
            agent_ids=agent_ids,
            agent_types=agent_types,
            frame_ids=np.arange(num_frames, dtype=np.int32),
            fps=10,
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            possession_team_by_frame=np.array(stable_teams, dtype=object),
        )
        return clip, coords


if __name__ == "__main__":
    unittest.main()
