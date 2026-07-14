"""
Human evaluation component: collect subjective ratings for retrieved music.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from datetime import datetime


class HumanEvaluation:
    """
    Manage human evaluation of search quality.
    Stores ratings and computes average scores.
    """

    def __init__(self, save_path: str = "experiments/human_eval.json"):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict:
        """Load existing human evaluation data."""
        if self.save_path.exists():
            with open(self.save_path, 'r') as f:
                return json.load(f)
        return {"ratings": [], "queries": [], "results": []}

    def _save(self):
        """Save current data."""
        with open(self.save_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

    def add_rating(
        self,
        query_id: str,
        query_description: str,
        retrieved_tracks: List[Dict],
        user_ratings: List[int]  # list of scores (1-5) for each retrieved track
    ):
        """
        Add a human rating session.
        user_ratings length must match retrieved_tracks length.
        """
        if len(user_ratings) != len(retrieved_tracks):
            raise ValueError("Number of ratings must match number of retrieved tracks.")

        entry = {
            "query_id": query_id,
            "query_description": query_description,
            "retrieved_tracks": retrieved_tracks,
            "user_ratings": user_ratings,
            "timestamp": datetime.now().isoformat()
        }
        self.data["ratings"].append(entry)
        self._save()

    def add_query(self, query_id: str, query_description: str, query_type: str = "audio"):
        """Register a query for future rating."""
        entry = {
            "query_id": query_id,
            "description": query_description,
            "type": query_type,
            "timestamp": datetime.now().isoformat()
        }
        self.data["queries"].append(entry)
        self._save()

    def average_rating(self) -> float:
        """Compute mean rating across all judged tracks."""
        all_ratings = []
        for entry in self.data["ratings"]:
            all_ratings.extend(entry["user_ratings"])
        return np.mean(all_ratings) if all_ratings else 0.0

    def per_query_average(self) -> Dict[str, float]:
        """Average rating per query."""
        avg = {}
        for entry in self.data["ratings"]:
            qid = entry["query_id"]
            avg[qid] = np.mean(entry["user_ratings"])
        return avg

    def per_track_average(self) -> Dict[str, float]:
        """Average rating per track (by track ID)."""
        track_ratings = {}
        for entry in self.data["ratings"]:
            for track, rating in zip(entry["retrieved_tracks"], entry["user_ratings"]):
                track_id = track.get("track_id") or track.get("id")
                if track_id is not None:
                    if track_id not in track_ratings:
                        track_ratings[track_id] = []
                    track_ratings[track_id].append(rating)
        return {tid: np.mean(ratings) for tid, ratings in track_ratings.items()}

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "num_queries_rated": len(self.data["ratings"]),
            "total_judgments": sum(len(e["user_ratings"]) for e in self.data["ratings"]),
            "mean_rating": self.average_rating(),
            "per_query": self.per_query_average(),
            "per_track": self.per_track_average(),
        }

    def export_to_csv(self, csv_path: str = "experiments/human_eval.csv"):
        """Export ratings to CSV format for further analysis."""
        import csv
        rows = []
        for entry in self.data["ratings"]:
            for track, rating in zip(entry["retrieved_tracks"], entry["user_ratings"]):
                row = {
                    "query_id": entry["query_id"],
                    "query_description": entry["query_description"],
                    "track_id": track.get("track_id") or track.get("id"),
                    "track_title": track.get("metadata", {}).get("title", ""),
                    "track_artist": track.get("metadata", {}).get("artist", ""),
                    "rating": rating,
                    "timestamp": entry["timestamp"]
                }
                rows.append(row)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
            writer.writeheader()
            writer.writerows(rows)
