"""Small smoke test for CSV ingestion and deterministic scoring (no API required)."""
import unittest
from pathlib import Path
from main import load_leads, deterministic_score

class LeadIntelligenceSmokeTest(unittest.TestCase):
    def test_sample_csv_has_at_least_30_leads(self):
        leads = load_leads(Path("leads.csv"))
        self.assertGreaterEqual(len(leads), 30)

    def test_score_is_valid_and_decision_is_known(self):
        lead = {
            "name": "Test Person", "company": "TestCo", "company_size": "100-200",
            "industry": "Software / SaaS", "source": "Demo request",
            "last_interaction_date": "2026-10-24"
        }
        result = deterministic_score(lead)
        self.assertTrue(1 <= result["score"] <= 10)
        self.assertIn(result["decision"], {"qualified", "rejected", "review"})

if __name__ == "__main__":
    unittest.main()
