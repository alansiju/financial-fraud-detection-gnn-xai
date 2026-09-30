import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import server.config
    import server.database
    import server.auth.utils
    import server.auth.router
    import server.ml.loader
    import server.ml.predictor
    import server.api.dashboard
    import server.api.transactions
    import server.api.predict
    import server.api.explain
    import server.api.graph
    import server.api.analytics
    import server.api.metrics
    import server.api.reports
    import server.api.settings
    import server.main
    print("ALL BACKEND MODULES IMPORTED SUCCESSFULLY")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback; traceback.print_exc()
