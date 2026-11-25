"""Injects docs/dashboard_data.json into the dashboard template -> docs/index.html"""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
data = (root / "docs" / "dashboard_data.json").read_text()
html = (root / "src" / "dashboard_template.html").read_text().replace("__DATA__", data)
(root / "docs" / "index.html").write_text(html)
print("docs/index.html", len(html) // 1024, "KB")
