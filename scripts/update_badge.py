import re, pathlib, math

readme_path = pathlib.Path("README.md")
text = readme_path.read_text(encoding="utf-8")

# Count completed vs total week checkboxes under Phase sections
# We look for patterns like "- [ ] **W..**" and "- [x] **W..**"
unchecked = len(re.findall(r"- \[ \] \*\*W\d+", text))
checked = len(re.findall(r"- \[x\] \*\*W\d+", text, flags=re.IGNORECASE))
total = checked + unchecked

percent = 0 if total == 0 else math.floor(100.0 * checked / total)
badge = f"![Progress](https://img.shields.io/badge/Progress-{percent}%25-brightgreen)"

new_text = re.sub(
    r"(<!--PROGRESS_BADGE_START-->)(.*?)(<!--PROGRESS_BADGE_END-->)",
    rf"<!--PROGRESS_BADGE_START-->\n{badge}\n<!--PROGRESS_BADGE_END-->",
    text,
    flags=re.DOTALL
)

if new_text != text:
    readme_path.write_text(new_text, encoding="utf-8")
    print(f"Updated progress badge to {percent}% ({checked}/{total})")
else:
    print("Badge already up to date.")
